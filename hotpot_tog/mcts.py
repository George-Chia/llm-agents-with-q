import itertools
import numpy as np
from functools import partial
from models import gpt
import wikienv, wrappers
import requests
import logging
import random

import os
import json
import copy
from fschat_templates import prompt_with_icl
import re
from node import *

from critique_templates import auto_j_single_template, template_v1, template_v2, template_huan, hotpot_description


env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="train")
env = wrappers.LoggingWrapper(env)

global reflection_map
global failed_trajectories
reflection_map = []
failed_trajectories = []

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    global reflection_map
    global failed_trajectories
    
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    value_prompt = task.value_prompt_wrap(x, y, unique_trajectories, reflection_map)
    logging.info(f"Current: {x}")
    logging.info(f"Current: {y}")
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    logging.info(f"VALUE PROMPT: {value_prompt}")
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    logging.info(f"VALUE OUTPUTS: {value_outputs}")
    value = task.value_outputs_unwrap(value_outputs)
    logging.info(f"VALUES: {value}")
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    global failed_trajectories
    global reflection_map
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    if len(unique_trajectories) > len(reflection_map) and len(unique_trajectories) < 4:
        print("generating reflections")
        reflection_map = task.generate_self_reflection(unique_trajectories, x)
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, reflection_map)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    logging.info(f"PROMPT: {prompt}")
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def get_unique_trajectories(failed_trajectories, num=5):
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get('final_answer')
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj['trajectory']))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories
    
def save_node_to_json(node, terminal_nodes, idx, trajectories_save_path):
    all_tree_nodes_list = collect_all_nodes(node)
    best_tree_child = max(all_tree_nodes_list, key=lambda x: x.reward)
    best_trajectory_index_list = collect_trajectory_index(best_tree_child)

    # all_tree_terminal_nodes_list = [child for child in all_tree_nodes_list if child.is_terminal==True]
    # rejected_child = min(all_tree_terminal_nodes_list, key=lambda x: x.reward)
    # rejected_trajectory_index_list = collect_trajectory_index(rejected_child)

    task_dict = node.to_dict()
    all_tree_nodes_list.extend(terminal_nodes)
    best_child = max(all_tree_nodes_list, key=lambda x: x.reward)
    task_dict['best reward'] = best_child.reward # contain nodes during rollout
    task_dict['best em'] = best_child.em
    task_dict['best child reward'] = best_tree_child.reward
    task_dict['best child em'] = best_tree_child.em
    task_dict['best_trajectory_index_list'] = best_trajectory_index_list
    task_id = idx
    json.dump(task_dict, open(os.path.join(trajectories_save_path, f"{task_id}.json"), 'w'), indent=4)

#对三元组打分
def triple_scores(triples, question,thought):
    scores = []
    prompt = f"Question: {question}\nThought: {thought}\n"
    for i, triple in enumerate(triples):
        prompt += f"Triple {i + 1}: {triple}\n"
    prompt += "Please evaluate how much each triple helps in answering the question on a scale from 0 to 0.9, where 0 means not helpful at all and 0.9 means very helpful. Provide the scores in square brackets, e.g., [0.8, 0.5, 0.3].\n\nExamples:\n1. Question: What is the capital of France?\nThought: The capital of France is a well-known city.\nTriple 1: (France, capital, Paris)\nTriple 2: (France, largest city, Paris)\nTriple 3: (France, language, French)\nScores: [0.9, 0.8, 0.2]"

    # 调用 GPT 模型获取评分
    score_output = gpt(prompt, n=1, stop=None)[0].strip()

    # 解析评分
    try:
        # 提取方括号中的分数列表
        score_str = score_output.split('[')[1].split(']')[0]
        scores = list(map(float, score_str.split(',')))
        # 确保分数在有效范围内
        scores = [score if 0 <= score <= 0.9 else 0.0 for score in scores]
    except (IndexError, ValueError):
        scores = [0.0] * len(triples)  # 如果无法解析为浮点数，设为0.0
    return scores


def fschat_mcts_search(args, task, idx, iterations=50, to_print=True, trajectories_save_path=None,
                       dpo_policy_model=None, dpo_reference_model=None, tokenizer=None, enable_reflection=False):
    global gpt
    global failed_trajectories
    global reflection_map
    #定义一个字符串，储存每次迭代后已经掌握的信息
    global information_explored
    # 定义一个字符串，储存每次迭代后搜索wiki得到的信息
    global wiki_explored
    information_explored = ""
    wiki_explored = ""
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    global critique_gpt
    if args.expansion_sampling_method == "critique":
        if args.critique_backend:
            if args.critique_temperature == None:
                args.critique_temperature = args.temperature
            critique_gpt = partial(gpt, model=args.critique_backend, temperature=args.critique_temperature)
        else:
            critique_gpt = gpt

    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
    #env.sessions[idx] = {'session': idx, 'page_type': 'init'}
    x = env.reset(idx=idx)
    if to_print:
        print(idx, x)

    # 把中心词传给根节点
    root = Node(state=None, question=x[0], topic_entity=x[1])
    cur_task = x[0]
    if enable_reflection:
        instruction_path = "../prompt/instructions/hotpot_inst_reflection.txt"
        icl_path = "../prompt/icl_examples/hotpot_icl_reflection.json"    
    else:
        instruction_path = "../prompt/instructions/mygraph_inst.txt"
        icl_path = "../prompt/icl_examples/mygraph_icl.json"
    with open(instruction_path) as f:
        instruction = f.read()
    # 文件编码 utf-8
    raw_icl = json.load(open(icl_path, encoding='utf-8'))

    observation, messages = prompt_with_icl(instruction, raw_icl, cur_task, 3)
    root.messages = messages

    #print("ROOTSTATE", root.env_state)
    all_nodes = []
    failed_trajectories = []
    reflection_map = []
    terminal_nodes = []

    for i in range(iterations):
        # print(f"Iteration {i + 1}...")
        node = select_node(root)

        while node is None or (node.is_terminal and node.reward != 1):
            logging.info(f"Need to backtrack or terminal node with reward 0 found at iteration {i + 1}, reselecting...")
            node = select_node(root)
        
        if node is None:
            logging.info("All paths lead to terminal nodes with reward 0. Ending search.")
            break

        if node.is_terminal and node.reward == 1:
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            backpropagate(node, node.reward)
            if args.disable_early_stop:
                continue
            else:
                save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)
                return node.state, node.value, all_nodes, node.reward, node.em
        
        expand_node(node, args, task, args.max_depth)


        while node.is_terminal or not node.children:
            logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root,i)
            expand_node(node, args, task, args.max_depth)

        if args.enable_value_evaluation:
            value = evaluate_node(node, args, task)

        # Find the child with the highest value or UCT? A: similar effect.
        if args.enable_rollout_with_critique:
            reward, terminal_node = rollout_with_critique(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=args.max_depth)
        else:
            reward, terminal_node = rollout_random(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=args.max_depth)

        terminal_nodes.append(terminal_node)

        # if terminal_node.reward == 1:
        #     logging.info("SUCCESSFUL TRAJECTORY FOUND DURING SIMULATION")
        #     return terminal_node.state, terminal_node.value, terminal_node.reward, terminal_node.em
        if args.enable_rollout_early_stop:
            if terminal_node.reward == 1:
                logging.info("Successful trajectory found")
                logging.info(f"Terminal node including rollouts with reward 1 found at iteration {i + 1}")
                backpropagate(terminal_node, reward)
                save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)
                return terminal_node.state, terminal_node.value, terminal_node.reward, terminal_node.em

        backpropagate(terminal_node, reward)
        all_nodes = [(node, node.value) for node in collect_all_nodes(root)]

        # Check for terminal nodes with a reward of 1
        terminal_nodes_with_reward_1 = [node for node in collect_all_nodes(root) if node.is_terminal and node.reward == 1]
        if terminal_nodes_with_reward_1:
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.value)
            if args.disable_early_stop:
                continue
            else:
                save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)
                return best_node.state, best_node.value, best_node.reward, best_node.em
    
        for j, (node, value) in enumerate(all_nodes):
            logging.info(f"Node {j+1}: {str(node)}")

        logging.info(f"State of all_nodes after iteration {i + 1}: {all_nodes}")
    
    all_nodes_list = collect_all_nodes(root)
    for node in all_nodes_list:
        if node.is_terminal and node.value==0:
            backpropagate(node, node.reward)
    save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)

    
    all_nodes_list.extend(terminal_nodes)
    best_child = max(all_nodes_list, key=lambda x: x.reward)
    failed_trajectories = []
    if best_child.reward == 1:
        logging.info("Successful trajectory found")
    else:
        logging.info("Unsuccessful trajectory found")
    if best_child is None:
        best_child = root
    return best_child.state, best_child.value, best_child.reward, best_child.em


def fschat_simple_search(args, task, idx, iterations=30, to_print=True, trajectories_save_path=None, enable_reflection=False):
    global gpt
    global failed_trajectories
    global reflection_map
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
    #env.sessions[idx] = {'session': idx, 'page_type': 'init'}
    x = env.reset(idx=idx)
    if to_print:
        print(idx, x)

    # 把中心词传给根节点
    root = Node(state=None, question=x[0],topic_entity=x[1])
    
    cur_task = x[0]
    if enable_reflection:
        instruction_path = "../prompt/instructions/hotpot_inst_reflection.txt"
        icl_path = "../prompt/icl_examples/hotpot_icl_reflection.json"    
    # 修改 prompt 地址
    else:
        instruction_path = "../prompt/instructions/mygraph_inst.txt"
        icl_path = "../prompt/icl_examples/mygraph_icl.json"
    with open(instruction_path) as f:
        instruction = f.read()
    # 文件编码 utf-8
    raw_icl = json.load(open(icl_path, encoding='utf-8'))
    observation, messages = prompt_with_icl(instruction, raw_icl, cur_task, 3)
    root.messages = messages

    all_nodes = []
    failed_trajectories = []
    terminal_nodes = []
    reflection_map = []
    successful_trajectories = []
    unsuccessful_trajectories = []
    not_finished_trajectories = []
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    for i in range(iterations):
        logging.info(f"Iteration {i + 1}...")
        node = select_node(root)
        depth = 0

        # Perform a simulation from the root
        while not node.is_terminal and depth < args.max_depth:
            expand_node(node, args, task, max_depth=args.max_depth)  # Expand current node
            if not node.children:
                break  # If no child can be generated, break
            node = random.choice(node.children)  # Randomly select a child node
            depth += 1
        # Check the terminal condition
        if node.is_terminal and node.reward == 1:
            logging.info(f"Successful trajectory found in iteration {i + 1}")
            successful_trajectories.append(node)
            break
        elif node.is_terminal and node.reward < 1:
            logging.info(f"Unsuccessful trajectory found in iteration {i + 1}")
            unsuccessful_trajectories.append(node)
        elif not node.is_terminal:
            logging.info(f"Not finished trajectory found in iteration {i + 1}")
            not_finished_trajectories.append(node)
        else:
            raise NotImplementedError

        # Reset the tree (optional)
        # root.children = []

    best_tree_child = node
    # best_trajectory_index_list = collect_trajectory_index(best_tree_child)
    task_dict = root.to_dict()
    task_dict['best child reward'] = best_tree_child.reward
    task_dict['best child em'] = best_tree_child.em
    # task_dict['best_trajectory_index_list'] = best_trajectory_index_list
    task_id = idx
    json.dump(task_dict, open(os.path.join(trajectories_save_path, f"{task_id}.json"), 'w'), indent=4)


    all_nodes = [(node, node.value) for node in collect_all_nodes(root)]

    # Post-process: select the best trajectory
    if successful_trajectories:
        best_node = max(successful_trajectories, key=lambda x: x.reward)
        return best_node.state, best_node.value, best_node.reward, best_node.em
    elif unsuccessful_trajectories:
        best_node = max(unsuccessful_trajectories, key=lambda x: x.reward)
        return best_node.state, best_node.value, best_node.reward, best_node.em
    elif not_finished_trajectories:
        return 0,0,0,0
    

def fschat_beam_search(args, task, idx, to_print=True, trajectories_save_path=None,
                       dpo_policy_model=None, dpo_reference_model=None, tokenizer=None, enable_reflection=False):
    global gpt
    global failed_trajectories
    global reflection_map
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
    #env.sessions[idx] = {'session': idx, 'page_type': 'init'}
    x = env.reset(idx=idx)
    if to_print:
        print(idx, x)

    root = Node(state=None, question=x)
    
    cur_task = x
    if enable_reflection:
        instruction_path = "prompt/instructions/hotpot_inst_reflection.txt"
        icl_path = "prompt/icl_examples/hotpot_icl_reflection.json"    
    else:  
        instruction_path = "prompt/instructions/hotpot_inst.txt"
        icl_path = "prompt/icl_examples/hotpot_icl.json"
    with open(instruction_path) as f:
        instruction = f.read()
    raw_icl = json.load(open(icl_path))
    observation, messages = prompt_with_icl(instruction, raw_icl, cur_task, 3)
    root.messages = messages

    all_nodes = []
    failed_trajectories = []
    terminal_nodes = []
    reflection_map = []
    successful_trajectories = []
    unsuccessful_trajectories = []
    not_finished_trajectories = []
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    # for i in range(iterations):
    # logging.info(f"Iteration {i + 1}...")
    node = select_node(root)
    depth = 0

    # Perform a simulation from the root
    while not node.is_terminal and depth < args.max_depth:
        expand_node(node, args, task, max_depth=args.max_depth)  # Expand current node
        if not node.children:
            break  # If no child can be generated, break
        node = beam_search(node, dpo_policy_model, dpo_reference_model, args.q_model_conv_template, tokenizer)
        depth += 1
    # Check the terminal condition
    if node.is_terminal and node.reward == 1:
        logging.info(f"Successful trajectory found")
        successful_trajectories.append(node)
        # break
    elif node.is_terminal and node.reward < 1:
        logging.info(f"Unsuccessful trajectory found")
        unsuccessful_trajectories.append(node)
    elif not node.is_terminal:
        logging.info(f"Not finished trajectory found")
        not_finished_trajectories.append(node)
    else:
        raise NotImplementedError

    # Reset the tree (optional)
    # root.children = []

    best_tree_child = node
    # best_trajectory_index_list = collect_trajectory_index(best_tree_child)
    task_dict = root.to_dict()
    task_dict['best child reward'] = best_tree_child.reward
    task_dict['best child em'] = best_tree_child.em
    # task_dict['best_trajectory_index_list'] = best_trajectory_index_list
    task_id = idx
    json.dump(task_dict, open(os.path.join(trajectories_save_path, f"{task_id}.json"), 'w'), indent=4)


    all_nodes = [(node, node.value) for node in collect_all_nodes(root)]

    # Post-process: select the best trajectory
    if successful_trajectories:
        best_node = max(successful_trajectories, key=lambda x: x.reward)
        return best_node.state, best_node.value, best_node.reward, best_node.em
    elif unsuccessful_trajectories:
        best_node = max(unsuccessful_trajectories, key=lambda x: x.reward)
        return best_node.state, best_node.value, best_node.reward, best_node.em
    elif not_finished_trajectories:
        return 0,0,0,0

def select_node(node,i = 0):
    #根据wiki选择初始节点
    if node.depth == 0 and i != 0:
        node_children_relation = []
        for child in node.children:
            node_children_relation.append(child.triple)
        score_relation = triple_scores(node_children_relation,node.question,wiki_explored)
        max_score_index = score_relation.index(max(score_relation))
        selected_child = node.children[max_score_index]
        return selected_child

    while node and node.children:
        logging.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
        
        terminal_children = [child for child in node.children if child.is_terminal]
        terminal_status = [child.is_terminal for child in node.children]

        if len(terminal_children) == len(node.children):
            logging.info(f"All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:  
                node.parent.children.remove(node)
            node = node.parent  
            continue  
        
        node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
        if node_with_reward_1:
            logging.info(f"Found terminal node with reward 1 at depth {node.depth}.")
            return node_with_reward_1
        
        node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)

        while node.is_terminal and node.reward != 1:
            node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
            
        logging.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")
        
    return node  # This will return None if all paths from the root are exhausted

#判断需要wiki搜索的内容
def generate_query(question="",myinformation_explored="",mywiki_explored=""):
    global gpt
    prompt = f"""Given the question and the existing information, the existing knowledge triplets, what additional information is needed to answer the question? Please provide a list of keywords (no more than 3) that can be used for further Wikipedia search. Here are some examples:

    Q: Find the person who said "Taste cannot be controlled by law", what did this person die from?
    Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
    existing information: Libya, officially the State of Libya, is a country in the Maghreb region of North Africa.
    Keywords: [Thomas, Jefferson]

    Q: Who is the coach of the team owned by Steve Bisciotti?
    Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
    Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
    Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
    existing information: Libya, officially the State of Libya, is a country in the Maghreb region of North Africa.
    Keywords: [Baltimore Ravens]

    Q: The country with the National Anthem of Bolivia borders which nations?
    Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
    National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
    National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
    UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
    Bolivia, location.country.national_anthem, UnName_Entity
    existing information: Libya, officially the State of Libya, is a country in the Maghreb region of North Africa.
    Keywords: [Bolivia]

    Q: {question}
    Knowledge Triplets: {myinformation_explored}
    existing information: {mywiki_explored}
    Keywords:"""
    response = gpt(prompt, n=1, stop=None)[0]

    # Extract keywords from the response using regular expression
    match = re.search(r'Keywords:\s*\[(.*?)\]', response)
    if match:
        keywords = [keyword.strip() for keyword in match.group(1).split(",")]
    else:
        keywords = []

    return keywords


def expand_node(node, args, task, max_depth):
    n = args.n_generate_sample
    if node.depth >= max_depth:
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        #达到最大深度，判断是否需要wiki
        myenv = wikienv.WikiEnv()
        if node.reward !=1:
            this_time_imformation = ""
            global wiki_explored
            keywords = generate_query(node.question, information_explored,wiki_explored)
            if len(keywords) != 0:
                for keyword in keywords:
                    this_time_imformation += myenv.search_step(keyword)
            wiki_explored = wiki_explored + this_time_imformation
        return

    assert args.expansion_sampling_method == 'vanilla' or args.enable_fastchat_conv  # only fastchat api supports various expansion_sampling_method

    if args.enable_fastchat_conv:
        if args.expansion_sampling_method == 'conditional':
            new_nodes = generate_new_states_conditional_fastchat_conv(node, args, task, n)
        elif args.expansion_sampling_method == 'critique':
            if args.critique_prompt_template == 'auto-j':
                critique_prompt_template = auto_j_single_template
            elif args.critique_prompt_template == 'template_v1':
                critique_prompt_template = template_v1
            elif args.critique_prompt_template == 'template_v2':
                critique_prompt_template = template_v2
            elif args.critique_prompt_template == 'template_huan':
                critique_prompt_template = template_huan.replace('{scenario_description}', hotpot_description)
            else:
                raise NotImplementedError
            new_nodes = generate_new_states_critique_fastchat_conv(node, args, task, n, critique_prompt_template)
        elif args.expansion_sampling_method == 'vanilla':
            new_nodes = generate_new_states_fastchat_conv(node, args, task, n)
    else:
        new_nodes = generate_new_states(node, args, task, n)
    # new_nodes = generate_new_states(node, args, task, args.n_generate_sample)
    node.children.extend(new_nodes)

# Copied by  from ETO-webshop-envs
def parse_action(llm_output: str) -> str:
    llm_output = llm_output.strip()
    try:
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
    except:
        # logging.info("Action Not Found in llm_output: ", llm_output)
        action = 'nothing'
    assert action is not None
    return action

def parse_thought(llm_output: str) -> str:
    llm_output = llm_output.strip()
    llm_output = llm_output.replace("\n"," ")
    try:
        pattern = re.compile(r"Thought: (.*)(?= Action:)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
    except:
        # logging.info("Thought Not Found in llm_output: ", llm_output)
        action = 'nothing'
    assert action is not None
    return action

def rollout(node, args, task, idx, max_depth=7):
    logging.info("ROLLING OUT")
    depth = node.depth
    # n = 5
    n = args.rollout_width
    rewards = [0]
    while not node.is_terminal and depth < max_depth:
        # Generate new states
        logging.info(f"ROLLING OUT {depth}")
        new_states = []
        values = []
        while len(new_states) == 0:
            new_states = generate_new_states(node, args, task, n)

        for state in new_states:
            if state.is_terminal:
                return state.reward, state
                
        child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        #new_state = new_state[0]
        while len(values) == 0:
            values = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
        max_value_index = values.index(max(values))
        rewards.append(max(values))
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            rewards = [-1]
    
    logging.info("ROLLOUT FINISHED")
    return sum(rewards) / len(rewards), node


def rollout_random(node, args, task, idx, max_depth=7):
    depth = node.depth
    n = args.rollout_width
    assert n==1, "large rollout_width is meanless for rollout_random"

    while not node.is_terminal and depth < max_depth:
        # Generate new states
        new_states = []
        values = []
        while len(new_states) == 0:
            if args.enable_fastchat_conv:
                new_states = generate_new_states_fastchat_conv(node, args, task, n)
            else:
                new_states = generate_new_states(node, args, task, n)

        for state in new_states:
            if state.is_terminal:
                return state.reward, state
                
        # child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        # #new_state = new_state[0]
        # while len(values) == 0:
        #     values = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
        
        # max_value_index = values.index(max(values))
        #max_value_index=random.randint(0,len(new_states)-1)
        max_value_index = random.randint(0,len(new_states)-1)
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            node.reward = -1
    return node.reward, node

def rollout_with_critique(node, args, task, idx, max_depth=15):
    depth = node.depth
    n = args.rollout_width
    assert n==1, "the same with rollout_random"
    while not node.is_terminal and depth < max_depth:
        # Generate new states
        new_states = []
        values = []
        while len(new_states) == 0:
            if args.enable_fastchat_conv:
                if args.critique_prompt_template == 'auto-j':
                    critique_prompt_template = auto_j_single_template
                elif args.critique_prompt_template == 'template_v1':
                    critique_prompt_template = template_v1
                elif args.critique_prompt_template == 'template_v2':
                    critique_prompt_template = template_v2
                elif args.critique_prompt_template == 'template_huan':
                    critique_prompt_template = template_huan.replace('{scenario_description}', hotpot_description)
                else:
                    raise NotImplementedError
                new_states_two = generate_new_states_critique_fastchat_conv(node=node, args=args, task=task, 
                                                                        n=2, critique_prompt_template=critique_prompt_template)
                new_states = new_states_two[-1:]
            else:
               raise NotImplementedError

        for state in new_states_two:
            if state.is_terminal:
                return state.reward, state
        node = new_states[0] 
        depth += 1
        if depth == max_depth:
            node.reward = -1
    return node.reward, node


def generate_new_states(node, args, task, n):
    global failed_trajectories
    prompt = generate_prompt(node)
    sampled_actions = get_samples(task, prompt, f"Thought {node.depth + 1}: ", n, prompt_sample=args.prompt_sample, stop="Observation")
    logging.info(f"SAMPLED ACTION: {sampled_actions}")
    tried_actions = []
    
    unique_states = {}  # Store unique states here
    for action in sampled_actions:
        action = action.replace("Thought 1:  Thought 1: ", "Thought 1: ")
        new_state = node.state.copy()  # Make a copy of the parent node's state

        thought_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")), '')
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}"
        
        if unique_key in unique_states:
            continue  # Skip if this state already exists

        tried_actions.append(action_line)
        
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")

            # Update the new state dictionary
            new_state['thought'] = thought_line
            new_state['action'] = action_line
            new_state['observation'] = f"Observation: {obs}"

            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done
            new_node.reward = r
            new_node.depth = node.depth + 1
            if r == 1:
                new_node.em = info.get('em')
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")
            logging.info(f"Feedback: {info}")

            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory_from_bottom(new_node)
                #print(trajectory)
                #if f"{action_type.lower()}[{action_param}]" not in failed_trajectories.values():
                failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})

    return list(unique_states.values())  # Return unique nodes as a list

def get_context(node, conv_template, backend):
    if "gpt-" in backend:
        messages = get_messages_from_bottom(node)
        context =  messages
    elif "Phi-3" in backend or "llama31" in backend or 'auto-j' in backend or 'Llama31-KTO' in backend:
        conv = get_conv_from_bottom(node, conv_template)
        conv.append_message(conv.roles[1], None)
        context =  conv
    else:
        raise NotImplementedError
    return context


# 找出所有候选关系
from tqdm import tqdm
import argparse
from utils import *
from tog.freebase_func import *
import random
from tog.client import *


def select(n,node):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=2048, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=n, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=1, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-4o-mini", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="sk-nPQVAFBDhZoMmYEnPPxYKk0p86jfCMxyQaqnCLV5qKq0XHxK",
                        help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=1, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="bm25", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    args = parser.parse_args()
    question = node.question
    topic_entity = node.topic_entity


    pre_relations = []
    pre_heads = [-1] * len(topic_entity)
    flag_printed = False
    search_depth = 1

    current_entity_relations_list = []
    i = 0
    for entity in topic_entity:
        if entity != "[FINISH_ID]":
            retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations,
                                                                   pre_heads[i], question, args)
            current_entity_relations_list.extend(retrieve_relations_with_scores)
        i += 1
    total_candidates = []
    total_scores = []
    total_relations = []
    total_entities_id = []
    total_topic_entities = []
    total_head = []
    num_relation = 0

    for entity in current_entity_relations_list:
        if entity['head']:
            my_entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
        else:
            my_entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)

        entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in my_entity_candidates_id]
        indices_to_remove_entity = []
        for i in range(len(entity_candidates)):
            if node.depth>0:
                if entity_candidates[i] == 'UnName_Entity' or my_entity_candidates_id[i] == entity['entity'] or entity_candidates[i] == str(node.parent.topic_entity.keys()):
                    indices_to_remove_entity.append(i)
            else:
                if entity_candidates[i] == 'UnName_Entity' or my_entity_candidates_id[i] == entity['entity']:
                    indices_to_remove_entity.append(i)
        for i in sorted(indices_to_remove_entity, reverse=True):
            del my_entity_candidates_id[i]

        if len(my_entity_candidates_id) == 0:
            continue
        else:
            #entity_candidates_id = random.sample(my_entity_candidates_id, args.num_retain_entity)
            # 保留全部实体
            entity_candidates_id = my_entity_candidates_id

        # 修改，不需要对实体的打分
        scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'],entity['relation'], args)
        total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations,total_entities_id, total_topic_entities, total_head)

    flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations,total_candidates,total_topic_entities, total_head,total_scores, args)
    #self.cluster_chain_of_entities.append(chain_of_entities)
    #移出 unname
    indices_to_remove = []
    for i in range(len(total_entities_id)):
        if total_candidates[i] == 'UnName_Entity':
            indices_to_remove.append(i)
    for i in sorted(indices_to_remove, reverse=True):
        del total_entities_id[i]
        del total_relations[i]
        del total_candidates[i]
        del total_scores[i]

    mychain_of_entities = []
    for i in range(len(total_relations)):
        mychain = [topic_entity[total_topic_entities[i]],total_relations[i],total_candidates[i]]
        mychain_of_entities.append(mychain)

    node.entity_list = entities_id
    new_topic_entity = []
    for i in range(len(total_candidates)):
        new_topic_entity.append({})
        new_topic_entity[i][total_entities_id[i]] = total_candidates[i]
    #node.topic_entity = new_topic_entity
    # 对整个三元组打分，select top N
    thought = str(node.state)
    total_scores = triple_scores(mychain_of_entities,question,thought)
    # 找到 N 个最高分的位置
    top_n_indices = sorted(range(len(total_scores)), key=lambda i: total_scores[i], reverse=True)[:n]

    # 根据这些位置重新排列 mychain_of_entities, new_topic_entity, total_relations, total_scores
    mychain_of_entities = [mychain_of_entities[i] for i in top_n_indices]
    new_topic_entity = [new_topic_entity[i] for i in top_n_indices]
    total_relations = [total_relations[i] for i in top_n_indices]
    total_scores = [total_scores[i] for i in top_n_indices]
    return total_relations,new_topic_entity ,mychain_of_entities,total_scores

#直接生成所有的新状态
def generate_new_states_fastchat_conv(node, args, task, n):
    global failed_trajectories

    context = get_context(node, args.conv_template, args.backend)
    tried_actions = []

    unique_states = {}

    # 扩展的关系
    current_entity_relations_list = []
    # 扩展的实体
    current_entity_list = []
    # 扩展的三元组
    current_chain_list = []
    total_scores = []
    current_entity_relations_list, current_entity_list, current_chain_list, total_scores = select(n, node)
    #扩展的节点数
    i = len(current_entity_relations_list)
    #图谱中没找到信息
    if i == 0:
        action_type = "Finish[]"
        action_param = "I don't found enough information"
        new_state = node.state.copy()

        if action_type.startswith("Finish[") and action_type.endswith("]"):
            # 把当前节点传进环境
            env.env.env.node = node
            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
            # Update the new state dictionary
            # new_state['thought'] = thought_line
            new_state['action'] = f"Thought: {action_param} Action: {action_type}"
            new_state['observation'] = f"Observation: {action_param}"
            unique_key = f"{action_param}::{action_type}"

            # 新节点的topic_entity即父节点的关系剪枝后的entity
            new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity)
            new_node.is_terminal = True
            #new_node.reward = -1
            new_node.depth = node.depth + 1
            if r == 1:
                new_node.em = info.get('em')
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")
            logging.info(f"Feedback: {info}")

            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory_from_bottom(new_node)
                # print(trajectory)
                # if f"{action_type.lower()}[{action_param}]" not in failed_trajectories.values():
                failed_trajectories.append(
                    {'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})
        return list(unique_states.values())
    response_list = gpt(context, n=i, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)
    thought_lines = [parse_thought(response) for response in copy.deepcopy(response_list)]
    action_lines = [parse_action(response) for response in copy.deepcopy(response_list)]
    # sampled_actions = response_list

    logging.info(f"SAMPLED ACTION: {action_lines}")
      # Store unique states here

    #处理图谱扩展
    for thought_line, action_line, current_entity_relation, current_entity, current_chain ,current_score in zip(thought_lines, action_lines,current_entity_relations_list,current_entity_list,current_chain_list,total_scores):
        new_state = node.state.copy()  # Make a copy of the parent node's state

        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}::{current_entity_relation}"
        
        if unique_key in unique_states:
            continue  # Skip if this state already exists

        tried_actions.append(action_line)
        
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

            if action_type.startswith("Finish[") and action_type.endswith("]"):
                # 把当前节点传进环境
                env.env.env.node = node
                obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
                # Update the new state dictionary
                # new_state['thought'] = thought_line
                new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
                new_state['observation'] = f"Observation: {obs}"

                # 新节点的topic_entity即父节点的关系剪枝后的entity
                new_node = Node(state=new_state, question=node.question, parent=node,topic_entity=node.topic_entity)
                new_node.is_terminal = True
                new_node.reward = 1
                new_node.depth = node.depth + 1
                if r == 1:
                    new_node.em = info.get('em')
                unique_states[unique_key] = new_node  # Add this state to unique_states
                logging.info(f"NEW NODE: {new_node}")
                logging.info(f"Feedback: {info}")

                if new_node.is_terminal and r == 0:
                    trajectory = collect_trajectory_from_bottom(new_node)
                    # print(trajectory)
                    # if f"{action_type.lower()}[{action_param}]" not in failed_trajectories.values():
                    failed_trajectories.append(
                        {'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})
            else:
                #直接扩展节点
                select_relation = current_entity_relation
                obs = f"Knowledge Triplets:  {current_chain}\n"
                new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
                new_state['observation'] = f"Observation: {obs}"
                new_node = Node(state=new_state, question=node.question, parent=node,topic_entity=current_entity)
                new_node.is_terminal = False
                new_node.triple = node.triple + str(current_chain)
                if current_score>0.9:
                    new_node.value = 0.9
                else:
                    new_node.value = current_score
                new_node.depth = node.depth + 1
                unique_states[unique_key] = new_node
                logging.info(f"NEW NODE: {new_node}")
                info = select_relation
                logging.info(f"Feedback: {info}")

    return list(unique_states.values())  # Return unique nodes as a list


    
def generate_new_states_conditional_fastchat_conv(node, args, task, n):
    global failed_trajectories
    assert args.enable_fastchat_conv

    sampled_response_list = []
    sampled_obs_list = []

    unique_states = {}  # Store unique states here

    for sampling_index in range(n):
        context = get_context(node, args.conv_template, args.backend)
        if len(sampled_response_list) > 0:
            original_observation = context.messages[-2][1]
            conditional_context = '\n\nBelow are the potential actions you might generate along with their corresponding environmental feedback: \n\n'
            for sampled_response,sampled_obs in zip(sampled_response_list, sampled_obs_list):
                conditional_context += sampled_response + "\n"
                conditional_context += 'Observation:'+'\n'+sampled_obs + "\n\n"
            conditional_context += 'Please summarize insights from the potential actions and feedback, and generate a new response with as much distinctiveness as possible for the Observation: \n'
            conditional_context += original_observation
            context.messages[-2][1] += conditional_context + "\n"
        response = gpt(context, n=1, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)[0]
        sampled_response_list.append(response)

        thought_line = parse_thought(response) 
        action_line = parse_action(response)
        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}"


        new_state = node.state.copy()  # Make a copy of the parent node's state

        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}"
        
        if unique_key in unique_states:
            continue  # Skip if this state already exists

        # tried_actions.append(action_line)
        
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")

            # Update the new state dictionary
            # new_state['thought'] = thought_line
            new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
            new_state['observation'] = f"Observation: {obs}"

            sampled_obs_list.append(f"Observation: {obs}")

            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done
            new_node.reward = r
            new_node.depth = node.depth + 1
            if r == 1:
                new_node.em = info.get('em')
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")
            logging.info(f"Feedback: {info}")

            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory_from_bottom(new_node)
                #print(trajectory)
                #if f"{action_type.lower()}[{action_param}]" not in failed_trajectories.values():
                failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})

    return list(unique_states.values())  # Return unique nodes as a list


def get_raw_observation(text):
    keyword = '\n\nBelow are the previous Thought and Action you generated along with their corresponding Observation:'
    index = text.find(keyword)
    if index != -1:
        return text[:index]
    else:
        return text

def get_historical_context(context):
    prompt = ''
    for message in context.messages[25:]: # todo
        if message[1] is not None:
            prompt += message[1].strip() + '\n'
    return prompt

def generate_new_states_critique_fastchat_conv(node, args, task, n, critique_prompt_template):
    global failed_trajectories
    assert args.enable_fastchat_conv

    previous_response = None
    previous_obs = None

    unique_states = {}  # Store unique states here

    for sampling_index in range(n):
        context = copy.deepcopy(get_context(node, args.conv_template, args.backend))
        critique = None
        regenerate_prompt = None
        if previous_response:
            # generating critique
            if not args.critique_backend:
                args.critique_backend = args.backend
            if not args.critique_conv_template:
                args.critique_conv_template = args.conv_template
            critique_context = copy.deepcopy(get_context(node, args.critique_conv_template, args.critique_backend))
            # generating critique
            if args.critique_prompt_template == 'template_huan':
                original_observation = get_raw_observation(context.messages[-2][1])
                critique_prompt_templat = critique_prompt_template.format(
                    user_inst=critique_context.messages[-2][1],
                    historical_context=get_historical_context(critique_context),
                    current_state = previous_response + '\n' + previous_obs
                )
                if 'gpt' in args.critique_backend:
                    raise NotImplementedError
                else:
                    critique_context.messages = [['system', critique_prompt_templat.split('</system>')[0]],
                                                ['user', critique_prompt_templat.split('</system>')[-1]],
                                                ['assistant', None]]
                critique = critique_gpt(critique_context, n=1, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)[0]
                if critique.startswith('Critique:'):
                    critique = critique[9:]
                regenerate_prompt = '\n\nBelow are the previous Thought and Action you generated along with their corresponding Observation: \n\n'
                regenerate_prompt += previous_response + "\n"
                regenerate_prompt += previous_obs + "\n"
                regenerate_prompt += 'Critique: ' + critique + "\n\n"
                regenerate_prompt += 'Based on the critique, generate a new Thought and Action with as much distinctiveness as possible for the Observation:' + "\n"
                context.messages[-2][1] += regenerate_prompt + "\n" + original_observation
            else:
                critique_prompt = critique_prompt_template.format(previous_response=previous_response, previous_obs=previous_obs)
                if isinstance(critique_context, list):  # for openai GPT
                    original_observation = get_raw_observation(critique_context[-1]['content'])
                    critique_context[-1]['content'] += critique_prompt + "\n"
                else: # for fastchat
                    original_observation = get_raw_observation(critique_context.messages[-2][1])
                    critique_context.messages[-2][1] += critique_prompt + "\n"
                critique = critique_gpt(critique_context, n=1, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)[0]
                # generating thought and action
                regenerate_prompt = '\n\nBelow are the previous Thought and Action you generated along with their corresponding Observation: \n\n'
                regenerate_prompt += previous_response + "\n"
                regenerate_prompt += previous_obs + "\n"
                regenerate_prompt += 'Critique: '+critique + "\n\n"
                regenerate_prompt += 'Based on the feedback, generate a new Thought and Action with as much distinctiveness as possible for the Observation:'+ "\n"
                context.messages[-2][1] += regenerate_prompt + "\n" + original_observation
            # critique_prompt = '\n\nBelow are the previous Thought and Action you generated along with their corresponding Observation: \n\n'
            # critique_prompt += previous_response + "\n"
            # critique_prompt += previous_obs + "\n\n"
            # critique_prompt += 'Review the previous Thought, Action, and Observation. Your role is to determine whether the action is effective for completing the task, and provide specific and constructive feedback. Please output feedback directly. \nFormat\nFeedback:[[Feedback]]'

            # if 'auto-j' in args.critique_backend:
            #     critique_prompt = auto_j_single_template.format(previous_response=previous_response, previous_obs=previous_obs)
            # else:

        response = gpt(context, n=1, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)[0]
        previous_response = response

        # if len(sampled_response_list) > 0:
        #     original_observation = context.messages[-2][1]
        #     conditional_context = '\n\nBelow are the potential actions you might generate along with their corresponding environmental feedback: \n\n'
        #     for sampled_response,sampled_obs in zip(sampled_response_list, sampled_obs_list):
        #         conditional_context += sampled_response + "\n"
        #         conditional_context += 'Observation:'+'\n'+sampled_obs + "\n\n"
        #     conditional_context += 'Please summarize insights from the potential actions and feedback, and generate a new response with as much distinctiveness as possible for the Observation: \n'
        #     conditional_context += original_observation
        #     context.messages[-2][1] += conditional_context + "\n"
        # response = gpt(context, n=1, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)[0]
        # sampled_response_list.append(response)

        thought_line = parse_thought(response) 
        action_line = parse_action(response)
        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}"


        new_state = node.state.copy()  # Make a copy of the parent node's state

        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}"
        
        if unique_key in unique_states:
            continue  # Skip if this state already exists

        # tried_actions.append(action_line)
        
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")

            # Update the new state dictionary
            # new_state['thought'] = thought_line
            new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
            new_state['observation'] = f"Observation: {obs}"

            new_state['critique'] = critique
            new_state['regenerate_prompt'] = regenerate_prompt

            previous_obs = f"Observation: {obs}"

            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done
            new_node.reward = r
            new_node.depth = node.depth + 1
            if r == 1:
                new_node.em = info.get('em')
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")
            logging.info(f"Feedback: {info}")

            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory_from_bottom(new_node)
                #print(trajectory)
                #if f"{action_type.lower()}[{action_param}]" not in failed_trajectories.values():
                failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})

    return list(unique_states.values())  # Return unique nodes as a list


def evaluate_node(node, args, task):
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
    
    logging.info(f"Length of votes: {len(votes)}")
    logging.info(f"Length of node.children: {len(node.children)}")
    
    # Pre-allocate votes list
    votes = votes + [0] * (len(node.children) - len(votes))
    for i, child in enumerate(node.children):
        child.value = votes[i] 
    # max_vote = max(votes) if votes else 1
    # if max_vote == 0:
    #     max_vote = 1  # Avoid division by zero
    
    # terminal_conditions = [1 if child.is_terminal else 0 for child in node.children]
    # for i, condition in enumerate(terminal_conditions):
    #     if condition == 1:
    #         votes[i] = max_vote + 1
    
    # for i, child in enumerate(node.children):
    #     child.value = votes[i] / max_vote  # Now safe from division by zero
    
    return sum(votes) / len(votes) if votes else 0


def print_tree(node, level=0):
    indent = "  " * level
    print(f"{indent}{node}")
    for child in node.children:
        print_tree(child, level + 1)

def backpropagate(node, value):
    while node:
        node.visits += 1
        if node.is_terminal:
            if node.reward == 0:
                node.value = (node.value * (node.visits - 1) + (-1)) / node.visits
                logging.info(f"Backpropagating with reward 0 at depth {node.depth}. New value: {node.value}.")
            else:
                node.value = (node.value * (node.visits - 1) + value) / node.visits
                logging.info(f"Backpropagating with reward 1 at depth {node.depth}. New value: {node.value}.")
        else:
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            logging.info(f"Backpropagating at depth {node.depth}. New value: {node.value}.")

        node = node.parent

def generate_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['thought']:
            new_segment.append(f"Thought {node.depth}: {node.state['thought']}")
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n'.join(reversed(trajectory))