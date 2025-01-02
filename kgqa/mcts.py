import itertools
import numpy as np
from functools import partial

from torch.ao.quantization.backend_config.onednn import observation_type
import sys
sys.path.append('/home/zhaiyuanzhao/LLM-Agents-with-Q')
from kgqa.tog.utils import *
from kgqa.tog.freebase_func import *
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

# 对三元组打分，没有用这个
def triple_scores(triples, question, thought, args):
    scores = []
    prompt = f"Question: {question}\nThought: {thought}\n"
    for i, triple in enumerate(triples):
        prompt += f"Triple {i + 1}: {triple}\n"
    prompt += "Please evaluate how much each triple helps in answering the question on a scale from 0 to 0.9, where 0 means not helpful at all and 0.9 means very helpful. Provide the scores in square brackets, e.g., [0.8, 0.5, 0.3].\n\nExamples:\n1. Question: What is the capital of France?\nThought: The capital of France is a well-known city.\nTriple 1: (France, capital, Paris)\nTriple 2: (France, largest city, Paris)\nTriple 3: (France, language, French)\nScores: [0.9, 0.8, 0.2]"

    # 调用 GPT 模型获取评分
    if args.enable_fastchat_conv and 'lama' in args.backend:
        score_output = llama31_instruct(prompt, model=args.backend, n=1)[0]
    else:
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
def get_value(task, x, y, n_evaluate_sample, args, cache_value=True):
    global reflection_map
    global failed_trajectories

    unique_trajectories = get_unique_trajectories(failed_trajectories)
    #修改提示词
    value_prompt = y+ "Please evaluate how much the triplet helps in answering the question on a scale from 0 to 0.9, where 0 means not helpful at all and 0.9 means very helpful. Provide the scores in square brackets, e.g., [0.8, 0.5, 0.3].\n\nExamples:\n1. Question: What is the capital of France?\nThought: The capital of France is a well-known city.\nAction: Choose[France, capital, Paris]\nScores: [0.8]"
    logging.info(f"Current: {x}")
    logging.info(f"Current: {y}")
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    logging.info(f"VALUE PROMPT: {value_prompt}")
    
    if args.enable_fastchat_conv and 'lama' in args.backend:
        value_outputs = llama31_instruct(value_prompt, model=args.backend, n=n_evaluate_sample)
    else:
        value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    logging.info(f"VALUE OUTPUTS: {value_outputs}")
    value = task.value_outputs_unwrap(value_outputs)
    logging.info(f"VALUES: {value}")
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, args, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, args, cache_value=cache_value)
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
    task_dict['best reward'] = best_child.reward  # contain nodes during rollout
    task_dict['answer'] = best_child.answer
    task_dict['best em'] = best_child.em
    task_dict['best child reward'] = best_tree_child.reward
    task_dict['best child em'] = best_tree_child.em
    task_dict['best_trajectory_index_list'] = best_trajectory_index_list
    task_dict['true_answer'] = node.true_answer
    task_id = idx
    json.dump(task_dict, open(os.path.join(trajectories_save_path, f"{task_id}.json"), 'w'), indent=4)



#生成推理链
def get_reasoning_chain(node):
    messages = []

    while node.parent:
        # if 'regenerate_prompt' in node.state.keys():
        #     critique = f"{node.state['regenerate_prompt']}"
        #     messages.insert(0,{'role':'user', 'content': f"{node.state['observation']}"+critique+node.state['observation']})
        # else:
        # 增加 wiki 信息
        #messages.insert(0,{'role':'user', 'content': f"{node.state['observation']}" + f"  {node.wikiinformation}"})
        messages.insert(0,node.triple)
        # if 'regenerate_prompt' in node.state.keys():
        #     critique = f"{node.state['regenerate_prompt']}"
        #     action = f"{node.state['action']}"
        #     messages.insert(0,{'role':'assistant', 'content': critique+"\n"+action})
        # else:
        #     messages.insert(0,{'role':'assistant', 'content': f"{node.state['action']}"})
        # messages.insert(0, [conv.roles[0], f"{node.state['observation']}"])
        # messages.insert(0, [conv.roles[1], f"{node.state['action']}"])
        messages.append(node.triple)
        node = node.parent
    return messages

#判断能否回答问题
def reasoning(reasoning_chain, question, args):
    prompt = prompt_evaluate + question
    chain_prompt = ''
    for sublist in reasoning_chain:
        chain_prompt += str(sublist)  # 确保正确拼接字符串
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    if args.enable_fastchat_conv and 'lama' in args.backend:
        result = llama31_instruct(prompt, model=args.backend, n=1)[0]
    else:
        result = gpt(prompt, n=1)[0]
    result = extract_answer(result)
    if if_true(result):
        return True
    else:
        return False
    
#回答问题
def get_answer(reasonging_chain, question, args):
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join(
        [', '.join([str(x) for x in chain]) for sublist in reasonging_chain for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    if args.enable_fastchat_conv and 'lama' in args.backend:
        result = llama31_instruct(prompt, model=args.backend, n=1)[0]
    else:
        result = gpt(prompt, n=1)[0]
    return result


#处理答案

def check_string(string):
    if string is None:
        return False
    return "{" in string





def fschat_mcts_search(args, task, idx, iterations=50, to_print=True, trajectories_save_path=None,
                       dpo_policy_model=None, dpo_reference_model=None, tokenizer=None, enable_reflection=False):
    global gpt
    global failed_trajectories
    global reflection_map
    # 定义一个字符串，储存每次迭代后已经掌握的信息
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

    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a')
    # env.sessions[idx] = {'session': idx, 'page_type': 'init'}
    x = env.reset(idx=idx)
    if to_print:
        print(idx, x)

    # 把中心词传给根节点
    root = Node(state=None, question=x[0], topic_entity=x[1], true_answer=x[2])
    # 增加observation
    # 扩展的关系
    # next_entity_relations_list = []
    # 扩展的实体
    # next_entity_list = []
    # 扩展的三元组
    # next_chain_list = []
    # total_scores = []
    next_entity_relations_list, next_entity_list, next_chain_list = find_next_triples(args.n_generate_sample, root,
                                                                                      args)
    root.state['observation'] = f"Here are some candidate knowledge triplets you can choose from: " + str(
        next_chain_list)
    root.next_triple_list = next_chain_list
    root.next_entity_relations_list = next_entity_relations_list
    root.next_entity_list = next_entity_list

    cur_task = x[0]
    if enable_reflection:
        instruction_path = "../prompt/instructions/mygraph_inst_reflection.txt"
        icl_path = "../prompt/icl_examples/mygraph_icl_reflection.json"
    else:
        instruction_path = "../prompt/instructions/mygraph_inst_new.txt"
        icl_path = "../prompt/icl_examples/mygraph_inst_new.json"
    with open(instruction_path) as f:
        instruction = f.read()
    # 文件编码 utf-8
    raw_icl = json.load(open(icl_path, encoding='utf-8'))

    observation, messages = prompt_with_icl(instruction, raw_icl, cur_task, 3)
    assert messages[-1]['role'] == 'user'
    messages[-1]['content'] += ' Observation: ' + root.state['observation']
    # messages[-1].append({
    #     "role": "user",
    #     "content": root.state['observation']
    # })
    root.messages = messages

    # print("ROOTSTATE", root.env_state)
    all_nodes = []
    failed_trajectories = []
    reflection_map = []
    terminal_nodes = []

    for i in range(iterations):
        # print(f"Iteration {i + 1}...")
        node = select_node(root, args, i)

        while node is None or (node.is_terminal and node.reward != 1):
            logging.info(f"Need to backtrack or terminal node with reward 0 found at iteration {i + 1}, reselecting...")
            node = select_node(root, args, i)
            last_selected_node = copy.deepcopy(node)
            if node == last_selected_node:
                break

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

        # 设置终止
        if node.is_terminal and node.reward != 1:
            logging.info(f"There is not enough information in the knowledge graph.")
            break

        expand_node(node, args, task, args.max_depth)

        while node.is_terminal or not node.children:
            logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root, args, i)
            expand_node(node, args, task, args.max_depth)

        if args.enable_value_evaluation:
            # TODO
            value = evaluate_node(node, args, task)

        # Find the child with the highest value or UCT? A: similar effect.
        if args.enable_rollout_with_critique:
            reward, terminal_node = rollout_with_critique(max(node.children, key=lambda child: child.value), args, task,
                                                          idx, max_depth=args.max_depth)
        else:
            reward, terminal_node = rollout_random(max(node.children, key=lambda child: child.value), args, task, idx,
                                                   max_depth=args.max_depth)
        # TODO
        # reward, terminal_node = rollout_on_kg(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=args.max_depth)

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
        terminal_nodes_with_reward_1 = [node for node in collect_all_nodes(root) if
                                        node.is_terminal and node.reward == 1]
        if terminal_nodes_with_reward_1:
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.value)
            if args.disable_early_stop:
                continue
            else:
                save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)
                return best_node.state, best_node.value, best_node.reward, best_node.em

        for j, (node, value) in enumerate(all_nodes):
            logging.info(f"Node {j + 1}: {str(node)}")

        logging.info(f"State of all_nodes after iteration {i + 1}: {all_nodes}")

    all_nodes_list = collect_all_nodes(root)
    for node in all_nodes_list:
        if node.is_terminal and node.value == 0:
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


def fschat_simple_search(args, task, idx, iterations=30, to_print=True, trajectories_save_path=None,
                         enable_reflection=False):
    global gpt
    global failed_trajectories
    global reflection_map
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a')
    # env.sessions[idx] = {'session': idx, 'page_type': 'init'}
    x = env.reset(idx=idx)
    if to_print:
        print(idx, x)

    # 把中心词传给根节点
    root = Node(state=None, question=x[0], topic_entity=x[1])

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
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a')

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
        return 0, 0, 0, 0


def fschat_beam_search(args, task, idx, to_print=True, trajectories_save_path=None,
                       dpo_policy_model=None, dpo_reference_model=None, tokenizer=None, enable_reflection=False):
    global gpt
    global failed_trajectories
    global reflection_map
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a')
    # env.sessions[idx] = {'session': idx, 'page_type': 'init'}
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
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a')

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
        return 0, 0, 0, 0


def select_node(node, args, i=0):
    # 根据wiki选择初始节点
    if node.depth == 0 and i != 0 and args.enable_wiki_search:
        node_children_relation = []
        for child in node.children:
            node_children_relation.append(child.triple)
        score_relation = triple_scores(node_children_relation, node.question, wiki_explored, args)
        max_score_index = score_relation.index(max(score_relation))
        selected_child = node.children[max_score_index] # list index error
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
            if node == None:
                break
            if node.depth == 0:
                break
            else:
                continue

        node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
        if node_with_reward_1:
            logging.info(f"Found terminal node with reward 1 at depth {node.depth}.")
            return node_with_reward_1

        node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(),
                   default=None)

        while node.is_terminal and node.reward != 1:
            node = max((child for child in node.parent.children if not child.is_terminal),
                       key=lambda child: child.uct(), default=None)

        logging.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")

    return node  # This will return None if all paths from the root are exhausted


# 判断需要wiki搜索的内容
def generate_query(question="", myinformation_explored="", mywiki_explored=""):
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
        # 达到最大深度，判断是否需要wiki
        '''
        myenv = wikienv.WikiEnv()
        if node.reward !=1:
            this_time_imformation = ""
            global wiki_explored
            keywords = generate_query(node.question, information_explored,wiki_explored)
            if len(keywords) != 0:
                for keyword in keywords:
                    this_time_imformation += myenv.search_step(keyword)
            #wiki_explored = wiki_explored + this_time_imformation
            #只保留当前的wiki检索信息,增加长度限制
            wiki_explored = this_time_imformation[:1000]
            '''
        return

    assert args.expansion_sampling_method == 'vanilla' or args.enable_fastchat_conv  # only fastchat api supports various expansion_sampling_method yet

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
    llm_output = llm_output.replace("\n", " ")
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
        # new_state = new_state[0]
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
    assert n == 1, "large rollout_width is meanless for rollout_random"

    while not node.is_terminal and depth < max_depth:
        # Generate new states
        new_states = []
        values = []
        while len(new_states) == 0:
            if args.enable_fastchat_conv:
                new_states = generate_new_states_fastchat_conv(node, args, task, n)

                '''
                for state in new_states:
                    if state.is_terminal and state.reward != 1:
                        logging.info(f"There is not enough information in the knowledge graph.")
                        break
                '''
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
        # max_value_index=random.randint(0,len(new_states)-1)
        max_value_index = random.randint(0, len(new_states) - 1)
        node.children = [new_states[max_value_index]]
        node = new_states[max_value_index]

        #evaluate_node(node.parent, args, task)
        depth += 1
        if depth == max_depth:
            reasoning_chain = get_reasoning_chain(node)
            if len(reasoning_chain) == 0:
                node.is_terminal = False
            else:
                answer = reasoning(reasoning_chain, node.question, args)
                if answer:
                    answer = get_answer(reasoning_chain, node.question, args)
                    if check_string(answer):
                        response = clean_results(answer)
                        if response == "NULL":
                            node.is_terminal = False
                        else:
                            if exact_match(response, node.true_answer):
                                node.is_terminal = True
                                node.reward = 1
                                node.answer = response
                            else:
                                node.is_terminal = False
                                node.reward = -1
                                node.answer = response
    return node.reward, node


def rollout_on_kg(node, args, task, idx, max_depth=7):
    # TODO
    raise NotImplementedError


def rollout_with_critique(node, args, task, idx, max_depth=15):
    depth = node.depth
    n = args.rollout_width
    assert n == 1, "the same with rollout_random"
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
                                                                            n=2,
                                                                            critique_prompt_template=critique_prompt_template)
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
    sampled_actions = get_samples(task, prompt, f"Thought {node.depth + 1}: ", n, prompt_sample=args.prompt_sample,
                                  stop="Observation")
    logging.info(f"SAMPLED ACTION: {sampled_actions}")
    tried_actions = []

    unique_states = {}  # Store unique states here
    for action in sampled_actions:
        action = action.replace("Thought 1:  Thought 1: ", "Thought 1: ")
        new_state = node.state.copy()  # Make a copy of the parent node's state

        thought_line = next(
            (line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")),
            '')
        action_line = next(
            (line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line),
            None)

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
                # print(trajectory)
                # if f"{action_type.lower()}[{action_param}]" not in failed_trajectories.values():
                failed_trajectories.append(
                    {'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})

    return list(unique_states.values())  # Return unique nodes as a list


def get_context(node, conv_template, backend):
    if "gpt-" in backend:
        messages = get_messages_from_bottom(node)
        context = messages
    elif "Phi-3" in backend or "lama31" in backend or 'auto-j' in backend or 'Llama31-KTO' in backend:
        conv = get_conv_from_bottom(node, conv_template)
        conv.append_message(conv.roles[1], None)
        context = conv
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


# 图谱中无信息，llm直接回答问题，提取答案
def clean_results(string):
    if "{" in string:
        start = string.find("{") + 1
        end = string.find("}")
        content = string[start:end]
        return content
    else:
        return "NULL"


# 这个没用了
def choose_node(question, observation_list, triple_list, args):
    idx = 0
    # 构造提示（prompt）
    prompt = f"""
    Question: {question}
    You are given a list of the triples you can obtain by choosing each branch.
    Your task is to determine which branch is most likely to help answer the question.

    Observations and corresponding triples:
    """

    for i, (obs, triples) in enumerate(zip(observation_list, triple_list)):
        prompt += f"\nBranch {i}:\n"
        prompt += f"  Observation: {obs}\n"
        prompt += f"  Triples: {', '.join(triples)}\n"

    prompt += """
    Examples:
    Branch 0:
      Observation: The sky is blue.
      Triples: (sky, color, blue)
    Branch 1:
      Observation: The grass is green.
      Triples: (grass, color, green)
    Response: [1]

    Branch 0:
      Observation: The cat is sleeping.
      Triples: (cat, action, sleeping)
    Branch 1:
      Observation: The dog is barking.
      Triples: (dog, action, barking)
    Response: [0]

    Please provide the index of the branch that is most likely to help answer the question in the format [index].
    """

    # 调用大模型
    response = gpt(prompt, n=1, stop=None)[0].strip()

    # 解析响应以获取索引
    try:
        # 提取索引部分
        idx_str = response.strip('[]')
        idx = int(idx_str)
        if idx < 0 or idx >= len(observation_list):
            raise ValueError("Index out of range")
    except ValueError:
        logging.error(f"Invalid response from GPT: {response}")
        idx = 0  # 默认选择第一个分支
    return idx


def string_to_list(input_string):
    # 使用正则表达式提取字符串中[]内部的内容
    match = re.search(r'\[(.*?)\]', input_string)
    if match:
        stripped_string = match.group(1)
        # 使用逗号分割字符串，并去除每个元素两边的空格和双引号
        result_list = [item.strip().strip('"').strip("'") for item in stripped_string.split(",")]
    else:
        result_list = []
    return result_list


# 找到当前节点的下一跳三元组
def find_next_triples(n, node, args):
    if len(node.next_triple_list) == 0:
        question = node.question
        topic_entity = node.topic_entity

        pre_relations = []
        pre_heads = [-1] * len(topic_entity)
        flag_printed = False
        search_depth = 1

        next_entity_relations_list = []
        i = 0
        for entity in topic_entity:
            if entity != "[FINISH_ID]":
                retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations,
                                                                       pre_heads[i], question, args)
                next_entity_relations_list.extend(retrieve_relations_with_scores)
            i += 1
        total_candidates = []
        total_scores = []
        total_relations = []
        total_entities_id = []
        total_topic_entities = []
        total_head = []
        num_relation = 0

        for entity in next_entity_relations_list:
            if entity['head']:
                my_entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
            else:
                my_entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)

            entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in my_entity_candidates_id]
            indices_to_remove_entity = []
            for i in range(len(entity_candidates)):
                if node.depth > 0:
                    if entity_candidates[i] == 'UnName_Entity' or my_entity_candidates_id[i] == entity['entity'] or \
                            entity_candidates[i] == str(node.parent.topic_entity.keys()):
                        indices_to_remove_entity.append(i)
                else:
                    if entity_candidates[i] == 'UnName_Entity' or my_entity_candidates_id[i] == entity['entity']:
                        indices_to_remove_entity.append(i)
            for i in sorted(indices_to_remove_entity, reverse=True):
                del my_entity_candidates_id[i]

            if len(my_entity_candidates_id) == 0:
                continue
            else:
                if len(my_entity_candidates_id) > 10:
                    my_entity_candidates_id = random.sample(my_entity_candidates_id, 10)
                # entity_candidates_id = random.sample(my_entity_candidates_id, args.num_retain_entity)
                # 保留全部实体
                entity_candidates_id = my_entity_candidates_id

            # 修改，不需要对实体的打分
            scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id,
                                                                           entity['score'], entity['relation'], args)
            total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(
                entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores,
                total_relations, total_entities_id, total_topic_entities, total_head)

        flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id,
                                                                                      total_relations, total_candidates,
                                                                                      total_topic_entities, total_head,
                                                                                      total_scores, args)
        # self.cluster_chain_of_entities.append(chain_of_entities)
        # 移出 unname
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
            mychain = [topic_entity[total_topic_entities[i]], total_relations[i], total_candidates[i]]
            mychain_of_entities.append(mychain)

        node.entity_list = entities_id
        new_topic_entity = []
        for i in range(len(total_candidates)):
            new_topic_entity.append({})
            new_topic_entity[i][total_entities_id[i]] = total_candidates[i]
        # node.topic_entity = new_topic_entity
        '''
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
        '''
    else:
        return node.next_entity_relations_list, node.next_entity_list, node.next_triple_list

    return total_relations, new_topic_entity, mychain_of_entities

#正确答案判断
def exact_match(response, answers):
    if response is None:
        return False
    clean_result = response.strip().replace(" ","").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ","").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False

# 直接生成所有的新状态
def generate_new_states_fastchat_conv(node, args, task, n):
    global failed_trajectories

    context = get_context(node, args.conv_template, args.backend)
    tried_actions = []

    unique_states = {}

    next_entity_relations_list, next_entity_list, next_chain_list = find_next_triples(n, node, args)
    # 扩展的节点数
    i = len(next_entity_relations_list)
    # 图谱中没找到信息
    if i == 0:
        # 直接回答
        results = generate_without_explored_paths(node.question, [], args, '')
        action_type = "Finish[]"
        action_param = clean_results(results)
        new_state = node.state.copy()
        if action_type.startswith("Finish[") and action_type.endswith("]"):
            # 把当前节点传进环境
            env.env.env.node = node
            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
            # Update the new state dictionary
            # new_state['thought'] = thought_line
            new_state['action'] = f"Thought: {action_param} Action: {action_type}"
            new_state['observation'] = f"Observation: {results}"
            unique_key = f"{action_param}::{action_type}::none"

            # 新节点的topic_entity即父节点的关系剪枝后的entity
            new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,true_answer=node.true_answer)
            new_node.is_terminal = True
            # 正确答案判断
            if exact_match(action_param, node.true_answer):
                new_node.reward = r
            else:
                new_node.reward = -1
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
    response_list = gpt(context, n=args.n_generate_sample, stop="Observation",
                        enable_fastchat_conv=args.enable_fastchat_conv)
    thought_lines = [parse_thought(response) for response in copy.deepcopy(response_list)]
    action_lines = [parse_action(response) for response in copy.deepcopy(response_list)]
    # sampled_actions = response_list

    logging.info(f"SAMPLED ACTION: {action_lines}")
    # Store unique states here

    # 处理图谱扩展
    for thought_line, action_line in zip(thought_lines, action_lines):
        idx = 0
        new_state = node.state.copy()  # Make a copy of the parent node's state

        # Use action to form a unique key
        unique_key = f"{action_line}"

        if unique_key in unique_states:
            continue  # Skip if this state already exists

        tried_actions.append(action_line)

        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = '[' + action_line.split('[')[1].split(']')[0] + ']' if '[' in action_line else ""
            # action_param_match = re.search(r'\[\[(.*)\]\]', action_line)
            # action_param = f"[{action_param_match.group(1)}]" if action_param_match else ""

            if action_type.startswith("Finish"):
                # Use action to form a unique key
                unique_key = f"{action_line}"

                if unique_key in unique_states:
                    continue  # Skip if this state already exists

                tried_actions.append(action_line)
                # 把当前节点传进环境
                env.env.env.node = node
                obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
                # Update the new state dictionary
                # new_state['thought'] = thought_line
                new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
                new_state['observation'] = f"Answer: {obs}"

                # 新节点的topic_entity即父节点的关系剪枝后的entity
                new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,
                                true_answer=node.true_answer)
                new_node.is_terminal = True
                # 正确答案判断
                if exact_match(action_param ,node.true_answer):
                    new_node.reward = r
                else:
                    new_node.reward = -1
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
                '''
                #再次判断选择哪个三元组，使用最简单的提示词
                #重复次数i
                i =5
                while True:
                    prompt = f"""
                        Question: {node.question}
                        Here are some candidate knowledge triplets you can choose from: {node.state['observation']}
                        You need to pick one triple that is most helpful to answer the question. You need to output the triple directly,with no other words.
                        """

                    # 调用大模型
                    action_param = gpt(prompt, n=1, stop=None)[0].strip()
                    #从action 提取选择的三元组
                    generated_chain = string_to_list(action_param)
                    i -= 1
                    if generated_chain in next_chain_list:
                        break
                    if i <=0:
                        break

                # Use action to form a unique key
                unique_key = f"{action_type}{action_param}"

                if unique_key in unique_states:
                    continue  # Skip if this state already exists

                tried_actions.append(action_line)
                '''
                # try:
                #     idx = next_chain_list.index(current_chain)
                #     print('LLM choose a triplet from the candidates')
                # except ValueError:
                #     print('LLM did not choose a triplet from the candidates')
                #     idx = random.randint(0, len(next_chain_list) - 1)
                generated_chain = string_to_list(action_param)
                if generated_chain in next_chain_list:
                    idx = next_chain_list.index(generated_chain)
                    next_entity_relation = next_entity_relations_list[idx]
                    next_entity = next_entity_list[idx]
                    next_chain = next_chain_list[idx]

                    select_relation = next_entity_relation
                    # obs = f"Knowledge Triplets:  {next_chain}\n"
                    new_state['action'] = f"Thought: {thought_line} Action: Choose{action_param}"
                    # new_state['observation'] = f"Here are some candidate knowledge triplets you can choose from: {obs}"
                    new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=next_entity,true_answer=node.true_answer)
                    new_node.triple = str(next_chain)
                    #判断能否回答，
                    reasoning_chain = get_reasoning_chain(new_node)
                    if len(reasoning_chain) == 0:
                        new_node.is_terminal = False
                    else:
                        answer = reasoning(reasoning_chain, node.question, args)
                        if answer:
                            answer = get_answer(reasoning_chain, node.question, args)
                            if check_string(answer):
                                response = clean_results(answer)
                                if response == "NULL":
                                    new_node.is_terminal = False
                                else:
                                    if exact_match(response, new_node.true_answer):
                                        new_node.is_terminal = True
                                        new_node.reward = 1
                                        new_node.answer = response

                                    else:
                                        new_node.is_terminal = False
                    new_node.depth = node.depth + 1
                    # 找到节点的下一跳
                    new_node.next_entity_relations_list, new_node.next_entity_list, new_node.next_triple_list = find_next_triples(
                        n, new_node, args)
                    new_node.state[
                        'observation'] = f"Observation: Here are some candidate knowledge triplets you can choose from: " + str(
                        new_node.next_triple_list)
                    unique_states[unique_key] = new_node
                    logging.info(f"NEW NODE: {new_node}")

                else:
                    # 如果生成不在next_chain_list候选三元组，告诉他，让他重新生成
                    # obs =

                    new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
                    original_observation = new_state['observation']
                    new_state[
                        'observation'] = f'Observation: Invalid action! You chose a triplet that does not match any of the candidate knowledge triplets. Please remember to choosing an exact and complete triplet from: ' + original_observation[
                                                                                                                                                                                                                             original_observation.find(
                                                                                                                                                                                                                                 '['):] if '[' in original_observation else original_observation
                    new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,true_answer=node.true_answer)
                    new_node.depth = node.depth + 1
                    unique_states[unique_key] = new_node
                    logging.info(f"NEW NODE: {new_node}")

                # 搜索信息
                '''
                myenv = wikienv.WikiEnv()
                this_time_imformation = ""
                keywords = generate_query(node.question, new_node.state['observation'], "")
                if len(keywords) != 0:
                    for keyword in keywords:
                        this_time_imformation += myenv.search_step(keyword)
                # wiki_explored = wiki_explored + this_time_imformation
                # 增加长度限制
                new_node.wikiinformation = f"Some information from wikipedia: "+this_time_imformation[:1000]
                '''
                # info = select_relation
                # logging.info(f"Feedback: {info}")
        idx += 1

    return list(unique_states.values())


def get_raw_observation(text):
    keyword = '\n\nBelow are the previous Thought and Action you generated along with their corresponding Observation:'
    index = text.find(keyword)
    if index != -1:
        return text[:index]
    else:
        return text


def get_historical_context(context):
    prompt = ''
    for message in context.messages[25:]:  # todo
        if message[1] is not None:
            prompt += message[1].strip() + '\n'
    return prompt


# 直接生成所有的新状态
def generate_new_states_critique_fastchat_conv(node, args, task, n, critique_prompt_template):
    global failed_trajectories

    unique_states = {}

    next_entity_relations_list, next_entity_list, next_chain_list = find_next_triples(n, node, args)
    # 扩展的节点数
    i = len(next_entity_relations_list)
    # 图谱中没找到信息
    if i == 0:
        # 直接回答
        results = generate_without_explored_paths(node.question, [], args, '')
        action_type = "Finish[]"
        action_param = clean_results(results)
        new_state = node.state.copy()
        if action_type.startswith("Finish[") and action_type.endswith("]"):
            # 把当前节点传进环境
            env.env.env.node = node
            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
            # Update the new state dictionary
            # new_state['thought'] = thought_line
            new_state['action'] = f"Thought: {action_param} Action: {action_type}"
            new_state['observation'] = f"Observation: {results}"
            unique_key = f"{action_param}::{action_type}::none"

            # 新节点的topic_entity即父节点的关系剪枝后的entity
            new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,true_answer=node.true_answer)
            new_node.is_terminal = True
            # 正确答案判断
            if exact_match(action_param, node.true_answer):
                new_node.reward = r
            else:
                new_node.reward = -1
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
    
    context = get_context(node, args.conv_template, args.backend)
    tried_actions = []
    previous_response = None
    previous_obs = None
    
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
                critique_prompt_template = critique_prompt_template.format(
                    user_inst=critique_context.messages[-2][1],
                    historical_context=get_historical_context(critique_context),
                    current_state=previous_response + '\n' + previous_obs
                )
                if 'gpt' in args.critique_backend:
                    raise NotImplementedError
                else:
                    critique_context.messages = [['system', critique_prompt_template.split('</system>')[0]],
                                                 ['user', critique_prompt_template.split('</system>')[-1]],
                                                 ['assistant', None]]
                critique = \
                critique_gpt(critique_context, n=1, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)[
                    0]
                if critique.startswith('Critique:'):
                    critique = critique[9:]
                regenerate_prompt = '\n\nBelow are the previous Thought and Action you generated along with their corresponding Observation: \n\n'
                regenerate_prompt += previous_response + "\n"
                regenerate_prompt += previous_obs + "\n"
                regenerate_prompt += 'Critique: ' + critique + "\n\n"
                regenerate_prompt += 'Based on the critique, generate a new Thought and Action with as much distinctiveness as possible for the Observation:' + "\n"
                context.messages[-2][1] += regenerate_prompt + "\n" + original_observation
            else:
                raise NotImplementedError
            

            
        response = gpt(context, n=1, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)[0]
        previous_response = response
        thought_line = parse_thought(response) 
        action_line = parse_action(response) 

        idx = 0
        new_state = node.state.copy()  # Make a copy of the parent node's state

        # Use action to form a unique key
        unique_key = f"{action_line}"

        if unique_key in unique_states:
            continue  # Skip if this state already exists

        tried_actions.append(action_line)

        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = '[' + action_line.split('[')[1].split(']')[0] + ']' if '[' in action_line else ""
            # action_param_match = re.search(r'\[\[(.*)\]\]', action_line)
            # action_param = f"[{action_param_match.group(1)}]" if action_param_match else ""

            if action_type.startswith("Finish"):
                # Use action to form a unique key
                unique_key = f"{action_line}"

                if unique_key in unique_states:
                    continue  # Skip if this state already exists

                tried_actions.append(action_line)
                # 把当前节点传进环境
                env.env.env.node = node
                obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
                # Update the new state dictionary
                # new_state['thought'] = thought_line
                new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
                new_state['observation'] = f"Answer: {obs}"
                new_state['critique'] = critique
                new_state['regenerate_prompt'] = regenerate_prompt

                previous_obs = f"Answer: {obs}"

                # 新节点的topic_entity即父节点的关系剪枝后的entity
                new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,
                                true_answer=node.true_answer)
                new_node.is_terminal = True
                # 正确答案判断
                if exact_match(action_param ,node.true_answer):
                    new_node.reward = r
                else:
                    new_node.reward = -1
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
                generated_chain = string_to_list(action_param)
                if generated_chain in next_chain_list:
                    idx = next_chain_list.index(generated_chain)
                    next_entity_relation = next_entity_relations_list[idx]
                    next_entity = next_entity_list[idx]
                    next_chain = next_chain_list[idx]

                    select_relation = next_entity_relation
                    # obs = f"Knowledge Triplets:  {next_chain}\n"
                    new_state['action'] = f"Thought: {thought_line} Action: Choose{action_param}"
                    new_state['critique'] = critique
                    new_state['regenerate_prompt'] = regenerate_prompt
                    # new_state['observation'] = f"Here are some candidate knowledge triplets you can choose from: {obs}"
                    new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,true_answer=node.true_answer)
                    new_node.triple = str(next_chain)
                    #判断能否回答，
                    reasoning_chain = get_reasoning_chain(new_node)
                    if len(reasoning_chain) == 0:
                        new_node.is_terminal = False
                    else:
                        answer = reasoning(reasoning_chain, node.question, args)
                        if answer:
                            answer = get_answer(reasoning_chain, node.question, args)
                            if check_string(answer):
                                response = clean_results(answer)
                                if response == "NULL":
                                    new_node.is_terminal = False
                                else:
                                    if exact_match(response, new_node.true_answer):
                                        new_node.is_terminal = True
                                        new_node.reward = 1
                                        new_node.answer = response

                                    else:
                                        new_node.is_terminal = False
                    new_node.depth = node.depth + 1
                    # 找到节点的下一跳
                    new_node.next_entity_relations_list, new_node.next_entity_list, new_node.next_triple_list = find_next_triples(
                        n, new_node, args)
                    new_node.state[
                        'observation'] = f"Observation: Here are some candidate knowledge triplets you can choose from: " + str(
                        new_node.next_triple_list)
                    previous_obs = f"Observation: Here are some candidate knowledge triplets you can choose from: " + str(
                        new_node.next_triple_list)
                    unique_states[unique_key] = new_node
                    logging.info(f"NEW NODE: {new_node}")

                else:
                    # 如果生成不在next_chain_list候选三元组，告诉他，让他重新生成
                    # obs =

                    new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
                    original_observation = new_state['observation']
                    new_state[
                        'observation'] = f'Observation: Invalid action! You chose a triplet that does not match any of the candidate knowledge triplets. Please remember to choosing an exact and complete triplet from: ' + original_observation[
                                                                                                                                                                                 original_observation.find('['):] if '[' in original_observation else original_observation
                    previous_obs = f'Observation: Invalid action! You chose a triplet that does not match any of the candidate knowledge triplets. Please remember to choosing an exact and complete triplet from: ' + original_observation[
                                                                                                                                                                                 original_observation.find('['):] if '[' in original_observation else original_observation
                    
                    new_state['critique'] = critique
                    new_state['regenerate_prompt'] = regenerate_prompt
                    new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,true_answer=node.true_answer)
                    new_node.depth = node.depth + 1
                    unique_states[unique_key] = new_node
                    logging.info(f"NEW NODE: {new_node}")

        idx += 1

    return list(unique_states.values())



def evaluate_node(node, args, task):
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample, args)

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
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n'.join(reversed(trajectory))