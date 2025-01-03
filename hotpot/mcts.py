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


def collect_trajectory_nodes(node):
    trajectory = [node.state]
    while node.parent:
        trajectory.append(node.parent.state)
        node = node.parent
    trajectory.append(node.question)
    return trajectory[::-1]


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
    task_dict['best_trajectory_nodes'] = collect_trajectory_nodes(best_child)
    task_id = idx
    json.dump(task_dict, open(os.path.join(trajectories_save_path, f"{task_id}.json"), 'w'), indent=4)

def fschat_mcts_search(args, task, idx, iterations=50, to_print=True, trajectories_save_path=None,
                       dpo_policy_model=None, dpo_reference_model=None, tokenizer=None, enable_reflection=False):
    global gpt
    global failed_trajectories
    global reflection_map
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

    root = Node(state=None, question=x)
    cur_task = x
    if enable_reflection:
        instruction_path = "../prompt/instructions/hotpot_inst_reflection.txt"
        icl_path = "../prompt/icl_examples/hotpot_icl_reflection.json"    
    else:  
        instruction_path = "../prompt/instructions/hotpot_inst.txt"
        icl_path = "../prompt/icl_examples/hotpot_icl.json"
    with open(instruction_path) as f:
        instruction = f.read()
    raw_icl = json.load(open(icl_path))

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
            node = select_node(root)
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


def _generate_reflection_query(log_str: str, memory, FEW_SHOT_EXAMPLES):
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = log_str
    query = []
    query.append(['system', 'You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.\nHere are some examples:'])
    for item in FEW_SHOT_EXAMPLES:
        query.append([item['role'], item['content']])
    tmp_memory = f'{scenario}'
    if len(memory) > 0:
        tmp_memory += '\n\nReflection from past attempts:\n'
        for i, m in enumerate(memory):
            tmp_memory += f'Trial #{i}: {m}\n'
    tmp_memory += "\nReflection:"
    query.append(['user', tmp_memory])
    query.append(['assistant', None])
    return query

def fschat_refine_search(args, task, idx, iterations=50, to_print=True, trajectories_save_path=None, refine_num=1):
    global gpt
    global failed_trajectories
    global reflection_map
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = env.reset(idx=idx)
    if to_print:
        print(idx, x)

    root = Node(state=None, question=x)
    cur_task = x
    instruction_path = "../prompt/instructions/hotpot_inst.txt"
    icl_path = "../prompt/icl_examples/hotpot_icl.json"
    with open(instruction_path) as f:
        instruction = f.read()
    raw_icl = json.load(open(icl_path))
    reflexion_icl_path = "../prompt/reflexion_icl_examples/hotpot.json"
    reflexion_few_shot = json.load(open(reflexion_icl_path))
    observation, messages = prompt_with_icl(instruction, raw_icl, cur_task, 3)
    root.messages = messages

    failed_trajectories = []
    reflection_map = []
    successful_trajectories = []
    unsuccessful_trajectories = []
    not_finished_trajectories = []

    break_flag = False
    # Main Loop
    for i in range(iterations):
        if break_flag:
            break
        logging.info(f"Iteration {i + 1}")
        trials = []
        memory = []
        for trial_idx in range(refine_num):
            if break_flag:
                break
            node = root  # Always start from the root node
            depth = 0
            # Perform a simulation from the root
            while not node.is_terminal and depth < args.max_depth:
                expand_node(node, args, task, args.max_depth, memory=memory)  # Expand current node
                if not node.children:
                    break  # If no child can be generated, break
                if args.expansion_sampling_method == "critique":
                    node = node.children[-1]
                else:
                    node = random.choice(node.children)  # Randomly select a child node
                depth += 1

            # Check the terminal condition
            if node.is_terminal and node.reward == 1:
                logging.info(f"Successful trajectory found in iteration {i + 1}")
                successful_trajectories.append(node)
                break_flag = True
            elif node.is_terminal and node.reward < 1:
                logging.info(f"Unsuccessful trajectory found in iteration {i + 1}")
                unsuccessful_trajectories.append(node)
            elif not node.is_terminal:
                logging.info(f"Not finished trajectory found in iteration {i + 1}")
                not_finished_trajectories.append(node)
            else:
                raise KeyError
            # todo: debug
            this_messages = collect_trajectory_nodes(node)
            inst = this_messages[0]
            this_trajectory = inst + '\n'
            for item in this_messages[2:]:
                new_observation = item['observation']
                this_trajectory += f"{item['action']}\n{new_observation}\n"
            trials.append([this_trajectory, this_messages[-1]['observation']])
            plan_query = _generate_reflection_query(this_trajectory, memory, reflexion_few_shot)
            reflexion_context = copy.deepcopy(get_context(node, args.conv_template, args.backend))
            reflexion_context.messages = plan_query
            next_plan = gpt(reflexion_context, n=1, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)[0]
            memory.append(next_plan)

    best_tree_child = node
    # best_trajectory_index_list = collect_trajectory_index(best_tree_child)
    task_dict = root.to_dict()
    task_dict['best child reward'] = best_tree_child.reward
    task_dict['memory'] = memory
    task_dict['trials'] = trials
    # task_dict['best_trajectory_index_list'] = best_trajectory_index_list
    # if args.add_fixed_prefix:
    #     task_id = idx.replace("fixed_", "")
    # else:
    task_id = idx
    json.dump(task_dict, open(os.path.join(trajectories_save_path, f"{task_id}.json"), 'w'), indent=4)
    # Post-process: select the best trajectory
    if successful_trajectories:
        best_node = max(successful_trajectories, key=lambda x: x.reward)
        return best_node.state, best_node.value, best_node.reward, best_node.em
    elif unsuccessful_trajectories:
        best_node = max(unsuccessful_trajectories, key=lambda x: x.reward)
        return best_node.state, best_node.value, best_node.reward, best_node.em
    elif not_finished_trajectories:
        return 0, 0, 0, 0

def fschat_simple_search(args, task, idx, iterations=30, to_print=True, trajectories_save_path=None, enable_reflection=False):
    global gpt
    global failed_trajectories
    global reflection_map
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

    root = Node(state=None, question=x)
    
    cur_task = x
    if enable_reflection:
        instruction_path = "../prompt/instructions/hotpot_inst_reflection.txt"
        icl_path = "../prompt/icl_examples/hotpot_icl_reflection.json"    
    else:  
        instruction_path = "../prompt/instructions/hotpot_inst.txt"
        icl_path = "../prompt/icl_examples/hotpot_icl.json"
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

    for i in range(iterations):
        logging.info(f"Iteration {i + 1}...")
        node = select_node(root)
        depth = 0

        # Perform a simulation from the root
        while not node.is_terminal and depth < args.max_depth:
            expand_node(node, args, task, max_depth=args.max_depth)  # Expand current node
            if not node.children:
                break  # If no child can be generated, break
            if args.expansion_sampling_method == "critique":
                node = node.children[-1]
            else:
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

def select_node(node):
    while node and node.children:
        logging.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
        
        terminal_children = [child for child in node.children if child.is_terminal]
        terminal_status = [child.is_terminal for child in node.children]
        
        if len(terminal_children) == len(node.children):
            logging.info(f"All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:  
                node.parent.children.remove(node)
            node = node.parent  
            if node.depth == 0:
                break
            else:
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

def expand_node(node, args, task, max_depth, memory=None):
    n = args.n_generate_sample
    if node.depth >= max_depth:
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
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
        elif args.expansion_sampling_method == 'memory':
            new_nodes = generate_new_states_memory_fastchat_conv(node, args, task, n, memory)
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
        max_value_index=random.randint(0,len(new_states)-1)
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
    elif "Phi-3" in backend or "lama31" in backend or 'auto-j' in backend:
        conv = get_conv_from_bottom(node, conv_template)
        conv.append_message(conv.roles[1], None)
        context =  conv
    else:
        raise NotImplementedError
    return context

def generate_new_states_memory_fastchat_conv(node, args, task, n, memory):
    global failed_trajectories

    context = get_context(node, args.conv_template, args.backend)
    if len(context.messages) == 26 and len(memory) > 0:
        query = '\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
        query += f"\nHere is the task:\n{context.messages[24][1]}"
        context.messages[24][1] = query
    response_list = gpt(context, n=n, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)

    thought_lines = [parse_thought(response) for response in copy.deepcopy(response_list)]
    action_lines = [parse_action(response) for response in copy.deepcopy(response_list)]
    # sampled_actions = response_list

    logging.info(f"SAMPLED ACTION: {action_lines}")
    tried_actions = []

    unique_states = {}  # Store unique states here
    for thought_line, action_line in zip(thought_lines, action_lines):
        new_state = node.state.copy()  # Make a copy of the parent node's state

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
            # new_state['thought'] = thought_line
            new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
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


def generate_new_states_fastchat_conv(node, args, task, n):
    global failed_trajectories

    context = get_context(node, args.conv_template, args.backend)
    response_list = gpt(context, n=n, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)

    thought_lines = [parse_thought(response) for response in copy.deepcopy(response_list)]
    action_lines = [parse_action(response) for response in copy.deepcopy(response_list)]
    # sampled_actions = response_list

    logging.info(f"SAMPLED ACTION: {action_lines}")
    tried_actions = []
    
    unique_states = {}  # Store unique states here
    for thought_line, action_line in zip(thought_lines, action_lines):
        new_state = node.state.copy()  # Make a copy of the parent node's state

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
            # new_state['thought'] = thought_line
            new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
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