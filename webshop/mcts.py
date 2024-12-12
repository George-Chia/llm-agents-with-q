
#!/usr/bin/env python
# coding: utf-8

import os
import openai
import backoff
import sys
import copy
import itertools
import numpy as np
from functools import partial
from models import gpt
import requests
import logging
import random
import json
import re
import time

from fschat_templates import prompt_with_icl

from critique_templates import auto_j_single_template, template_v1, template_v2, template_huan, webshop_description


 
completion_tokens = prompt_tokens = 0
# openai.api_key = os.environ["OPENAI_API_KEY"]

global reflection_map
global failed_trajectories
reflection_map = []
failed_trajectories = []


from webshopEnv import webshopEnv
from node import *

env = webshopEnv()

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


    # all_tree_nodes_list = collect_all_nodes(root)
    # best_tree_child = max(all_tree_nodes_list, key=lambda x: x.reward)
    # best_trajectory_index_list = collect_trajectory_index(best_tree_child)
    # task_dict = root.to_dict()
    # task_dict['best reward'] = best_child.reward # contain nodes during rollout
    # task_dict['best child reward'] = best_tree_child.reward
    # task_dict['best_trajectory_index_list'] = best_trajectory_index_list
    # if args.add_fixed_prefix:
    #     task_id = idx.replace("fixed_","")
    # else:
    #     task_id = idx
    # json.dump(task_dict, open(os.path.join(trajectories_save_path, f"{task_id}.json"), 'w'), indent=4)


def fschat_mcts_search(args, task, idx, iterations=50, to_print=True, trajectories_save_path=None,
                       dpo_policy_model=None, dpo_reference_model=None, tokenizer=None):
    global gpt
    global failed_trajectories
    global reflection_map
    action = 'reset'
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
    x = env.step(idx, action, args.enable_seq_mode)[0]
    if to_print:
        print(idx, x)

    root = Node(state=None, question=x)
    root.env_state = copy.deepcopy(env.sessions)
    cur_task = x
    instruction_path = "../prompt/instructions/webshop_inst.txt"
    icl_path = "../prompt/icl_examples/webshop_icl.json"
    with open(instruction_path) as f:
        instruction = f.read()
    raw_icl = json.load(open(icl_path))
    observation, messages = prompt_with_icl(instruction, raw_icl, cur_task, 1)
    root.messages = messages

    #print("ROOTSTATE", root.env_state)
    all_nodes = []
    failed_trajectories = []
    reflection_map = []
    terminal_nodes = [] # containing terminal nodes during rollout

    for i in range(iterations):
        logging.info(f"Iteration {i + 1}...")
        node = select_node(root, args.using_puct, dpo_policy_model, dpo_reference_model, args.conv_template, tokenizer, args.puct_coeff)

        while node is None or (node.is_terminal and node.reward != 1):
            logging.info(f"Need to backtrack or terminal node with reward 0 found at iteration {i + 1}, reselecting...")
            node = select_node(root, args.using_puct, dpo_policy_model, dpo_reference_model, args.conv_template, tokenizer, args.puct_coef)
        
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
        
        expand_node(node, args, task, idx, max_depth=args.max_depth)


        while node.is_terminal:
            logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root, args.using_puct, dpo_policy_model, dpo_reference_model, args.conv_template, tokenizer, args.puct_coef)
            expand_node(node, args, task, idx)

        if args.enable_value_evaluation:
            val = evaluate_node(node, args, task, idx)

        # Simulation or rollout
        # terminal_node = rollout(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=args.max_depth)
        try:
            if args.enable_rollout_with_q:
                terminal_node = rollout_with_q(node, args, task, idx, args.max_depth,
                                            dpo_policy_model, dpo_reference_model, args.conv_template, tokenizer, args.puct_coeff)
            elif args.enable_rollout_with_critique:
                terminal_node = rollout_with_critique(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=args.max_depth)     
            else:
                # TODO: according to UCT instead of value?
                terminal_node = rollout_random(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=args.max_depth)     
        except:
            terminal_node = node
            terminal_node.reward == -0.5
        terminal_nodes.append(terminal_node)

        if args.enable_rollout_early_stop:
            if terminal_node.reward == 1:
                logging.info("Successful trajectory found")
                logging.info(f"Terminal node including rollouts with reward 1 found at iteration {i + 1}")
                backpropagate(terminal_node, terminal_node.reward)
                save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)
                return terminal_node.state, terminal_node.value, terminal_node.reward, terminal_node.em
        # Backpropagate reward
        backpropagate(terminal_node, terminal_node.reward)
        
        #all_nodes.extend(collect_all_nodes(root))
        #value = evaluate_node(node, args, task, idx)
        #backpropagate(node, value)
        all_nodes = [(node, node.reward) for node in collect_all_nodes(root)]
        logging.info("searching all nodes...")
        # Check for terminal nodes with a reward of 1
        terminal_nodes_with_reward_1 = [node for node, reward in all_nodes if node.is_terminal and node.reward == 1]

        if terminal_nodes_with_reward_1:
            logging.info("Successful trajectory found")
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.reward)
            if args.disable_early_stop:
                continue
            else:
                save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)
                return best_node.state, best_node.value, best_node.reward, best_node.em
    
        for j, (node, value) in enumerate(all_nodes):
            logging.info(f"Node {j+1}: {str(node)}")

        node_strings = '\n'.join(str(node[0]) for node in all_nodes)
        logging.info(f"State of all_nodes after iteration {i + 1}:\n{node_strings}")

        
    all_nodes_list = collect_all_nodes(root)
    for node in all_nodes_list:
        if node.is_terminal and node.value==0:
            backpropagate(node, node.reward)
    save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)


    all_nodes_list.extend(terminal_nodes)
    best_child = max(all_nodes_list, key=lambda x: x.reward)
    failed_trajectories = []
    print("best value found", best_child.reward)
    if best_child.reward == 1:
        logging.info("Successful trajectory found")
    else:
        logging.info("Unsuccessful/Partially Successful trajectory found")
    if best_child is None:
        best_child = root
    

    return best_child.state, best_child.value, best_child.reward, best_child.em


def fschat_simple_search(args, task, idx, iterations=50, to_print=True, trajectories_save_path=None):
    global gpt
    global failed_trajectories
    global reflection_map
    action = 'reset'
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    global critique_gpt
    if args.expansion_sampling_method == "critique":
        if args.critique_backend:
            if args.critique_temperature == None:
                args.critique_temperature = args.temperature
            critique_gpt = partial(gpt, model=args.critique_backend, temperature=args.critique_temperature)
        else:
            critique_gpt = gpt

    # start_time = time.time()
    x = env.step(idx, action, args.enable_seq_mode)[0]
    # end_time = time.time()
    # print("env.step initial 执行时间: {:.4f} 秒".format(end_time - start_time))

    root = Node(state=None, question=x)
    root.env_state = copy.deepcopy(env.sessions)
    successful_trajectories = []
    unsuccessful_trajectories = []
    not_finished_trajectories = []
    failed_trajectories = []
    reflection_map = []

    if to_print:
        print(f"{idx}: {x}")

    cur_task = x
    instruction_path = "../prompt/instructions/webshop_inst.txt"
    icl_path = "../prompt/icl_examples/webshop_icl.json"
    with open(instruction_path) as f:
        instruction = f.read()
    raw_icl = json.load(open(icl_path))
    observation, messages = prompt_with_icl(instruction, raw_icl, cur_task, 1)
    root.messages = messages

    # Main Loop
    for i in range(iterations):
        logging.info(f"Iteration {i + 1}")
        node = root  # Always start from the root node
        depth = 0

        # Perform a simulation from the root
        while not node.is_terminal and depth < args.max_depth:
            expand_node(node, args, task, idx, max_depth=args.max_depth)  # Expand current node
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


    best_tree_child = node
    # best_trajectory_index_list = collect_trajectory_index(best_tree_child)
    task_dict = root.to_dict()
    task_dict['best child reward'] = best_tree_child.reward
    # task_dict['best_trajectory_index_list'] = best_trajectory_index_list
    if args.add_fixed_prefix:
        task_id = idx.replace("fixed_","")
    else:
        task_id = idx
    json.dump(task_dict, open(os.path.join(trajectories_save_path, f"{task_id}.json"), 'w'), indent=4)

    end_time = time.time()
    
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
                       dpo_policy_model=None, dpo_reference_model=None, tokenizer=None):
    global gpt
    global failed_trajectories
    global reflection_map
    action = 'reset'
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    x = env.step(idx, action, args.enable_seq_mode)[0]
    root = Node(state=None, question=x)
    root.env_state = copy.deepcopy(env.sessions)
    successful_trajectories = []
    unsuccessful_trajectories = []
    not_finished_trajectories = []
    failed_trajectories = []
    reflection_map = []

    if to_print:
        print(f"{idx}: {x}")

    cur_task = x
    instruction_path = "prompt/instructions/webshop_inst.txt"
    icl_path = "prompt/icl_examples/webshop_icl.json"
    with open(instruction_path) as f:
        instruction = f.read()
    raw_icl = json.load(open(icl_path))
    observation, messages = prompt_with_icl(instruction, raw_icl, cur_task, 1)
    root.messages = messages



    # logging.info(f"Iteration {i + 1}")
    node = root  # Always start from the root node
    depth = 0

    # Perform a simulation from the root
    while not node.is_terminal and depth < args.max_depth:
        expand_node(node, args, task, idx, max_depth=args.max_depth)  # Expand current node
        if not node.children:
            break  # If no child can be generated, break
        # node = random.choice(node.children)  # Randomly select a child node
        node = beam_search(node, dpo_policy_model, dpo_reference_model, args.q_value_conv_template, tokenizer)
        depth += 1

    # Check the terminal condition
    if node.is_terminal and node.reward == 1:
        logging.info(f"Successful trajectory found")
        successful_trajectories.append(node)
    elif node.is_terminal and node.reward < 1:
        logging.info(f"Unsuccessful trajectory found")
        unsuccessful_trajectories.append(node)
    elif not node.is_terminal:
        logging.info(f"Not finished trajectory found")
        not_finished_trajectories.append(node)

    best_tree_child = node
    # best_trajectory_index_list = collect_trajectory_index(best_tree_child)
    task_dict = root.to_dict()
    task_dict['best child reward'] = best_tree_child.reward
    # task_dict['best_trajectory_index_list'] = best_trajectory_index_list
    if args.add_fixed_prefix:
        task_id = idx.replace("fixed_","")
    else:
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
        return 0,0,0,0


def rollout(node, args, task, idx, max_depth=15):
    depth = node.depth
    n = args.rollout_width
    while not node.is_terminal and depth < max_depth:
        # Generate new states
        new_states = []
        values = []
        while len(new_states) == 0:
            new_states = generate_new_states(node, args, task, idx, n)

        for state in new_states:
            if state.is_terminal:
                return state
                
        child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        #new_state = new_state[0]
        while len(values) == 0:
            values = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
        
        max_value_index = values.index(max(values))
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            node.reward = -0.5
    return node  


def rollout_random(node, args, task, idx, max_depth=15):
    depth = node.depth
    n = args.rollout_width
    assert n==1, "large rollout_width is meanless for rollout_random"
    while not node.is_terminal and depth < max_depth:
        # Generate new states
        new_states = []
        values = []
        while len(new_states) == 0:
            if args.enable_fastchat_conv:
                new_states = generate_new_states_fastchat_conv(node, args, task, idx, n)
            else:
                new_states = generate_new_states(node, args, task, idx, n)

        for state in new_states:
            if state.is_terminal:
                return state
                
        # child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        # #new_state = new_state[0]
        # while len(values) == 0:
        #     values = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
        
        # max_value_index = values.index(max(values))
        max_value_index=random.randint(0,len(new_states)-1)
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            node.reward = -0.5
    return node  


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
                    critique_prompt_template = template_huan.replace('{scenario_description}', webshop_description)
                else:
                    raise NotImplementedError
                new_states_two = generate_new_states_critique_fastchat_conv(node, args, task, idx, 
                                                                        n=2, critique_prompt_template=critique_prompt_template)
                new_states = new_states_two[-1:]
            else:
               raise NotImplementedError

        for state in new_states_two:
            if state.is_terminal:
                return state
                

        node = new_states[0] 
        depth += 1
        if depth == max_depth:
            node.reward = -0.5
    return node  



def rollout_with_q(node, args, task, idx, max_depth, dpo_policy_model, dpo_reference_model, conv_template, tokenizer, puct_coef):
    depth = node.depth
    n = args.rollout_width
    # assert n==1, "large rollout_width is meanless for rollout_random"
    while not node.is_terminal and depth < max_depth:
        # Generate new states
        new_states = []
        values = []
        while len(new_states) == 0:
            if args.enable_fastchat_conv:
                new_states = generate_new_states_fastchat_conv(node, args, task, idx, n)
            else:
                new_states = generate_new_states(node, args, task, idx, n)

        for state in new_states:
            if state.is_terminal:
                return state
        
        q_value_list = get_node_children_q_value_list(node, new_states, dpo_policy_model, dpo_reference_model, conv_template, tokenizer)
                
        max_value_index=q_value_list.index(max(q_value_list))
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            node.reward = -0.5
    return node  


def select_node(node, using_puct=False, dpo_policy_model=None, dpo_reference_model=None, conv_template=None, tokenizer=None, puct_coeff=0):
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
        
        if using_puct:
            
            node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.puct(dpo_policy_model, dpo_reference_model, conv_template, tokenizer, puct_coeff), default=None)

            while node.is_terminal and node.reward != 1:
                node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.puct(dpo_policy_model, dpo_reference_model, conv_template, tokenizer, puct_coeff), default=None)
                
            logging.info(f"Selected node at depth {node.depth} with PUCT {node.puct(dpo_policy_model, dpo_reference_model, conv_template, tokenizer, puct_coeff)}.")
        else: 
            node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)

            while node.is_terminal and node.reward != 1:
                node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
                
            logging.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")
        
    return node 

def expand_node(node, args, task, idx, max_depth):
    n = args.n_generate_sample
    if node.depth >= max_depth: # default value of max_depth is 15
        logging.info("Depth limit reached")
        return
    # if node.depth == 0:  
    #     n *= 2
    if args.enable_fastchat_conv:
        if args.expansion_sampling_method == 'conditional':
            new_nodes = generate_new_states_conditional_fastchat_conv(node, args, task, idx, n)
        elif args.expansion_sampling_method == 'critique':
            if args.critique_prompt_template == 'auto-j':
                critique_prompt_template = auto_j_single_template
            elif args.critique_prompt_template == 'template_v1':
                critique_prompt_template = template_v1
            elif args.critique_prompt_template == 'template_v2':
                critique_prompt_template = template_v2
            elif args.critique_prompt_template == 'template_huan':
                critique_prompt_template = template_huan.replace('{scenario_description}', webshop_description)
            else:
                raise NotImplementedError
            new_nodes = generate_new_states_critique_fastchat_conv(node, args, task, idx, n, critique_prompt_template)
        elif args.expansion_sampling_method == 'vanilla':
            new_nodes = generate_new_states_fastchat_conv(node, args, task, idx, n)
    else:
        new_nodes = generate_new_states(node, args, task, idx, n)
    node.children.extend(new_nodes)

# Copied by  from ETO-webshop-envs
def parse_action(llm_output: str) -> str:
    llm_output = llm_output.strip()
    try:
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        # pattern = re.search(r"(search\[.*?\])", action)
        # if pattern:
        #     action = pattern.group(1)
    except:
        # logging.info("Action Not Found in llm_output: ", llm_output)
        action = 'nothing'
    return action
    

def generate_new_states(node, args, task, idx, n):
    global failed_trajectories
    prompt = generate_prompt(node)
    #print(prompt)
    sampled_actions = get_samples(task, prompt, "\nAction: ", n, prompt_sample=args.prompt_sample, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)
    logging.info(f"SAMPLED ACTION: {sampled_actions}")
    #  added
    # logger = logging.getLogger('loggingtest')
    # logger.info(f"SAMPLED ACTION: {sampled_actions}")
    unique_states = {}  # Store unique states here
    added = False
    for action in sampled_actions:
        local_sessions = copy.deepcopy(node.env_state) # : for roll back
        env.sessions = local_sessions
        logging.info(env.sessions)
        new_state = node.state.copy()  # Make a copy of the parent node's state
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

        # Use thought and action to form a unique key
        unique_key = f"{action_line}"
        
        # if unique_key in unique_states:
        #     continue  # Skip if this state already exists

        if action_line:
            try:
                res = env.step(idx, action_line, args.enable_seq_mode)
                #print("res", res)
                obs = res[0]
                r = res[1]
                done = res[2]
            except AssertionError:
                obs = 'Invalid action!'
                # res = env.step(idx, action_line, args.enable_seq_mode)
                # print("err")
                r = -1
                done = False
            
            if action_line.startswith('think'):
                observation = 'OK.'
      
            # Update the new state dictionary
            new_state['action'] = action_line
            new_state['observation'] = obs
            
            env_state_clone = env.clone_state()  # Clone current environment state
            new_node = Node(state=new_state, question=node.question, env_state=env_state_clone, parent=node)
            new_node.env_state = local_sessions
            if r > 0 or done:
                logging.info(f"reward:{r}")
                new_node.is_terminal = True
                #print("rew", r)
            new_node.reward = r
            new_node.value = r
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")

            if new_node.is_terminal and r < 1.0 and r > 0.0 and added == False:
                trajectory = collect_trajectory_from_bottom(new_node)

                # Check if there is already a failed trajectory with the same reward
                existing_rewards = [t['r'] for t in failed_trajectories]

                if r not in existing_rewards:
                    print("adding to failed")
                    added = True
                    failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_line}", 'r': r})

    return list(unique_states.values())  # Return unique nodes as a list

def get_context(node, args, backend):
    if "gpt-" in backend:
        messages = get_messages_from_bottom(node)
        context =  messages
    elif "Phi-3" in backend or "llama31" in backend or "Llama31" in backend:
        conv = get_conv_from_bottom(node, args.conv_template)
        conv.append_message(conv.roles[1], None)
        context =  conv
    else:
        raise NotImplementedError
    return context


def get_critique_context(node, args, backend):
    if "gpt-" in backend:
        messages = get_messages_from_bottom_critique(node)
        context =  messages
    elif "Phi-3" in backend or "llama31" in backend or "Llama31" in backend:
        conv = get_conv_from_bottom_critique(node, args.conv_template)
        conv.append_message(conv.roles[1], None)
        context =  conv
    else:
        raise NotImplementedError
    return context

def generate_new_states_fastchat_conv(node, args, task, idx, n):
    global failed_trajectories
    assert args.enable_fastchat_conv

    context = get_context(node, args, args.backend)
    response_list = gpt(context, n=n, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)
    sampled_actions = ["\nAction: "+parse_action(response) for response in copy.deepcopy(response_list)]


    logging.info(f"SAMPLED ACTION: {sampled_actions}")
    #  added
    # logger = logging.getLogger('loggingtest')
    # logger.info(f"SAMPLED ACTION: {sampled_actions}")
    unique_states = {}  # Store unique states here
    added = False
    for index, action in enumerate(sampled_actions):
        local_sessions = copy.deepcopy(node.env_state) # : for roll back
        env.sessions = local_sessions
        logging.info(env.sessions)
        new_state = node.state.copy()  # Make a copy of the parent node's state
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

        # Use thought and action to form a unique key
        unique_key = f"{action_line}"
        
        # if unique_key in unique_states:
        #     continue  # Skip if this state already exists

        if action_line:
            try:
                # buy now action cost very long time
                # start_time = time.time()
                res = env.step(idx, action_line, args.enable_seq_mode)
                # end_time = time.time()
                # print("env.step action_line 执行时间: {:.4f} 秒".format(end_time - start_time))
                obs = res[0]
                r = res[1]
                done = res[2]
            except AssertionError:
                obs = 'Invalid action!'
                # res = env.step(idx, action_line, args.enable_seq_mode)
                # print("err")
                r = -1
                done = False
            
            if action_line.startswith('think'):
                observation = 'OK.'
      
            # Update the new state dictionary
            new_state['action'] = response_list[index]
            new_state['observation'] = f"Observation: \n{obs}"
            
            env_state_clone = env.clone_state()  # Clone current environment state
            new_node = Node(state=new_state, question=node.question, env_state=env_state_clone, parent=node)
            new_node.env_state = local_sessions
            if r > 0 or done:
                logging.info(f"reward:{r}")
                new_node.is_terminal = True
                #print("rew", r)
            new_node.reward = r
            new_node.value = r
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")

            if new_node.is_terminal and r < 1.0 and r > 0.0 and added == False:
                trajectory = collect_trajectory_from_bottom(new_node)

                # Check if there is already a failed trajectory with the same reward
                existing_rewards = [t['r'] for t in failed_trajectories]

                if r not in existing_rewards:
                    print("adding to failed")
                    added = True
                    failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_line}", 'r': r})
    if len(list(unique_states.values())) == 0:
        print('sss')
    return list(unique_states.values())  # Return unique nodes as a list


def generate_new_states_conditional_fastchat_conv(node, args, task, idx, n):
    global failed_trajectories
    assert args.enable_fastchat_conv

    sampled_response_list = []
    sampled_obs_list = []

    unique_states = {}  # Store unique states here
    added = False

    for sampling_index in range(n):
        context = copy.deepcopy(get_context(node, args))
        # conditional generation
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
        action = "\nAction: "+parse_action(response)
    

        local_sessions = copy.deepcopy(node.env_state) # : for roll back
        env.sessions = local_sessions
        logging.info(env.sessions)
        new_state = node.state.copy()  # Make a copy of the parent node's state
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

        # Use thought and action to form a unique key
        unique_key = f"{action_line}"
        
        # if unique_key in unique_states:
        #     continue  # Skip if this state already exists

        if action_line:
            try:
                # buy now action cost very long time
                # start_time = time.time()
                res = env.step(idx, action_line, args.enable_seq_mode)
                # end_time = time.time()
                # print("env.step action_line 执行时间: {:.4f} 秒".format(end_time - start_time))
                obs = res[0]
                r = res[1]
                done = res[2]
            except AssertionError:
                obs = 'Invalid action!'
                # res = env.step(idx, action_line, args.enable_seq_mode)
                # print("err")
                r = -1
                done = False

            sampled_obs_list.append(obs)
            
            if action_line.startswith('think'):
                observation = 'OK.'
      
            # Update the new state dictionary
            new_state['action'] = response
            new_state['observation'] = f"Observation: \n{obs}"
            
            env_state_clone = env.clone_state()  # Clone current environment state
            new_node = Node(state=new_state, question=node.question, env_state=env_state_clone, parent=node)
            new_node.env_state = local_sessions
            if r > 0 or done:
                logging.info(f"reward:{r}")
                new_node.is_terminal = True
                #print("rew", r)
            new_node.reward = r
            new_node.value = r
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")

            if new_node.is_terminal and r < 1.0 and r > 0.0 and added == False:
                trajectory = collect_trajectory_from_bottom(new_node)

                # Check if there is already a failed trajectory with the same reward
                existing_rewards = [t['r'] for t in failed_trajectories]

                if r not in existing_rewards:
                    print("adding to failed")
                    added = True
                    failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_line}", 'r': r})
    if len(list(unique_states.values())) == 0:
        print('sss')
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
    for message in context.messages[11:]:
        if message[1] is not None:
            prompt += message[1].strip() + '\n'
    return prompt

def generate_new_states_critique_fastchat_conv(node, args, task, idx, n, critique_prompt_template):
    global failed_trajectories
    assert args.enable_fastchat_conv

    previous_response = None
    previous_obs = None

    unique_states = {}  # Store unique states here
    added = False

    for sampling_index in range(n):
        context = copy.deepcopy(get_context(node, args, args.backend))
        critique = None
        regenerate_prompt = None
        if previous_response:
            if not args.critique_backend:
                args.critique_backend = args.backend
            critique_context = copy.deepcopy(get_context(node, args, args.critique_backend))
            # generating critique
            if args.critique_prompt_template == 'template_huan':
                original_observation = get_raw_observation(context.messages[-2][1])
                critique_prompt_templat = critique_prompt_template.format(
                    user_inst=critique_context.messages[10][1],
                    historical_context=get_historical_context(critique_context) + f"{previous_response}\n{previous_obs}",
                )
                critique_context.messages = [['system', critique_prompt_templat.split('</system>')[0]],
                                             ['user', critique_prompt_templat.split('</system>')[-1]],
                                             ['assistant', None]]
                critique = critique_gpt(critique_context, n=1, stop=["Observation:",], enable_fastchat_conv=args.enable_fastchat_conv)[0]
                if critique.startswith('Critique:'):
                    critique = critique[9:]
                regenerate_prompt = '\nBelow are the previous Thought and Action you generated along with their corresponding Observation:\n'
                regenerate_prompt += f"<previous Thought, Action, and Observation>\n{previous_response}\n{previous_obs}\n</previous Thought, Action, and Observation>\n"
                regenerate_prompt += f'Critique:\n<critique>{critique}</critique>\n'
                regenerate_prompt += 'Based on the critique, generate a new Thought and Action with as much distinctiveness as possible for the current Observation:' + "\n"
                original_observation = "<current observation>\nObservation:" + original_observation.replace("Observation:", '') + '</current observation>\n'
                constraint = "Consider the feasibility of new actions based on the current observation, rather than relying on the previous observation."
                context.messages[-2][1] += regenerate_prompt + "\n" + original_observation + '\n' + constraint
            else:
                critique_prompt = critique_prompt_template.format(previous_response=previous_response, previous_obs=previous_obs)
                if isinstance(critique_context, list):  # for openai GPT
                    original_observation = get_raw_observation(critique_context[-1]['content'])
                    critique_context[-1]['content'] += critique_prompt + "\n"
                else: # for fastchat
                    original_observation = get_raw_observation(critique_context.messages[-2][1])
                    critique_context.messages[-2][1] += critique_prompt + "\n"
                critique = critique_gpt(critique_context, n=1, stop=["Observation:", ], enable_fastchat_conv=args.enable_fastchat_conv)[0]
                regenerate_prompt = '\n\nBelow are the previous Thought and Action you generated along with their corresponding Observation: \n\n'
                regenerate_prompt += previous_response + "\n"
                regenerate_prompt += previous_obs + "\n"
                regenerate_prompt += 'Critique: ' + critique + "\n\n"
                regenerate_prompt += 'Based on the critique, generate a new Thought and Action with as much distinctiveness as possible for the Observation:' + "\n"
                context.messages[-2][1] += regenerate_prompt + "\n" + original_observation

        response = gpt(context, n=1, stop="Observation", enable_fastchat_conv=args.enable_fastchat_conv)[0]
        previous_response = response
        action = "\nAction: "+parse_action(response)
    

        local_sessions = copy.deepcopy(node.env_state) # : for roll back
        env.sessions = local_sessions
        logging.info(env.sessions)
        new_state = node.state.copy()  # Make a copy of the parent node's state
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

        # Use thought and action to form a unique key
        unique_key = f"{action_line}"
        
        # if unique_key in unique_states:
        #     continue  # Skip if this state already exists

        if action_line:
            try:
                # buy now action cost very long time
                # start_time = time.time()
                res = env.step(idx, action_line, args.enable_seq_mode)
                # end_time = time.time()
                # print("env.step action_line 执行时间: {:.4f} 秒".format(end_time - start_time))
                obs = res[0]
                r = res[1]
                done = res[2]
            except AssertionError:
                obs = 'Invalid action!'
                # res = env.step(idx, action_line, args.enable_seq_mode)
                # print("err")
                r = -1
                done = False

            previous_obs = f"Observation: \n{obs}"
            
            if action_line.startswith('think'):
                observation = 'OK.'
      
            # Update the new state dictionary
            new_state['action'] = response
            new_state['observation'] = f"Observation: \n{obs}"

            new_state['critique'] = critique
            new_state['regenerate_prompt'] = regenerate_prompt

            
            env_state_clone = env.clone_state()  # Clone current environment state
            new_node = Node(state=new_state, question=node.question, env_state=env_state_clone, parent=node)
            new_node.env_state = local_sessions
            if r > 0 or done:
                logging.info(f"reward:{r}")
                new_node.is_terminal = True
                #print("rew", r)
            new_node.reward = r
            new_node.value = r
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")

            if new_node.is_terminal and r < 1.0 and r > 0.0 and added == False:
                trajectory = collect_trajectory_from_bottom(new_node)

                # Check if there is already a failed trajectory with the same reward
                existing_rewards = [t['r'] for t in failed_trajectories]

                if r not in existing_rewards:
                    print("adding to failed")
                    added = True
                    failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_line}", 'r': r})
    if len(list(unique_states.values())) == 0:
        print('sss')
    return list(unique_states.values())  # Return unique nodes as a list



def evaluate_node(node, args, task, idx):
    #actions_to_node = collect_actions_to_node(node)
    #env.restore_state(actions_to_node, idx)
    
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]

    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
    
    logging.info(f"Length of votes: {len(votes)}")
    logging.info(f"Length of node.children: {len(node.children)}")
    
    # Pre-allocate votes list
    votes = votes + [0] * (len(node.children) - len(votes))
    
    max_vote = max(votes) if votes else 1
    if max_vote == 0:
        max_vote = 1  # Avoid division by zero
    
    terminal_conditions = [1 if child.is_terminal else 0 for child in node.children]
    for i, condition in enumerate(terminal_conditions):
        if condition == 1:
            votes[i] = max_vote + 1
    
    for i, child in enumerate(node.children):
        child.value = votes[i] / max_vote  # Now safe from division by zero
    
    return sum(votes) / len(votes) if votes else 0


def print_tree(node, level=0):
    indent = "  " * level
    print(f"{indent}{node}")
    for child in node.children:
        print_tree(child, level + 1)

def backpropagate(node, value): # here, value refers to reward of terminal node
    while node:
        node.visits += 1
        node.value = (node.value * (node.visits - 1) + value) / node.visits
        logging.info(f"Backpropagating with reward {value} at depth {node.depth}. New value: {node.value}.")
        # else:
        #     node.value = (node.value * (node.visits - 1) + value) / node.visits
        #     logging.info(f"Backpropagating at depth {node.depth}. New value: {node.value}.")

        node = node.parent

def generate_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['action']:
            new_segment.append(f"Action: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
            new_segment.append(f"Observation: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n\n'.join(reversed(trajectory))



def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    global reflection_map
    global failed_trajectories
    #unique_trajectories = get_unique_trajectories(failed_trajectories)
    value_prompt = task.value_prompt_wrap(x, y, failed_trajectories, reflection_map)
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

def get_values(task, x, ys, n_evaluate_sample, cache_value=False):
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

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop, enable_fastchat_conv=False):
    global reflection_map
    global failed_trajectories
    #print("MCTS FAIELD", failed_trajectories)
    #unique_trajectories = get_unique_trajectories(failed_trajectories)
    #unique_trajectories = failed_trajectories
    #print(len(unique_trajectories))
    #print(len(reflection_map))
    if len(failed_trajectories) > len(reflection_map) and len(failed_trajectories) < 4:
        print("generating reflections")
        print("len(failed_trajectories): ",len(failed_trajectories))
        print("len(reflection_map): ", len(reflection_map))
        reflection_map = task.generate_self_reflection(failed_trajectories, x)
    # : maybe cot is actually react according?
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, '\n'+y, reflection_map)
    # elif prompt_sample == 'fastchat_eto':
    #     prompt = task.fastchat_eto_prompt_wrap(x, '\n'+y, reflection_map)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    logging.info(f"PROMPT: {prompt}")
    samples = gpt(prompt, n=n_generate_sample, stop=stop, enable_fastchat_conv=enable_fastchat_conv)
    return [y + _ for _ in samples]

def get_unique_trajectories(failed_trajectories, num=3):
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