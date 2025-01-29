import itertools
import numpy as np
from functools import partial
from torch.ao.quantization.backend_config.onednn import observation_type
import sys
from kgqa_lintong.tog.utils import *
from kgqa_lintong.tog.freebase_func import *
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

# Initialize environment
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="train")
env = wrappers.LoggingWrapper(env)

# Global variables
global reflection_map
global failed_trajectories
global wiki_explored
global enable_wiki_search

reflection_map = []
failed_trajectories = []


def step(env, action):
    """Attempts to perform a step in the environment with retries on timeout."""
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1


def triple_scores(triples, question, thought, args):
    """
    Evaluates the helpfulness of triples in answering a question using llm.
    Returns scores for each triple.
    """
    scores = [0]
    prompt = f"Question: {question}\nInformation: {thought}\n"
    for i, triple in enumerate(triples):
        prompt += f"Triple {i + 1}: {triple}\n"
    prompt += ("Please evaluate how much each triplet helps in answering the question on a scale from 0 to 1, "
               "where 0 means not helpful at all and 1 means very helpful (the sum of the scores of all triplets is 1). "
               "Provide the scores in square brackets, e.g., [0.5, 0.2, 0.3].\n\nExamples:\n"
               "1. Question: What is the capital of France?\nInformation: The capital of France is a well-known city.\n"
               "Triple 1: (France, capital, Paris)\nTriple 2: (France, largest city, Paris)\n"
               "Triple 3: (France, language, French)\nScores: [0.7, 0.2, 0.1]")

    # Call GPT model to get scores
    if args.enable_fastchat_conv and 'lama' in args.backend:
        score_output = llama31_instruct(prompt, model=args.backend, n=1)[0]
    else:
        score_output = gpt(prompt, n=1, stop=None)[0].strip()

    # Parse scores
    try:
        score_str = score_output.split('[')[1].split(']')[0]
        scores = list(map(float, score_str.split(',')))
        scores = [score if 0 <= score <= 1 else 0.0 for score in scores]
    except (IndexError, ValueError):
        scores = [0.0] * len(triples)

    return scores


def path_score(node, args):
    """
    Evaluates the helpfulness of a path in answering a question using a GPT model.
    Returns the score for the path.
    """
    score = 0
    node_path = []
    node_path.append(node.triple)
    this_node = node
    while this_node.parent:
        this_node = this_node.parent
        node_path.insert(0, this_node.triple)
    node_path = str(node_path)

    prompt = (f"Question: {node.question}\nWikiinformation: {wiki_explored}\nPath: {node_path}\n"
              "Please evaluate how much the path helps in answering the question on a scale from 0 to 1, "
              "where 0 means not helpful at all and 1 means very helpful. You may refer to relevant information from Wikipedia to assist in your assessment. Provide the score in square brackets, e.g., [0.5]\n\n"
              "Examples:\n1. Question: What is the capital of France?\nWikiinformation: France is a country located in Western Europe. Its capital and largest city is Paris, which is also the capital of the region of Île-de-France. \nPath: [[France, located_in, Europe], [Europe, capital_of_country, Paris]]\nScore: [0.9]\n\n"
              "2. Question: Who wrote 'To Kill a Mockingbird'?\nWikiinformation: 'To Kill a Mockingbird' is renowned for its exploration of racial injustice and moral growth, set in the Deep South of the United States. \nPath: [[To Kill a Mockingbird, genre, Novel], [Novel, written_by, Harper Lee]]\nScore: [0.8]\n\n"
              "3. Question: What is the tallest mountain in the world?\nWikiinformation: Mount Everest is the highest mountain in the world, with its peak at an elevation of 8,848 meters (29,029 feet) above sea level.\nPath: [[Earth, has_feature, Mount Everest], [Mount Everest, height, 8,848 meters]]\nScore: [0.7]")

    # Call GPT model to get score
    if args.enable_fastchat_conv and 'lama' in args.backend:
        score_output = llama31_instruct(prompt, model=args.backend, n=1)[0]
    else:
        score_output = gpt(prompt, n=1, stop=None)[0].strip()

    # Parse score
    try:
        score_str = score_output.split('[')[1].split(']')[0]
        scores = float(score_str)
    except (IndexError, ValueError):
        scores = 0

    return score


def get_value(task, x, y, n_evaluate_sample, args, cache_value=True):
    """
    Evaluates the value of a given task using the GPT model.
    """
    global reflection_map
    global failed_trajectories
    unique_trajectories = get_unique_trajectories(failed_trajectories)

    value_prompt = (y + "Please evaluate how much the triplet helps in answering the question on a scale from 0 to 1, "
                        "where 0 means not helpful at all and 0.9 means very helpful. Provide the scores in square brackets, e.g., [0.8, 0.5, 0.3].\n\n"
                        "Examples:\n1. Question: What is the capital of France?\nThought: The capital of France is a well-known city.\n"
                        "Action: Choose[France, capital, Paris]\nScores: [0.8]")

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
    """Evaluates multiple values for a task."""
    values = []
    local_value_cache = {}
    for y in ys:
        if y in local_value_cache:
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, args, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    """
    Generates samples using the GPT model.
    """
    global failed_trajectories
    global reflection_map
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    if len(unique_trajectories) > len(reflection_map) and len(unique_trajectories) < 4:
        print("Generating reflections")
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
    """
    Extracts unique trajectories from failed trajectories.
    """
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
    """
    Saves the node and its information to a JSON file.
    """
    all_tree_nodes_list = collect_all_nodes(node)
    best_tree_child = max(all_tree_nodes_list, key=lambda x: x.reward)
    best_trajectory_index_list = collect_trajectory_index(best_tree_child)

    task_dict = node.to_dict()
    all_tree_nodes_list.extend(terminal_nodes)
    best_child = max(all_tree_nodes_list, key=lambda x: x.reward)
    task_dict['best reward'] = best_child.reward
    task_dict['answer'] = best_child.answer
    task_dict['best em'] = best_child.em
    task_dict['best child reward'] = best_tree_child.reward
    task_dict['best child em'] = best_tree_child.em
    task_dict['best_trajectory_index_list'] = best_trajectory_index_list
    task_dict['true_answer'] = node.true_answer
    task_dict['check'] = node.check
    task_dict['wiki'] = wiki_explored
    task_id = idx

    json.dump(task_dict, open(os.path.join(trajectories_save_path, f"{task_id}.json"), 'w'), indent=4)


def get_reasoning_chain(node):
    """
    Constructs a reasoning chain from the given node.
    """
    messages = []
    while node.parent:
        messages.insert(0, node.triple)
        messages.append(node.triple)
        node = node.parent
    return messages


def reasoning(reasoning_chain, question, args):
    """
    Determines if the reasoning chain can answer the question.
    """
    prompt = prompt_evaluate + question
    chain_prompt = ''
    for sublist in reasoning_chain:
        chain_prompt += str(sublist)
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    if args.enable_fastchat_conv and 'lama' in args.backend:
        result = llama31_instruct(prompt, model=args.backend, n=1)[0]
    else:
        result = gpt(prompt, n=1)[0]

    result = extract_answer(result)
    return if_true(result)


def get_answer(reasoning_chain, question, args):
    """
    Retrieves the answer to the question using the reasoning chain.
    """
    prompt = answer_prompt + question + '\n'
    chain_prompt = ''
    for sublist in reasoning_chain:
        chain_prompt += str(sublist)
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    if args.enable_fastchat_conv and 'lama' in args.backend:
        result = llama31_instruct(prompt, model=args.backend, n=1)[0]
    else:
        result = gpt(prompt, n=1)[0]

    return result


def check_string(string):
    """
    Checks if the given string contains a specific character.
    """
    if string is None:
        return False
    return "{" in string


def fschat_mcts_search(args, task, idx, iterations=50, to_print=True, trajectories_save_path=None,
                       dpo_policy_model=None, dpo_reference_model=None, tokenizer=None, enable_reflection=False):
    """
    Performs MCTS search with the given arguments and task.
    """
    global gpt
    global failed_trajectories
    global reflection_map

    global information_explored
    global wiki_explored
    global enable_wiki_search

    enable_wiki_search = args.enable_wiki_search
    information_explored = ""
    wiki_explored = ""

    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    global critique_gpt
    if args.expansion_sampling_method == "critique":
        if args.critique_backend:
            if args.critique_temperature is None:
                args.critique_temperature = args.temperature
            critique_gpt = partial(gpt, model=args.critique_backend, temperature=args.critique_temperature)
        else:
            critique_gpt = gpt

    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a')

    x = env.reset(idx=idx)
    if to_print:
        print(idx, x)

    root = Node(state=None, question=x[0], topic_entity=x[1], true_answer=x[2])

    next_entity_relations_list, next_entity_list, next_chain_list = find_next_triples(args.n_generate_sample, root,
                                                                                      args)
    root.state['observation'] = f"Here are some candidate knowledge triplets you can choose from: " + str(
        next_chain_list)
    root.next_triple_list = next_chain_list
    root.next_entity_relations_list = next_entity_relations_list
    root.next_entity_list = next_entity_list
    root.depth = 0

    cur_task = x[0]

    if enable_reflection:
        instruction_path = "../prompt/instructions/mygraph_inst_reflection.txt"
        icl_path = "../prompt/icl_examples/mygraph_icl_reflection.json"
    else:
        instruction_path = "../prompt/instructions/mygraph_inst_new.txt"
        icl_path = "../prompt/icl_examples/mygraph_inst_new.json"

    with open(instruction_path) as f:
        instruction = f.read()

    raw_icl = json.load(open(icl_path, encoding='utf-8'))
    observation, messages = prompt_with_icl(instruction, raw_icl, cur_task, 3)

    assert messages[-1]['role'] == 'user'
    messages[-1]['content'] += ' Observation: ' + root.state['observation']

    root.messages = messages

    all_nodes = []
    failed_trajectories = []
    reflection_map = []
    terminal_nodes = []

    for i in range(iterations):
        node = select_node(root, args, i)

        while node is None or (node.is_terminal and node.reward != 1):
            logging.info(f"Need to backtrack or terminal node with reward 0 found at iteration {i + 1}, reselecting...")
            last_selected_node = copy.deepcopy(node)
            node = select_node(root, args, i)
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
                all_nodes_list = collect_all_nodes(root)
                if all(node.check == 0 for node in all_nodes_list):
                    root.check = 0
                else:
                    root.check = 1
                save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)
                return node.state, node.value, all_nodes, node.reward, node.em

        if node.is_terminal and node.reward != 1:
            logging.info(f"There is not enough information in the knowledge graph.")
            break

        expand_node(node, args, task, args.max_depth)

        while node.is_terminal or not node.children:
            logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root, args, i)
            if node is None:
                logging.info("All paths lead to terminal nodes with reward 0. Ending search.")
                break
            expand_node(node, args, task, args.max_depth)
            if node.depth >= args.max_depth:
                break

        if node is None:
            logging.info("All paths lead to terminal nodes with reward 0. Ending search.")
            break

        if args.enable_rollout_with_critique:
            reward, terminal_node = rollout_with_critique(max(node.children, key=lambda child: child.value), args, task,
                                                          idx, max_depth=args.max_depth)
        else:
            if node.children:
                selected_child = max(node.children, key=lambda child: child.value)
                reward, terminal_node = rollout_random(selected_child, args, task, idx, max_depth=args.max_depth)
            else:
                reward, terminal_node = node.reward, node

        terminal_nodes.append(terminal_node)

        if args.enable_rollout_early_stop:
            if terminal_node.reward == 1:
                logging.info("Successful trajectory found")
                logging.info(f"Terminal node including rollouts with reward 1 found at iteration {i + 1}")
                backpropagate(terminal_node, reward)
                all_nodes_list = collect_all_nodes(root)
                if all(node.check == 0 for node in all_nodes_list):
                    root.check = 0
                else:
                    root.check = 1
                save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)
                return terminal_node.state, terminal_node.value, terminal_node.reward, terminal_node.em

        reward = path_score(terminal_node, args)
        backpropagate(terminal_node, reward)
        all_nodes = [(node, node.value) for node in collect_all_nodes(root)]

        terminal_nodes_with_reward_1 = [node for node in collect_all_nodes(root) if
                                        node.is_terminal and node.reward == 1]
        if terminal_nodes_with_reward_1:
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.value)
            if args.disable_early_stop:
                continue
            else:
                all_nodes_list = collect_all_nodes(root)
                if all(node.check == 0 for node in all_nodes_list):
                    root.check = 0
                else:
                    root.check = 1
                save_node_to_json(root, terminal_nodes, idx, trajectories_save_path)
                return best_node.state, best_node.value, best_node.reward, best_node.em

        for j, (node, value) in enumerate(all_nodes):
            logging.info(f"Node {j + 1}: {str(node)}")
        logging.info(f"State of all_nodes after iteration {i + 1}: {all_nodes}")

    all_nodes_list = collect_all_nodes(root)
    for node in all_nodes_list:
        if node.is_terminal and node.value == 0:
            backpropagate(node, node.reward)

    if all(node.reward != 1 for node in all_nodes_list):
        node_children_relation = []
        for child in all_nodes_list:
            node_children_relation.append(child.triple)
        chain_prompt = ''
        for sublist in node_children_relation:
            chain_prompt += str(sublist)
        myenv = wikienv.WikiEnv()
        this_time_imformation = ""
        keywords = generate_query(root.question, chain_prompt, wiki_explored)
        if len(keywords) != 0:
            for keyword in keywords:
                this_time_imformation += myenv.search_step(keyword)
        else:
            keywords = list(root.topic_entity.values())
            for keyword in keywords:
                this_time_imformation += myenv.search_step(keyword)

        if enable_wiki_search:
            wiki_explored += this_time_imformation[:1000] + '\n'
            this_time_imformation = this_time_imformation[:1000]
        else:
            wiki_explored = ''

        prompt = answer_wiki + root.question + '\n'
        prompt += "\nKnowledge Triplets: " + chain_prompt
        prompt += "\nwikipedia: " + this_time_imformation + 'A: '

        if args.enable_fastchat_conv and 'lama' in args.backend:
            result = llama31_instruct(prompt, model=args.backend, n=1)[0]
        else:
            result = gpt(prompt, n=1)[0]

        if check_string(result):
            response = clean_results(result)
            if response == "NULL":
                root.check = 0
            else:
                root.answer = response
                if exact_match(response, root.true_answer):
                    root.check = 1

    if all(node.check == 0 for node in all_nodes_list):
        root.check = 0
    else:
        root.check = 1

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


def select_node(node, args, i=0):
    """
    Selects a node based on UCT values.
    """
    while node and node.children:
        logging.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
        terminal_children = [child for child in node.children if child.is_terminal]

        if len(terminal_children) == len(node.children):
            logging.info(f"All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:
                node.parent.children.remove(node)
            node = node.parent
            if node is None:
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

    return node


def generate_query(question="", myinformation_explored="", mywiki_explored=""):
    """
    Generates a query for additional information needed to answer a question.
    """
    global gpt
    prompt = (
        f"Given the question and the existing information, the existing knowledge triplets, what additional information is needed to answer the question? "
        "Please provide a list of keywords (no more than 3) that can be used for further Wikipedia search. Here are some examples:\n"
        "Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?\n"
        "Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson\n"
        "existing information: Libya, officially the State of Libya, is a country in the Maghreb region of North Africa.\n"
        "Keywords: [Thomas, Jefferson]\n"
        "Q: Who is the coach of the team owned by Steve Bisciotti?\n"
        "Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens\n"
        "Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens\n"
        "Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group\n"
        "existing information: Libya, officially the State of Libya, is a country in the Maghreb region of North Africa.\n"
        "Keywords: [Baltimore Ravens]\n"
        "Q: The country with the National Anthem of Bolivia borders which nations?\n"
        "Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity\n"
        "National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti\n"
        "National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés\n"
        "UnName_Entity, government.national_anthem_of_a_country.country, Bolivia\n"
        "Bolivia, location.country.national_anthem, UnName_Entity\n"
        "existing information: Libya, officially the State of Libya, is a country in the Maghreb region of North Africa.\n"
        "Keywords: [Bolivia]\n"
        f"Q: {question}\n"
        f"Knowledge Triplets: {myinformation_explored}\n"
        f"existing information: {mywiki_explored}\n"
        "Keywords:")

    response = gpt(prompt, n=1, stop=None)[0]

    match = re.search(r'Keywords:\s*\[(.*?)\]', response)
    if match:
        keywords = [keyword.strip() for keyword in match.group(1).split(",")]
    else:
        keywords = []

    return keywords


def expand_node(node, args, task, max_depth):
    """
    Expands a node by generating new states.
    """
    n = args.n_generate_sample
    if node.depth >= max_depth:
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        return

    assert args.expansion_sampling_method == 'vanilla' or args.enable_fastchat_conv

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

    if enable_wiki_search:
        node_triplets = []
        for child in new_nodes:
            node_triplets.append(child.triple)
        if len(node_triplets) != 0:
            node_chain = []
            node_chain.append(node.triple)
            this_node = node
            while this_node.parent:
                this_node = this_node.parent
                node_chain.insert(0, this_node.triple)
            triplets_value = triple_scores(node_triplets, node.question, '', args)
            for value, child in zip(triplets_value, new_nodes):
                child.rawvalue = value
            this_time_imformation = ""
            myenv = wikienv.WikiEnv()
            global wiki_explored
            keywords = generate_query(node.question, str(node_chain), wiki_explored)
            if len(keywords) != 0:
                for keyword in keywords:
                    this_time_imformation += myenv.search_step(keyword)
            else:
                keywords = list(node.topic_entity.values())
                for keyword in keywords:
                    this_time_imformation += myenv.search_step(keyword)
            wiki_explored += this_time_imformation[:1000] + '\n'
            triplets_value = triple_scores(node_triplets, node.question, this_time_imformation[:1000], args)
            while len(triplets_value) != len(new_nodes):
                triplets_value = triple_scores(node_triplets, node.question, this_time_imformation[:1000], args)
            for value, child in zip(triplets_value, new_nodes):
                child.value = value
                child.wikivalue = value
        else:
            node_triplets = [0]

    node.children = new_nodes


def parse_action(llm_output: str) -> str:
    """
    Parses the action from the LLM output.
    """
    llm_output = llm_output.strip()
    try:
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
    except:
        action = 'nothing'
    assert action is not None
    return action


def parse_thought(llm_output: str) -> str:
    """
    Parses the thought from the LLM output.
    """
    llm_output = llm_output.strip()
    llm_output = llm_output.replace("\n", " ")
    try:
        pattern = re.compile(r"Thought: (.*)(?= Action:)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
    except:
        action = 'nothing'
    assert action is not None
    return action


def rollout(node, args, task, idx, max_depth=7):
    """
    Performs a rollout for a node.
    """
    logging.info("ROLLING OUT")
    depth = node.depth
    n = args.rollout_width
    rewards = [0]

    while not node.is_terminal and depth < max_depth:
        logging.info(f"ROLLING OUT {depth}")
        new_states = []
        values = []

        while len(new_states) == 0:
            new_states = generate_new_states(node, args, task, n)

        for state in new_states:
            if state.is_terminal:
                return state.reward, state

        child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]

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
    """
    Performs a random rollout for a node.
    """
    depth = node.depth
    n = args.rollout_width
    assert n == 1, "large rollout_width is meaningless for rollout_random"

    while not node.is_terminal and depth < max_depth:
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

        max_value_index = random.randint(0, len(new_states) - 1)
        node.children = [new_states[max_value_index]]
        node = new_states[max_value_index]
        depth += 1

        if depth == max_depth:
            reasoning_chain = get_reasoning_chain(node)
            if len(reasoning_chain) == 0:
                node.is_terminal = True
                node.reward = -1
            else:
                answer = reasoning(reasoning_chain, node.question, args)
                if answer:
                    answer = get_answer(reasoning_chain, node.question, args)
                    if check_string(answer):
                        response = clean_results(answer)
                        if response == "NULL":
                            node.is_terminal = True
                            node.reward = -1
                        else:
                            node.is_terminal = True
                            node.reward = 1
                            node.answer = response
                            if exact_match(response, node.true_answer):
                                node.check = 1
                    else:
                        node.is_terminal = True
                        node.reward = -1
                else:
                    node.is_terminal = True
                    node.reward = -1

    return node.reward, node


def rollout_with_critique(node, args, task, idx, max_depth=15):
    """
    Performs a rollout with critique for a node.
    """
    depth = node.depth
    n = args.rollout_width
    assert n == 1, "the same with rollout_random"

    while not node.is_terminal and depth < max_depth:
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
    """
    Generates new states for a node.
    """
    global failed_trajectories
    prompt = generate_prompt(node)
    sampled_actions = get_samples(task, prompt, f"Thought {node.depth + 1}: ", n, prompt_sample=args.prompt_sample,
                                  stop="Observation")
    logging.info(f"SAMPLED ACTION: {sampled_actions}")
    tried_actions = []
    unique_states = {}

    for action in sampled_actions:
        action = action.replace("Thought 1:  Thought 1: ", "Thought 1: ")
        new_state = node.state.copy()
        thought_line = next(
            (line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")),
            '')
        action_line = next(
            (line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line),
            None)

        unique_key = f"{thought_line}::{action_line}"
        if unique_key in unique_states:
            continue

        tried_actions.append(action_line)
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""
            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")

            new_state['thought'] = thought_line
            new_state['action'] = action_line
            new_state['observation'] = f"Observation: {obs}"
            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done
            new_node.reward = r
            new_node.depth = node.depth + 1

            if r == 1:
                new_node.em = info.get('em')

            unique_states[unique_key] = new_node
            logging.info(f"NEW NODE: {new_node}")
            logging.info(f"Feedback: {info}")

            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory_from_bottom(new_node)
                failed_trajectories.append(
                    {'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})

    return list(unique_states.values())


def get_context(node, conv_template, backend):
    """
    Retrieves the context for a node.
    """
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


def clean_results(string):
    """
    Cleans the results from a string.
    """
    if "{" in string:
        start = string.find("{") + 1
        end = string.find("}")
        content = string[start:end]
        return content
    else:
        return "NULL"


def string_to_list(input_string):
    """
    Converts a string to a list.
    """
    match = re.search(r'\[(.*?)\]', input_string)
    if match:
        stripped_string = match.group(1)
        result_list = [item.strip().strip('"').strip("'") for item in stripped_string.split(",")]
    else:
        result_list = []

    return result_list


def find_next_triples(n, node, args):
    """
    Finds the next triples for a node.
    """
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
                entity_candidates_id = my_entity_candidates_id

            scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id,
                                                                           entity['score'], entity['relation'], args)
            total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(
                entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores,
                total_relations, total_entities_id, total_topic_entities, total_head)

        flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id,
                                                                                      total_relations, total_candidates,
                                                                                      total_topic_entities, total_head,
                                                                                      total_scores, args)

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
    else:
        return node.next_entity_relations_list, node.next_entity_list, node.next_triple_list

    return total_relations, new_topic_entity, mychain_of_entities


def exact_match(response, answers):
    """
    Checks if the response matches any of the answers.
    """
    if response is None:
        return False

    clean_result = str(response).strip().replace(" ", "").lower()
    for answer in answers:
        clean_answer = str(answer).strip().replace(" ", "").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True

    return False


def generate_new_states_fastchat_conv(node, args, task, n):
    """
    Generates new states using FastChat Conv.
    """
    global failed_trajectories
    context = get_context(node, args.conv_template, args.backend)
    tried_actions = []
    unique_states = {}

    next_entity_relations_list, next_entity_list, next_chain_list = find_next_triples(n, node, args)
    i = len(next_entity_relations_list)

    if i == 0:
        reasoning_chain = get_reasoning_chain(node)
        chain_prompt = ''
        for sublist in reasoning_chain:
            chain_prompt += str(sublist)
        myenv = wikienv.WikiEnv()
        this_time_imformation = ""
        global wiki_explored
        keywords = generate_query(node.question, chain_prompt, wiki_explored)
        if len(keywords) != 0:
            for keyword in keywords:
                this_time_imformation += myenv.search_step(keyword)
        else:
            keywords = list(node.topic_entity.values())
            for keyword in keywords:
                this_time_imformation += myenv.search_step(keyword)

        if enable_wiki_search:
            wiki_explored += this_time_imformation[:1000] + '\n'
        else:
            wiki_explored = ''

        node.wikiinformation = wiki_explored
        prompt = answer_wiki + node.question + '\n'
        prompt += "\nKnowledge Triplets: " + chain_prompt
        prompt += "\nwikipedia: " + this_time_imformation[:1000] + 'A: '

        if args.enable_fastchat_conv and 'lama' in args.backend:
            result = llama31_instruct(prompt, model=args.backend, n=1)[0]
        else:
            result = gpt(prompt, n=1)[0]

        if check_string(result):
            response = clean_results(result)
            if response == "NULL":
                node.reward = -1
            else:
                node.answer = response
                node.reward = 1
        else:
            response = "NULL"

        action_type = "Finish[]"
        action_param = response
        new_state = node.state.copy()
        if action_type.startswith("Finish[") and action_type.endswith("]"):
            env.env.env.node = node
            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
            new_state['action'] = f"Thought: {action_param} Action: {action_type}"
            new_state['observation'] = f"Observation: {result}"
            unique_key = f"{action_param}::{action_type}::none"
            new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,
                            true_answer=node.true_answer)
            new_node.is_terminal = True
            new_node.reward = r
            if exact_match(action_param, node.true_answer):
                new_node.check = 1
            else:
                new_node.reward = 0
            new_node.depth = node.depth + 1
            if r == 1:
                new_node.em = info.get('em')
            unique_states[unique_key] = new_node
            logging.info(f"NEW NODE: {new_node}")
            logging.info(f"Feedback: {info}")
            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory_from_bottom(new_node)
                failed_trajectories.append(
                    {'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})
        return list(unique_states.values())

    response_list = gpt(context, n=args.n_generate_sample, stop="Observation",
                        enable_fastchat_conv=args.enable_fastchat_conv)
    thought_lines = [parse_thought(response) for response in copy.deepcopy(response_list)]
    action_lines = [parse_action(response) for response in copy.deepcopy(response_list)]
    logging.info(f"SAMPLED ACTION: {action_lines}")

    for thought_line, action_line in zip(thought_lines, action_lines):
        idx = 0
        new_state = node.state.copy()
        unique_key = f"{action_line}"
        if unique_key in unique_states:
            continue

        tried_actions.append(action_line)
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = '[' + action_line.split('[')[1].split(']')[0] + ']' if '[' in action_line else ""
            if action_type.startswith("Finish"):
                unique_key = f"{action_line}"
                if unique_key in unique_states:
                    continue

                tried_actions.append(action_line)
                env.env.env.node = node
                obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
                new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
                new_state['observation'] = f"Answer: {obs}"
                new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,
                                true_answer=node.true_answer)
                new_node.is_terminal = True
                if exact_match(action_param, node.true_answer):
                    new_node.reward = r
                else:
                    new_node.reward = -1
                new_node.depth = node.depth + 1
                if r == 1:
                    new_node.em = info.get('em')
                unique_states[unique_key] = new_node
                logging.info(f"NEW NODE: {new_node}")
                logging.info(f"Feedback: {info}")
                if new_node.is_terminal and r == 0:
                    trajectory = collect_trajectory_from_bottom(new_node)
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
                    new_state['action'] = f"Thought: {thought_line} Action: Choose{action_param}"
                    new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=next_entity,
                                    true_answer=node.true_answer)
                    new_node.triple = str(next_chain)
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
                                    new_node.is_terminal = True
                                    new_node.reward = 1
                                    new_node.answer = response
                                    if exact_match(response, new_node.true_answer):
                                        new_node.check = 1
                    new_node.depth = node.depth + 1
                    new_node.next_entity_relations_list, new_node.next_entity_list, new_node.next_triple_list = find_next_triples(
                        n, new_node, args)
                    new_node.state[
                        'observation'] = f"Observation: Here are some candidate knowledge triplets you can choose from: " + str(
                        new_node.next_triple_list)
                    unique_states[unique_key] = new_node
                    logging.info(f"NEW NODE: {new_node}")
                else:
                    new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
                    original_observation = new_state['observation']
                    new_state[
                        'observation'] = f'Observation: Invalid action! You chose a triplet that does not match any of the candidate knowledge triplets. Please remember to choosing an exact and complete triplet from: ' + original_observation[
                                                                                                                                                                                                                             original_observation.find(
                                                                                                                                                                                                                                 '['):] if '[' in original_observation else original_observation
                    new_node = Node(state=new_state, question=node.question, parent=node,
                                    topic_entity=node.topic_entity, true_answer=node.true_answer)
                    new_node.depth = node.depth + 1
                    unique_states[unique_key] = new_node
                    logging.info(f"NEW NODE: {new_node}")
        idx += 1

    return list(unique_states.values())


def get_raw_observation(text):
    """
    Retrieves the raw observation from the text.
    """
    keyword = '\n\nBelow are the previous Thought and Action you generated along with their corresponding Observation:'
    index = text.find(keyword)
    if index != -1:
        return text[:index]
    else:
        return text


def get_historical_context(context):
    """
    Retrieves the historical context from the context.
    """
    prompt = ''
    for message in context.messages[25:]:
        if message[1] is not None:
            prompt += message[1].strip() + '\n'
    return prompt


def generate_new_states_critique_fastchat_conv(node, args, task, n, critique_prompt_template):
    """
    Generates new states using critique FastChat Conv.
    """
    global failed_trajectories
    unique_states = {}

    next_entity_relations_list, next_entity_list, next_chain_list = find_next_triples(n, node, args)
    i = len(next_entity_relations_list)

    if i == 0:
        results = generate_without_explored_paths(node.question, [], args, '')
        action_type = "Finish[]"
        action_param = clean_results(results)
        new_state = node.state.copy()

        if action_type.startswith("Finish[") and action_type.endswith("]"):
            env.env.env.node = node
            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
            new_state['action'] = f"Thought: {action_param} Action: {action_type}"
            new_state['observation'] = f"Observation: {results}"
            unique_key = f"{action_param}::{action_type}::none"
            new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,
                            true_answer=node.true_answer)
            new_node.is_terminal = True
            if exact_match(action_param, node.true_answer):
                new_node.reward = r
            else:
                new_node.reward = -1
            new_node.depth = node.depth + 1
            if r == 1:
                new_node.em = info.get('em')
            unique_states[unique_key] = new_node
            logging.info(f"NEW NODE: {new_node}")
            logging.info(f"Feedback: {info}")
            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory_from_bottom(new_node)
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
            if not args.critique_backend:
                args.critique_backend = args.backend
            if not args.critique_conv_template:
                args.critique_conv_template = args.conv_template
            critique_context = copy.deepcopy(get_context(node, args.critique_conv_template, args.critique_backend))

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
        new_state = node.state.copy()
        unique_key = f"{action_line}"
        if unique_key in unique_states:
            continue

        tried_actions.append(action_line)
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = '[' + action_line.split('[')[1].split(']')[0] + ']' if '[' in action_line else ""
            if action_type.startswith("Finish"):
                unique_key = f"{action_line}"
                if unique_key in unique_states:
                    continue

                tried_actions.append(action_line)
                env.env.env.node = node
                obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
                new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
                new_state['observation'] = f"Answer: {obs}"
                new_state['critique'] = critique
                new_state['regenerate_prompt'] = regenerate_prompt
                previous_obs = f"Answer: {obs}"
                new_node = Node(state=new_state, question=node.question, parent=node, topic_entity=node.topic_entity,
                                true_answer=node.true_answer)
                new_node.is_terminal = True
                if exact_match(action_param, node.true_answer):
                    new_node.reward = r
                else:
                    new_node.reward = -1
                new_node.depth = node.depth + 1
                if r == 1:
                    new_node.em = info.get('em')
                unique_states[unique_key] = new_node
                logging.info(f"NEW NODE: {new_node}")
                logging.info(f"Feedback: {info}")
                if new_node.is_terminal and r == 0:
                    trajectory = collect_trajectory_from_bottom(new_node)
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
                    new_state['action'] = f"Thought: {thought_line} Action: Choose{action_param}"
                    new_state['critique'] = critique
                    new_state['regenerate_prompt'] = regenerate_prompt
                    new_node = Node(state=new_state, question=node.question, parent=node,
                                    topic_entity=node.topic_entity, true_answer=node.true_answer)
                    new_node.triple = str(next_chain)
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
                    new_state['action'] = f"Thought: {thought_line} Action: {action_line}"
                    original_observation = new_state['observation']
                    new_state[
                        'observation'] = f'Observation: Invalid action! You chose a triplet that does not match any of the candidate knowledge triplets. Please remember to choosing an exact and complete triplet from: ' + original_observation[
                                                                                                                                                                                                                             original_observation.find(
                                                                                                                                                                                                                                 '['):] if '[' in original_observation else original_observation
                    previous_obs = f'Observation: Invalid action! You chose a triplet that does not match any of the candidate knowledge triplets. Please remember to choosing an exact and complete triplet from: ' + original_observation[
                                                                                                                                                                                                                       original_observation.find(
                                                                                                                                                                                                                           '['):] if '[' in original_observation else original_observation
                    new_state['critique'] = critique
                    new_state['regenerate_prompt'] = regenerate_prompt
                    new_node = Node(state=new_state, question=node.question, parent=node,
                                    topic_entity=node.topic_entity, true_answer=node.true_answer)
                    new_node.depth = node.depth + 1
                    unique_states[unique_key] = new_node
                    logging.info(f"NEW NODE: {new_node}")
        idx += 1

    return list(unique_states.values())


def generate_new_states_conditional_fastchat_conv(node, args, task, n):
    """
    Generates new states using conditional FastChat Conv.
    """
    node = node


def evaluate_node(node, args, task):
    """
    Evaluates a node by generating values for its children.
    """
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample, args)
    logging.info(f"Length of votes: {len(votes)}")
    logging.info(f"Length of node.children: {len(node.children)}")

    votes = votes + [0] * (len(node.children) - len(votes))

    for i, child in enumerate(node.children):
        child.value = votes[i]

    return sum(votes) / len(votes) if votes else 0


def print_tree(node, level=0):
    """
    Prints the tree structure of nodes.
    """
    indent = "  " * level
    print(f"{indent}{node}")
    for child in node.children:
        print_tree(child, level + 1)


def backpropagate(node, value):
    """
    Backpropagates the value through the tree.
    """
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
    """
    Generates a prompt for the node.
    """
    trajectory = []
    question = node.question

    while node:
        new_segment = []
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")

        trajectory.append('\n'.join(new_segment))
        node = node.parent

    return question + '\n'.join(reversed(trajectory))