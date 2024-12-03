import os
import json
import copy
import traceback
import webshop_prompt
import hotpot_prompt
from collections import deque
from collections import Counter
from transformers import AutoTokenizer

preferred_value = []
less_preferred_value = []
difference_value = []

tokenizer = AutoTokenizer.from_pretrained('/home/data/huan/llm/Llama3.1-8B-Instruct')

def get_max_min_children_index(node: list) -> tuple:
    value_list = []
    for child in node['children']:
        value_list.append(child['value'])
    return value_list.index(max(value_list)), value_list.index(min(value_list))


def get_pointwise_data(trajectories_save_path, filter_num, depth_list, threshold=0, enable_bfs=False, test=False):
    pointwise_data = []
    sample_num = 0
    for file in os.listdir(trajectories_save_path):
        if sample_num==1000:
            break
        sample_num += 1
        if not file.endswith('json'):
            continue
        try:
            with open(os.path.join(trajectories_save_path, file)) as f:
                root = json.load(f)
        except:
            print(file)
        # whole_trajectory_index = get_whole_trajectory_index(root)
        prompt = [{'role': 'user', 'content': root['messages'][0]['content']}]
        prompt.append({'role': 'assistant', 'content': root['messages'][1]['content']})
        prompt.append({'role': 'user', 'content': root['messages'][-1]['content']})

        if enable_bfs:
            all_node = []
            queue = deque([(root, prompt, 0, 0, None)])  # 队列中存储节点、当前路径的累加message和当前分支的节点列表
            while queue:
                current_node, accumulated_message, current_index, dep, _ = queue.popleft()
                new_accumulated_message = copy.deepcopy(accumulated_message)
                if current_node != root:
                    new_accumulated_message.append({'role': 'assistant', 'content': current_node['state']['action']})
                    new_accumulated_message.append({'role': 'user', 'content': current_node['state']['observation']})

                if current_node['children'] is not None:
                    # 将当前节点的所有子节点入队，并记录它们的累加message和新的分支节点列表
                    for index, child in enumerate(current_node['children']):
                        queue.append((child, new_accumulated_message, index, dep+1, current_node))
                        all_node.append((child, new_accumulated_message, index, dep+1, current_node, current_index))
            try:
                for node1 in all_node:
                    for node2 in all_node:
                        if node1[4]==node2[4] and node1[1]==node2[1] and node1[5]==node2[5] and node1[3]==node2[3] and node1[2]==node2[2]-1:
                            critique_prompt = copy.deepcopy(node1[1])
                            original_observation = critique_prompt[-1]['content']
                            if 'hotpot' in trajectories_save_path:
                                critique_prompt[-1]['content'] = original_observation + node2[0]['state']['regenerate_prompt'].split('Critique:')[0] + hotpot_prompt.template_v1.split('{previous_obs}')[-1][1:]
                            elif 'webshop' in trajectories_save_path:
                                critique_prompt[-1]['content'] = original_observation + node2[0]['state']['regenerate_prompt'].split('Critique:')[0] + hotpot_prompt.template_v1.split('{previous_obs}')[-1][1:]
                            else:
                                exit(666)
                            critique = [{'role': 'assistant', 'content': node2[0]['state']['critique']}]

                            if node2[0]['value'] > node1[0]['value'] + threshold:
                                is_effective = 1
                                # print("critique is effective")
                            elif node2[0]['value'] + threshold < node1[0]['value']:
                                is_effective = 0
                                # print("critique is ineffective")
                            else:
                                continue
                            if not test and len(tokenizer.encode(str(critique_prompt)+str(critique))) > 2048:
                                filter_num += 1
                                break
                            pointwise_data.append({
                                "critique_prompt": critique_prompt,
                                'critique': critique,
                                "is_effective": is_effective,
                                "id": f"{file}-depth{node2[3]}-father_index{node2[5]}-child_index{node2[2]}"
                            })
                            depth_list.append(node2[3])
            except Exception as e:
                # pass
                traceback.print_exc()
        else:
            node = root
            best_trajectory_index_list = root['best_trajectory_index_list']  # children index
            try:
                for depth, best_trajectory_index in enumerate(best_trajectory_index_list):  # preference for each depth
                    while True:
                        if node['depth'] == depth:
                            if node['depth'] != 0:
                                prompt.append({'role': 'assistant', 'content': node['state']['action']})
                                prompt.append({'role': 'user', 'content': node['state']['observation']})
                            if node['children'] is None:
                                break
                            for (child_index, child) in enumerate(node['children']):
                                is_effective = 0
                                if child_index == 0:
                                    continue
                                critique_prompt = copy.deepcopy(prompt)
                                original_observation = critique_prompt[-1]['content']
                                if 'hotpot' in trajectories_save_path:
                                    critique_prompt[-1]['content'] = original_observation + node['children'][child_index]['state']['regenerate_prompt'].split('Critique:')[0] + hotpot_prompt.template_v1.split('{previous_obs}')[-1][1:]
                                elif 'webshop' in trajectories_save_path:
                                    critique_prompt[-1]['content'] = original_observation + node['children'][child_index]['state']['regenerate_prompt'].split('Critique:')[0] + hotpot_prompt.template_v1.split('{previous_obs}')[-1][1:]
                                else:
                                    exit(666)
                                critique = [{'role': 'assistant', 'content': node['children'][child_index]['state']['critique']}]

                                if node['children'][child_index]['value'] > node['children'][child_index - 1]['value'] + threshold:
                                    is_effective = 1
                                    # print("critique is effective")
                                elif node['children'][child_index]['value'] + threshold < node['children'][child_index - 1]['value']:
                                    is_effective = 0
                                else:
                                    continue
                                    # print("critique is ineffective")
                                if not test and len(tokenizer.encode(str(critique_prompt)+str(critique))) > 2048-512:
                                    filter_num += 1
                                    break
                                pointwise_data.append({
                                    "critique_prompt": critique_prompt,
                                    'critique': critique,
                                    "is_effective": is_effective,
                                    "id": f"{file}-depth{depth}-child_index{child_index}"
                                })
                                depth_list.append(depth)
                            break
                        node = node['children'][best_trajectory_index]
            except Exception as e:
                pass
                # traceback.print_exc()
    print('filter number: ', filter_num)
    return pointwise_data

# dataset = 'hotpot'
dataset = 'webshop'
bfs_flag = True
q_threshold = 0.

depth_list = []
trajectories_save_path = f"data/{dataset}/raw/trajectories-MCTS-critique-disable_early_stop_train_llama31_mcts_20iterations"
preference_train_data = get_pointwise_data(trajectories_save_path, 0, depth_list, q_threshold, enable_bfs=bfs_flag, test=False)
print('total training samples: ', len(preference_train_data))
count = Counter(depth_list)
proportions = {item: count[item] / len(depth_list) for item in count}
print('depth proportion: ', proportions)
print('avg depth: ', sum(depth_list)/len(depth_list))

json.dump(preference_train_data, open(f"data/{dataset}/post/train.json", "w"), indent=4)

depth_list = []
trajectories_save_path = f"data/{dataset}/raw/trajectories-MCTS-critique-disable_early_stop_test_llama31_mcts_20iterations"
preference_test_data = get_pointwise_data(trajectories_save_path, 0, depth_list, q_threshold, bfs_flag, test=True)
json.dump(preference_test_data, open(f"data/{dataset}/post/test.json", "w"), indent=4)

# import random
# one_tenth_size = len(preference_data) // 10
#
# # 随机抽取十分之一的元素
# sampled_elements = random.sample(preference_data, one_tenth_size)
#
# # 从原始列表中删除这些随机抽取的元素
# remaining_elements = [element for element in preference_data if element not in sampled_elements]



