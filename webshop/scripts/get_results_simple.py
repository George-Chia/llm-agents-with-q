import os
import json
import copy
import traceback
from collections import deque

template_v1=""""

Below are the previous Thought and Action you generated along with their corresponding Observation: 

{previous_response}
{previous_obs}

Review the previous Thought, Action, and Observation. Your role is to determine whether the action is effective for completing the task, and provide specific and constructive feedback. Please output feedback directly. 
Format
Feedback:[[Feedback]]"""


def get_high_value_traj(trajectories_save_path):
    pointwise_data = []
    for file in os.listdir(trajectories_save_path):
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
                    queue.append((child, new_accumulated_message, index, dep+1, current_node, ))
                    all_node.append((child, new_accumulated_message, index, dep+1, current_node, current_index, [child['state']['action'], child['state']['observation']]))
        try:
            # for node1 in all_node:
                for node2 in all_node:
                    # if node1[4]==node2[4] and node1[1]==node2[1] and node1[5]==node2[5] and node1[3]==node2[3] and node1[2]==node2[2]-1:
                        critique_prompt = copy.deepcopy(node2[1])
                        original_observation = critique_prompt[-1]['content']
                        # if 'hotpot' in trajectories_save_path:
                        #     critique_prompt[-1]['content'] = original_observation + node2[0]['state']['regenerate_prompt'].split('Critique:')[0] + template_v1.split('{previous_obs}')[-1][1:]
                        # elif 'webshop' in trajectories_save_path:
                        #     critique_prompt[-1]['content'] = original_observation + node2[0]['state']['regenerate_prompt'].split('Critique:')[0] + template_v1.split('{previous_obs}')[-1][1:]
                        # else:
                        #     exit(666)
                        # critique = [{'role': 'assistant', 'content': node2[0]['state']['critique']}]

                        if 'invalid action' in node2[-1][1].lower() and 'action: click' in node2[-1][0].lower():
                            is_effective = 1
                        else:
                            is_effective = 0
                        # if node2[0]['value'] > node1[0]['value'] and node2[0]['value'] > 0.8:
                        #     is_effective = 1
                        # else:
                        #     is_effective = 0
                        if is_effective == 0:
                            continue
                        pointwise_data.append({
                            # "critique_prompt": critique_prompt,
                            # 'critique': critique,
                            "new_action": node2[-1][0],
                            "new_observation": node2[-1][1],
                            # "is_effective": is_effective,
                            "id": f"{file}-depth{node2[3]}-father_index{node2[5]}-child_index{node2[2]}"
                        })
        except Exception as e:
            # pass
            traceback.print_exc()
    return pointwise_data

trajectories_save_path = 'webshop/webshop_trajectories-MCTS-3n-gpt4o_critique_test_llama31_T1.0_mcts_20iterations'
my_data = get_high_value_traj(trajectories_save_path)
# for item in my_data:
#     if len(item['critique_prompt']) <= 4:
#         print(1)
#     if 'previous action was marked as invalid' in item['critique'][0]:
#         a = 1
#         print(item['new_action'])
print(len(my_data))

# done_task_id = []
# best_reward = []
# best_child_reward = []
# for file in os.listdir(trajectories_save_path):
#     if not file.endswith('json'):
#         continue
#     with open(os.path.join(trajectories_save_path, file)) as f:
#         result=json.load(f)
#     # best_reward.append(result['best reward'])
#     # best_child_reward.append(result['best child reward'])
#     best_child_reward.append(0 if result['best child reward']==-1 else result['best child reward'])
# print("Sample number: ", len(best_child_reward))
# # print("average best reward: ", sum(best_reward)/len(best_reward))
# print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
