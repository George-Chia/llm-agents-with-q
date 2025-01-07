import copy
import os
import json

trajectories_save_path = 'webshop/trajectories_test_gpt-turbo_T1.0_simple_1iterations'

def get_trajectory_length(root):
    node = copy.deepcopy(root)
    length = 1
    while node['children'] and len(node['children']) != 0:
        node = node['children'][0]
        length += 1
    return length

best_child_reward = []
success_length_list = []
solved_num = 0
for file in os.listdir(trajectories_save_path):
    file_id = int(file.split('.json')[0])
    if not file.endswith('json'):
        continue
    try:
        with open(os.path.join(trajectories_save_path, file)) as f:
            result=json.load(f)
    except:
        print(file)
    if result['best child reward'] == 1:
        # print(file)
        solved_num += 1
    if result['best child reward'] > 0:
        success_length_list.append(get_trajectory_length(result['children'][0]))
    best_child_reward.append(0 if result['best child reward']==-1 else result['best child reward'])
print("Sample number: ", len(best_child_reward))
print("average success length: ", sum(success_length_list)/len(success_length_list))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
print("solved num: ", solved_num)

#
# done_task_id = []
# best_reward = []
# best_child_reward = []
# success_length_list = []
# solved_num = 0
# for file in os.listdir(trajectories_save_path):
#     file_id = int(file.split('.json')[0])
#     if not file.endswith('json') and file_id not in dataset_idx_list:
#         continue
#     try:
#         with open(os.path.join(trajectories_save_path, file)) as f:
#             result=json.load(f)
#     except:
#         print(file)
#     if result['best child reward'] == 1:
#         # print(file)
#         solved_num += 1
#     if result['best child reward'] > 0:
#         success_length_list.append(len(result['best_trajectory_index_list']))
#
#     best_reward.append(result['best reward'])
#     # best_child_reward.append(result['best child reward'])
#     best_child_reward.append(0 if result['best child reward']==-1 else result['best child reward'])
#
# print("Sample number: ", len(best_child_reward))
# print("average success length: ", sum(success_length_list)/len(success_length_list))
# print("average best reward: ", sum(best_reward)/len(best_reward))
# print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
# print("solved num: ", solved_num)