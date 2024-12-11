import os
import json


with open('webshop/data_split/train_indices.json', 'r', encoding='utf-8') as file:
    # 加载JSON文件内容
    dataset_idx_list = json.load(file)

# trajectories_save_path = 'webshop/trajectories-MCTS_test_llama31_T1.0_mcts_30iterations'

'''
trajectories_save_path = 'webshop/trajectories-MCTS-KTO-critique_test_llama31_T1.0_mcts_20iterations'
trajectories_save_path_comparision = 'webshop/trajectories-MCTS_test_llama31_T1.0_mcts_20iterations'
打败了MCTS
'''

'''
trajectories_save_path = 'webshop/trajectories-MCTS-KTO-critique-rwc_test_llama31_T1.0_mcts_20iterations'
trajectories_save_path_comparision = 'webshop/trajectories-MCTS-KTO-critique_test_llama31_T1.0_mcts_20iterations'
webshop中，加rwc还不如不加
'''

trajectories_save_path = 'webshop/trajectories-MCTS-KTO-critique_test_llama31_T1.0_mcts_20iterations'
trajectories_save_path = 'webshop/trajectories-MCTS-KTO-8000-critique_test_llama31_T1.0_mcts_20iterations'
trajectories_save_path = 'webshop/trajectories-MCTS-critique-template_huan_invalid_test_llama31_T1.0_mcts_30iterations'
# trajectories_save_path = 'webshop/trajectories-MCTS-16000-critique_test_llama31_T1.0_mcts_20iterations'
# trajectories_save_path = 'webshop/trajectories-MCTS-SFT-critique_test_llama31_T1.0_mcts_20iterations'

# trajectories_save_path = 'webshop/trajectories-MCTS-KTO-critique-rwc_test_llama31_T1.0_mcts_20iterations'
# trajectories_save_path_comparision = 'webshop/trajectories-MCTS-KTO-critique_test_llama31_T1.0_mcts_20iterations_from_wrong_Inturn'
trajectories_save_path_comparision = 'webshop/trajectories-MCTS-critique_test_llama31_T1.0_mcts_30iterations'
# trajectories_save_path_comparision = 'webshop/trajectories-MCTS_test_llama31_T1.0_mcts_20iterations'
# trajectories_save_path_comparision = 'webshop/trajectories-MCTS-3n-gpt4o_critique_test_llama31_T1.0_mcts_20iterations'


# trajectories_save_path_comparision = 'webshop/trajectories-MCTS-KTO-critique_test_llama31_T1.0_mcts_20iterations'

best_reward_comparision = []
best_child_reward_comparision = []
success_length_list_comparision = []
success_failure_list_comparision = []


best_reward = []
best_child_reward = []
success_length_list = []
success_failure_list = []

for file in os.listdir(trajectories_save_path):
    if not file.endswith('json'):
        continue
    if not file in os.listdir(trajectories_save_path_comparision):
        continue
    with open(os.path.join(trajectories_save_path, file)) as f:
        result=json.load(f)
    if result['best child reward'] > 0:
        success_length_list.append(len(result['best_trajectory_index_list']))
    best_reward.append(result['best reward'])
    # best_child_reward.append(result['best child reward'])
    best_child_reward.append(0 if result['best child reward']==-1 else result['best child reward'])
    success_failure_list.append(1 if result['best reward']==1 else 0)

    with open(os.path.join(trajectories_save_path_comparision, file)) as f:
        result_comparision=json.load(f)
    if result_comparision['best child reward'] > 0:
        success_length_list_comparision.append(len(result_comparision['best_trajectory_index_list']))
    best_reward_comparision.append(result_comparision['best reward'])
    # best_child_reward.append(result['best child reward'])
    best_child_reward_comparision.append(0 if result_comparision['best child reward']==-1 else result_comparision['best child reward'])
    success_failure_list_comparision.append(1 if result_comparision['best reward']==1 else 0)

print("Sample number: ", len(best_child_reward))

print('--------------------------------------------------------')

print("average best reward: ", sum(best_reward)/len(best_reward))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
print("average success length: ", sum(success_length_list)/len(success_length_list), "numbers: ", len(success_length_list))
print("average success rate: ", sum(success_failure_list)/len(success_length_list))

print('--------------------------------------------------------')


print("average best reward_comparision: ", sum(best_reward_comparision)/len(best_reward_comparision))
print("average best child reward_comparision: ", sum(best_child_reward_comparision)/len(best_child_reward_comparision))
print("average success length_comparision: ", sum(success_length_list_comparision)/len(success_length_list_comparision), "numbers: ", len(success_length_list_comparision))
print("average success rate_comparision: ", sum(success_failure_list_comparision)/len(success_length_list))