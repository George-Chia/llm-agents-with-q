import os
import json

'''
交互前后对比
trajectories_save_path = 'hotpot/trajectories-MCTS-gpt4o_critique_test_llama31_mcts_20iterations'

trajectories_save_path_comparision = 'hotpot/trajectories-3n/trajectories-MCTS-gpt4o_critique_test_llama31_mcts_30iterations'
'''

trajectories_save_path = 'hotpot/trajectories-MCTS-critique-template_huan_test_llama31_mcts_30iterations'
trajectories_save_path_comparision = 'hotpot/trajectories-5n-critique-wrong-InTurn/trajectories-MCTS-critique_test_llama31_mcts_30iterations'


best_reward = []
best_child_reward = []

best_reward_comparision = []
best_child_reward_comparision = []
success_length_list = []
success_length_list_comparision = []


for file in os.listdir(trajectories_save_path):
    if not file.endswith('json'):
        continue
    if file not in os.listdir(trajectories_save_path_comparision):
        continue
    
    with open(os.path.join(trajectories_save_path, file)) as f:
        result=json.load(f)

    with open(os.path.join(trajectories_save_path_comparision, file)) as f:
        result_comparision=json.load(f)

    best_reward.append(result['best reward'])
    best_child_reward.append(result['best child reward'])

    best_reward_comparision.append(result_comparision['best reward'])
    best_child_reward_comparision.append(result_comparision['best child reward'])

    if result['best child reward'] > 0:
        success_length_list.append(len(result['best_trajectory_index_list']))
    if result_comparision['best child reward'] > 0:
        success_length_list_comparision.append(len(result_comparision['best_trajectory_index_list']))

print("Sample number: ", len(best_child_reward))
print("average best reward: ", sum(best_reward)/len(best_reward))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
print("average success length: ", sum(success_length_list)/len(success_length_list))


print('--------------------------------------------------------')

print("average best reward_comparision: ", sum(best_reward_comparision)/len(best_reward))
print("average best child reward_comparision: ", sum(best_child_reward_comparision)/len(best_child_reward))
print("average success length_comparision: ", sum(success_length_list_comparision)/len(success_length_list_comparision))

