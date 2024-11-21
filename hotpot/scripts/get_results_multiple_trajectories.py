import os
import json
import glob

# with open('hotpot/data_split/train_indices.json', 'r', encoding='utf-8') as file:
#     # 加载JSON文件内容
#     dataset_idx_list = json.load(file)


# trajectories_save_path = 'hotpot/trajectories-MCTS_test_llama31-'
# Sample number:  100
# average success length:  2.6451612903225805
# average best reward:  0.71
# average best em:  0.71
# average best child reward:  0.62
# average best child em:  0.62

trajectories_save_path = 'hotpot/trajectories-MCTS-critique_test_llama31-'
# Sample number:  100 (without direct feedback)
# average success length:  2.4603174603174605
# average best reward:  0.66
# average best em:  0.66
# average best child reward:  0.63
# average best child em:  0.63

# Sample number:  100 (direct feedback)
# average success length:  2.425925925925926
# average best reward:  0.64
# average best em:  0.64
# average best child reward:  0.54
# average best child em:  0.54


# trajectories_save_path = 'hotpot/trajectories-MCTS-gpt4o_critique_test_llama31-' (without direct feedback)
# Sample number:  100
# average success length:  2.242857142857143
# average best reward:  0.75
# average best em:  0.75
# average best child reward:  0.7
# average best child em:  0.7


done_task_id = []

best_reward = []
best_em = []

best_child_reward = []
best_child_em = []

file_idx_list = []

success_length_list = []

# 匹配所有以 trajectories_save_path 开头的目录，并遍历其中的文件
for dir_path in glob.glob(f"{trajectories_save_path}*"):
    for file in os.listdir(dir_path):
        file_idx_list.append(int(file.split('.')[0]))
        # if int(file.split('.')[0]) not in dataset_idx_list:
        #     raise ValueError
        if not file.endswith('json'):
            continue
        with open(os.path.join(dir_path, file)) as f:
            result = json.load(f)
        if result['best child reward'] == 1:
            success_length_list.append(len(result['best_trajectory_index_list']))
        best_reward.append(result['best reward'])
        best_em.append(result['best em'] if result['best em']  is not None else 0)
        best_child_reward.append(result['best child reward'])
        best_child_em.append(result['best child em'] if result['best child em']  is not None else 0)


print("Sample number: ", len(best_child_reward))
print("average success length: ", sum(success_length_list)/len(success_length_list))
print("average best reward: ", sum(best_reward)/len(best_reward))
print("average best em: ", sum(best_em)/len(best_reward))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
print("average best child em: ", sum(best_child_em)/len(best_child_em))
