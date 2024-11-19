import os
import json
import glob

with open('hotpot/data_split/train_indices.json', 'r', encoding='utf-8') as file:
    # 加载JSON文件内容
    dataset_idx_list = json.load(file)


trajectories_save_path = 'hotpot/trajectories_train_llama31-'

done_task_id = []

best_reward = []
best_em = []

best_child_reward = []
best_child_em = []

file_idx_list = []

# 匹配所有以 trajectories_save_path 开头的目录，并遍历其中的文件
for dir_path in glob.glob(f"{trajectories_save_path}*"):
    for file in os.listdir(dir_path):
        file_idx_list.append(int(file.split('.')[0]))
        if int(file.split('.')[0]) not in dataset_idx_list:
            raise ValueError
        if not file.endswith('json'):
            continue
        with open(os.path.join(dir_path, file)) as f:
            result = json.load(f)
        best_reward.append(result['best reward'])
        best_em.append(result['best em'] if result['best em']  is not None else 0)
        best_child_reward.append(result['best child reward'])
        best_child_em.append(result['best child em'] if result['best child em']  is not None else 0)
    



print("Sample number: ", len(best_child_reward))
print("average best reward: ", sum(best_reward)/len(best_reward))
print("average best em: ", sum(best_em)/len(best_reward))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
print("average best child em: ", sum(best_child_em)/len(best_child_em))
