import os
import json
import glob

trajectories_save_path = 'hotpot/trajectories_train_llama31-'

done_task_id = []

best_reward = []
best_child_reward = []

# 匹配所有以 trajectories_save_path 开头的目录，并遍历其中的文件
for dir_path in glob.glob(f"{trajectories_save_path}*"):
    for file in os.listdir(dir_path):
        if not file.endswith('json'):
            continue
        with open(os.path.join(dir_path, file)) as f:
            result = json.load(f)
        best_reward.append(result['best reward'])
        best_child_reward.append(result['best child reward'])
        # best_child_reward.append(0 if result['best child reward'] == -1 else result['best child reward'])

print("Sample number: ", len(best_child_reward))
print("average best reward: ", sum(best_reward)/len(best_reward))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
