import os
import json


trajectories_save_path = 'webshop/trajectories_test_llama31-T0_simple_1iterations'



done_task_id = []

best_reward = []
best_child_reward = []
for file in os.listdir(trajectories_save_path):
    if not file.endswith('json'):
        continue
    with open(os.path.join(trajectories_save_path, file)) as f:
        result=json.load(f)
    # best_reward.append(result['best reward'])
    # best_child_reward.append(result['best child reward'])
    best_child_reward.append(0 if result['best child reward']==-1 else result['best child reward'])
print("Sample number: ", len(best_child_reward))
# print("average best reward: ", sum(best_reward)/len(best_reward))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
