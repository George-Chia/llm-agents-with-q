import os
import json

trajectories_save_path = 'hotpot/trajectories_800samples_test_Phi-3_beam_1iterations'


done_task_id = []

best_reward = []
best_em = []

best_child_reward = []
best_child_em = []
for file in os.listdir(trajectories_save_path):
    if not file.endswith('json'):
        continue
    with open(os.path.join(trajectories_save_path, file)) as f:
        result=json.load(f)
    # best_reward.append(result['best reward'])
    # best_em.append(result['best em'] if result['best em']  is not None else 0)
    best_child_reward.append(result['best child reward'])
    best_child_em.append(result['best child em'] if result['best child em']  is not None else 0)
print("Sample number: ", len(best_child_reward))
# print("average best reward: ", sum(best_reward)/len(best_reward))
# print("average best em: ", sum(best_em)/len(best_reward))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
print("average best child em: ", sum(best_child_em)/len(best_child_em))