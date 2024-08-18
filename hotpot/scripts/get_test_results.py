import os
import json

# Phi3
# Sample number:  100
# average best child reward:  0.20
# average best child em:  0.20
# trajectories_save_path = "hotpot/trajectories_Phi3_1T/trajectories_test_Phi-3_simple_1iterations"

# Phi3 with Q (T05)
# average best child reward:  0.35
# average best child em:  0.26
# trajectories_save_path = 'hotpot/trajectories_Phi3_05T/trajectories_05T_test_Phi-3_beam_1iterations_035'


# GPT-4o-mini
# average best child reward:  0.31
# average best child em:  0.31
# trajectories_save_path = "hotpot/trajectories/trajectories_4omini/trajectories_test_gpt-4o-mini_simple_1iterations_31"

# GPT-4o-mini with Q
# average best child reward:  0.44
# average best child em:  0.37
# trajectories_save_path = "hotpot/trajectories/trajectories_4omini/trajectories_Phi3Qepoch2_test_gpt-4o-mini_beam_1iterations_44"

# GPT-4-turbo
# average best child reward:  0.44
# average best child em:  0.44
# trajectories_save_path = "hotpot/trajectories/trajectories_4turbo/trajectories_test_gpt-4-turbo_simple_1iterations"

# GPT-4-turbo with Q
# average best child reward:  0.5
# average best child em:  0.48
# trajectories_save_path = "hotpot/trajectories/trajectories_4turbo/trajectories_Phi3Qepoch2_test_gpt-4-turbo-2024-04-09_beam_1iterations"

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