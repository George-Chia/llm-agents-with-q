import os
import json

# Sample number:  1000
# average best reward:  0.602
# average best em:  0.188
# average best child reward:  0.602
# average best child em:  0.188
# version1: rollout with value
# trajectories_save_path = "hotpot/trajectories/trajectories_4omini/trajectories_train_gpt-4o-mini_mcts_30iterations_1000samples_60.2"


'''
abalation of iterations
# Sample number:  1000 -> 1454
# average best reward:  0.65
# average best em:  0.402
# average best child reward:  0.647
# average best child em:  0.399

# trajectories_save_path = "hotpot/trajectories_train_Phi-3_mcts_10iterations"
# 10iterations Sample number:  1000 ->1111
# average best reward:  0.546
# average best em:  0.287
# average best child reward:  0.53
# average best child em:  0.271


# trajectories_save_path = "hotpot/trajectories_train_Phi-3_mcts_50iterations"
# 50iterations Sample number:  1000 -> 1622
# average best reward:  0.701
# average best em:  0.424
# average best child reward:  0.7
# average best child em:  0.423
'''

# version2: rollout with UCT
# trajectories_save_path = "hotpot/trajectories-reflection_train_gpt-4o-mini_mcts_30iterations"


# Sample number:  1000
# average best reward:  0.672
# average best em:  0.213
# average best child reward:  0.671
# average best child em:  0.212
# trajectories_save_path = "hotpot/trajectories_train_gpt-4o_mcts_30iterations_1000samples"

trajectories_save_path = "hotpot/trajectories_backup_NO_Observation/trajectories-MCTS_test_llama31-2_mcts_30iterations"
# Sample number:  55
# average best reward:  0.6909090909090909
# average best em:  0.6909090909090909
# average best child reward:  0.6545454545454545
# average best child em:  0.6545454545454545

# trajectories_save_path = "hotpot/trajectories_backup_NO_Observation/trajectories-MCTS-critique_test_llama31-3_mcts_30iterations"
# Sample number:  8
# average best reward:  0.75
# average best em:  0.75
# average best child reward:  0.75
# average best child em:  0.75

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
    best_reward.append(result['best reward'])
    best_em.append(result['best em'] if result['best em']  is not None else 0)
    best_child_reward.append(result['best child reward'])
    best_child_em.append(result['best child em'] if result['best child em']  is not None else 0)
print("Sample number: ", len(best_child_reward))
print("average best reward: ", sum(best_reward)/len(best_reward))
print("average best em: ", sum(best_em)/len(best_reward))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
print("average best child em: ", sum(best_child_em)/len(best_child_em))