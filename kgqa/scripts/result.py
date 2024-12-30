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



# trajectories_save_path = "hotpot/trajectories-MCTS_test_llama31_mcts_30iterations"
# Sample number:  100
# average best reward:  0.71
# average best em:  0.71
# average best child reward:  0.62
# average best child em:  0.62

# trajectories_save_path = 'hotpot/trajectories-MCTS-critique_test_llama31_no-backpro'
# Sample number:  100
# average best reward:  0.64
# average best em:  0.64
# average best child reward:  0.54
# average best child em:  0.54

trajectories_save_path = 'hotpot/trajectories-MCTS-KTO_critique_test_llama31_mcts_30iterations'
trajectories_save_path = '../trajectories_test_gpt_mcts_5iterations'

# trajectories_save_path = "hotpot/trajectories-MCTS-critique_train_llama31_mcts_30iterations"
# Sample number:  950
# average best reward:  0.6936842105263158
# average best em:  0.6936842105263158
# average best child reward:  0.6157894736842106
# average best child em:  0.6157894736842106

# trajectories_save_path = "hotpot/trajectories-MCTS-critique-disable_early_stop_train_llama31_mcts_20iterations"
# Sample number:  1000
# average best reward:  0.66
# average best em:  0.66
# average best child reward:  0.558
# average best child em:  0.558

done_task_id = []

best_reward = []
best_em = []

best_child_reward = []
best_child_em = []

success_length_list = []

correct_num = 0
total_num =0
for file in os.listdir(trajectories_save_path):
    total_num +=1
    if not file.endswith('json'):
        continue
    with open(os.path.join(trajectories_save_path, file)) as f:
        result=json.load(f)
    if result['best child reward'] == 1:
        correct_num +=1
print("Sample number: ", total_num)
print("correct_num: ", correct_num)
print("正确率: ", correct_num/total_num)
