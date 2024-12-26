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

trajectories_save_path = '/home/zhaiyuanzhao/LLM-Agents-with-Q/hotpot/trajectories_SIGIR/Iteration1/test/trajectories-Critique-MCTS-3n-policy-critique-iter1Epoch3-huan_test_llama31_mcts_30iterations'

# trajectories_save_path = ""

'''
policy+critique 3-20:
Sample number:  58
average success length:  2.4722222222222223
average best reward:  0.6896551724137931
average best em:  0.6896551724137931
average best child reward:  0.6206896551724138
average best child em:  0.6206896551724138

policy+critique 3-30:
Sample number:  100
average success length:  2.6875
average best reward:  0.69
average best em:  0.69
average best child reward:  0.64
average best child em:  0.64

policy  3-20:
Sample number:  67
average success length:  2.3181818181818183
average best reward:  0.6865671641791045
average best em:  0.6865671641791045
average best child reward:  0.6567164179104478
average best child em:  0.6567164179104478

policy 3-30:
Sample number:  71
average success length:  2.7435897435897436
average best reward:  0.6338028169014085
average best em:  0.6338028169014085
average best child reward:  0.5492957746478874
average best child em:  0.5492957746478874


'''

done_task_id = []

best_reward = []
best_em = []

best_child_reward = []
best_child_em = []

success_length_list = []


for file in os.listdir(trajectories_save_path):
    if not file.endswith('json'):
        continue
    with open(os.path.join(trajectories_save_path, file)) as f:
        result=json.load(f)
    if result['best child reward'] >0:
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