import os
import json


# phi-3-iteration0
# trajectories_save_path = "webshop/trajectories_forbid-double-search/trajectories_test_phi3_1iterations" # average best child reward:  0.10441666666666666
trajectories_save_path = "webshop/trajectories_fixed/trajectories_forbid-double-search/trajectories_train_phi3_mcts_30iterations" # average best child reward:  0.42721501272264745

# phi-3-iteration0-test-with-q-model
# trajectories_save_path = "webshop/trajectories_forbid-double-search/trajectories_epoch1_test_phi3_beam_1iterations" # average best child reward:  0.14464166666666667
# trajectories_save_path = "webshop/trajectories_forbid-double-search/trajectories_epoch2_test_phi3_beam_1iterations" # average best child reward:  0.14475
# trajectories_save_path = "webshop/trajectories_forbid-double-search/trajectories_epoch3_test_phi3_beam_1iterations" # average best child reward:  0.10485
# trajectories_save_path = "webshop/trajectories_forbid-double-search/trajectories_ETO_test_phi3_beam_1iterations" # average best child reward:  0.12743333333333334

# phi-3-iteration1
# trajectories_save_path = "webshop/trajectories-round2_train_Phi-3_mcts_30iterations_puct1e5"
# trajectories_save_path = "webshop/trajectories-round2_train_Phi-3_mcts_30iterations"

# gpt-4o-mini-iteration0
# trajectories_save_path = "webshop/trajectories_forbid-double-search/trajectories_test_gpt-4o-mini_simple_1iterations" # average best child reward:  0.13184079601990048
trajectories_save_path = "webshop/trajectories-Phi3/trajectories_train_phi3_mcts_30iterations"
# trajectories_save_path = 'webshop/trajectories_train_Phi-3_mcts_10iterations'


# gpt-4-turbo-iteration0
# trajectories_save_path = "webshop/trajectories_forbid-double-search/trajectories_test_gpt-4-turbo_simple_1iterations" # average best child reward:  0.11035897435897435



# phi-3-iteration0_allow-double-search
# trajectories_save_path = "webshop/trajectories_test_Phi-3-3_simple_1iterations" # average best child reward:  0.10441666666666666


trajectories_save_path = 'webshop/trajectories_backup/trajectories_train_llama31-0_mcts_30iterations'

done_task_id = []

best_reward = []
best_child_reward = []
for file in os.listdir(trajectories_save_path):
    if not file.endswith('json'):
        continue
    with open(os.path.join(trajectories_save_path, file)) as f:
        result=json.load(f)
    best_reward.append(result['best reward'])
    best_child_reward.append(result['best child reward'])
    # best_child_reward.append(0 if result['best child reward']==-1 else result['best child reward'])
print("Sample number: ", len(best_child_reward))
print("average best reward: ", sum(best_reward)/len(best_reward))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
