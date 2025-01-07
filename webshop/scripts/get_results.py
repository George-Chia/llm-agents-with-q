import os
import json

# phi-3-mcts-unfintuned
# trajectories_save_path = "trajectories_test_exp-sft-Phi-3-0_mcts_30iterations" #


# phi-3-iteration0
# trajectories_save_path = "webshop/trajectories_forbid-double-search/trajectories_test_phi3_1iterations" # average best child reward:  0.10441666666666666
# trajectories_save_path = "/home/zhaiyuanzhao/LanguageAgentTreeSearch/webshop/trajectories_AAAI_Submission/trajectories-Phi3/trajectories_train_phi3_lats_30iterations" # average best child reward:  0.42721501272264745

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
# trajectories_save_path = "webshop/trajectories-Phi3/trajectories_train_phi3_mcts_30iterations"
# trajectories_save_path = 'webshop/trajectories_train_Phi-3_mcts_10iterations'


# gpt-4-turbo-iteration0
# trajectories_save_path = "webshop/trajectories_forbid-double-search/trajectories_test_gpt-4-turbo_simple_1iterations" # average best child reward:  0.11035897435897435



# phi-3-iteration0_allow-double-search
# trajectories_save_path = "webshop/trajectories_test_Phi-3-3_simple_1iterations" # average best child reward:  0.10441666666666666

# trajectories_save_path = 'webshop/trajectories_test_llama31-T0_simple_1iterations'  # average best child reward:  0.4815750000000002
# trajectories_save_path = 'webshop/trajectories_test_llama31-T1_simple_1iterations'  # average best child reward:  0.42213333333333336
#
#
# trajectories_save_path = 'webshop/trajectories_test_llama31-0_T1.0_mcts_30iterations'
# trajectories_save_path = 'webshop/trajectories-n3MCTS_test_llama31-1_T1.0_mcts_30iterations'
# trajectories_save_path = 'webshop/trajectories-n3MCTS-conditional_test_llama31-2_T1.0_mcts_30iterations'


# trajectories_save_path = 'webshop/trajectories-n3MCTS_test_llama31-1_T1.0_mcts_30iterations'
# Sample number:  200
# average best reward:  0.6899583333333337
# average best child reward:  0.38587499999999997


# trajectories_save_path = 'webshop/trajectories-n3MCTS-conditional_test_llama31-2_T1.0_mcts_30iterations'
# Sample number:  200
# average best reward:  0.7077083333333338
# average best child reward:  0.4426833333333334


with open('webshop/data_split/train_indices.json', 'r', encoding='utf-8') as file:
    # 加载JSON文件内容
    dataset_idx_list = json.load(file)

trajectories_save_path = 'webshop/trajectories_iteration0/trajectories-MCTS-critique-template_v1_policy-critique_test_llama31_T1.0_mcts_30iterations'
# trajectories_save_path = 'webshop/trajectories-MCTS-critique_test_llama31_T1.0_mcts_30iterations'

# trajectories_save_path = 'webshop/trajectories-MCTS-3n-gpt4o_critique_test_llama31_T1.0_mcts_20iterations'
# trajectories_save_path = 'webshop/trajectories-MCTS-critique-disable_early_stop_test_llama31_T1.0_mcts_20iterations'

# trajectories_save_path = 'webshop/trajectories-MCTS-critique_test_llama31_T1.0_mcts_20iterations'
# Sample number:  100
# average success length:  3.336734693877551
# average best reward:  0.6799166666666663
# average best child reward:  0.6319999999999998

# trajectories_save_path = 'webshop/trajectories-MCTS-critique-disable_early_stop_test_llama31_T1.0_mcts_20iterations'
# Sample number:  100
# average success length:  3.422680412371134
# average best reward:  0.6915666666666667
# average best child reward:  0.6559833333333331

done_task_id = []

best_reward = []
best_child_reward = []
success_length_list = []

for file in os.listdir(trajectories_save_path):
    if not file.endswith('json'):
        continue
    with open(os.path.join(trajectories_save_path, file)) as f:
        result=json.load(f)

    if result['best child reward'] > 0:
        success_length_list.append(len(result['best_trajectory_index_list']))
    best_reward.append(result['best reward'])
    # best_child_reward.append(result['best child reward'])
    best_child_reward.append(0 if result['best child reward']==-1 else result['best child reward'])
print("Sample number: ", len(best_child_reward))
print("average success length: ", sum(success_length_list)/len(success_length_list))
print("average best reward: ", sum(best_reward)/len(best_reward))
print("average best child reward: ", sum(best_child_reward)/len(best_child_reward))
