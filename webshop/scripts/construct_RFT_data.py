import os
import json

def get_max_min_children_index(node: list) -> tuple:
    value_list = []
    for child in node['children']:
        value_list.append(child['value'])
    return value_list.index(max(value_list)), value_list.index(min(value_list))

# def get_whole_trajectory_index(node: list) -> int:

# version1: rollout with value(used)
# trajectories_save_path = "hotpot/trajectories_rollout_by_value/trajectories_train_gpt-4o_mcts_30iterations_1000samples"

# version2: rollout with UCT
# trajectories_save_path = "hotpot/trajectories_train_gpt-4o-mini_mcts_30iterations"

# phi-3
trajectories_save_path = "webshop/trajectories-Phi3/trajectories_train_phi3_mcts_30iterations"

RFT_data = []
sample_num = 0
for file in os.listdir(trajectories_save_path):
    if sample_num==1000:
        break
    sample_num += 1
    if not file.endswith('json'):
        continue
    with open(os.path.join(trajectories_save_path, file)) as f:
        root=json.load(f)
    
    # whole_trajectory_index = get_whole_trajectory_index(root)

    prompt = [{'from': 'human', 'value': root['messages'][0]['content']}]
    prompt.append({'from': 'gpt', 'value': root['messages'][1]['content']})
    prompt.append({'from': 'human', 'value': root['messages'][-1]['content']})
    # early_stop = False
    best_trajectory_index_list = root['best_trajectory_index_list'] # children index
    if len(best_trajectory_index_list) == 0:
        continue
    node = root
    for depth, best_trajectory_index in enumerate(best_trajectory_index_list): # preference for each depth
        chosen = {'from': 'gpt', 'value':node['children'][best_trajectory_index]['state']['action']}
        prompt.append(chosen)
        prompt.append({'from': 'human', 'value':node['children'][best_trajectory_index]['state']['observation']})
        node = node['children'][best_trajectory_index]
    prompt.pop()
    RFT_data.append( 
                {     
                "id": str(file.split(".")[0]),
                "conversations": prompt.copy(),
                "reward": 1.0,
                "source": "human"
                }
            )

    # print(preference_data)
json.dump(RFT_data, open("webshop/webshop-mcts_RFT_data-iteration1.json", "w"), indent=4)

