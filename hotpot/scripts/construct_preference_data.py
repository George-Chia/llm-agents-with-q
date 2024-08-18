import os
import json

preferred_value = []
less_preferred_value = []
difference_value = []

def get_max_min_children_index(node: list) -> tuple:
    value_list = []
    for child in node['children']:
        value_list.append(child['value'])
    return value_list.index(max(value_list)), value_list.index(min(value_list))

def get_preference_data(trajectories_save_path):
    preference_data = []
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
        node = root

        for depth, best_trajectory_index in enumerate(best_trajectory_index_list): # preference for each depth
            is_preference = True
            while True:
                max_index, min_index = get_max_min_children_index(node)

                # if node['children'][max_index]['children'] == None:
                #     print(f"early stop in the next depth")
                #     early_stop = True 
                if node['depth']==depth:
                    if max_index==min_index:
                        print(f"max_index==min_index at depth={depth}")
                        is_preference = False
                    else:
                        preferred_value.append(node['children'][max_index]['value'])
                        less_preferred_value.append(node['children'][min_index]['value'])
                        difference_value.append(node['children'][max_index]['value'] - node['children'][min_index]['value'])
                    # if node['children'][max_index]['reward']==1 and node['children'][min_index]['reward']==1:
                    #     print(f"max_index reward==min_index reward==1 at depth={depth}")
                    #     is_preference = False
                    chosen = [{'from': 'gpt', 'value':node['children'][max_index]['state']['action']}]
                    rejected = [{'from': 'gpt', 'value':node['children'][min_index]['state']['action']}]
                    if node['depth'] != 0:
                        prompt.append({'from': 'gpt', 'value':node['state']['action']})
                        prompt.append({'from': 'human', 'value':node['state']['observation']})
                    node = node['children'][best_trajectory_index]
                    break
                # extend_index = max_index
                node = node['children'][best_trajectory_index]
                

            if is_preference:
                preference_data.append( 
                    {     
                    "prompt": prompt.copy(),
                    'chosen': chosen,
                    "rejected": rejected
                    }
                )
    return  preference_data
    # print(preference_data)

# def get_whole_trajectory_index(node: list) -> int:

# version1: rollout with value(used)
# trajectories_save_path = "hotpot/trajectories_rollout_by_value/trajectories_train_gpt-4o_mcts_30iterations_1000samples"

# version2: rollout with UCT
# trajectories_save_path = "hotpot/trajectories_train_gpt-4o-mini_mcts_30iterations"

# phi-3
# trajectories_save_path = "hotpot/trajectories_Phi3_1T/trajectories_train_Phi-3_mcts_30iterations"
# trajectories_save_path = "hotpot/trajectories_test_Phi-3_mcts_30iterations"
# trajectories_save_path = "hotpot/trajectories_test_llama31_mcts_30iterations"


trajectories_save_path = "hotpot/trajectories_train_Phi-3_mcts_30iterations"

preference_data = get_preference_data(trajectories_save_path)
json.dump(preference_data, open("hotpot/hotpot-mcts_pm_data-iteration50.json", "w"), indent=4)

