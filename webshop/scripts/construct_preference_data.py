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

# def get_whole_trajectory_index(node: list) -> int:


# trajectories_save_path = "webshop/trajectories_fixed/trajectories_forbid-double-search/trajectories_train_phi3_mcts_30iterations"
# trajectories_save_path = "webshop/trajectories_train_gpt-4o-mini_mcts_30iterations"
# trajectories_save_path = "webshop/trajectories_test_Phi-3_mcts_30iterations"
trajectories_save_path = "webshop/trajectories_iteration0/trajectories-MCTS-critique-disable_early_stop_train_train_llama31_T1.0_mcts_20iterations"

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
                # chosen = [{'role': 'assistant', 'content':node['children'][max_index]['state']['action']}]
                # rejected = [{'role': 'assistant', 'content':node['children'][min_index]['state']['action']}]
                chosen = [{'from': 'gpt', 'value':node['children'][max_index]['state']['action']}]
                rejected = [{'from': 'gpt', 'value':node['children'][min_index]['state']['action']}]
                if node['depth'] != 0:
                    # prompt.append({'role': 'assistant', 'content':node['state']['action']})
                    # prompt.append({'role': 'user', 'content':node['state']['observation']})
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
    
    # print(preference_data)
json.dump(preference_data, open("webshop/webshop-Llama31-CritiqueMCTS.json", "w"), indent=4)

