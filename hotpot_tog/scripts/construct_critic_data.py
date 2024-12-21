import os
import json
import copy

preferred_value = []
less_preferred_value = []
difference_value = []

def get_max_min_children_index(node: list) -> tuple:
    value_list = []
    for child in node['children']:
        value_list.append(child['value'])
    return value_list.index(max(value_list)), value_list.index(min(value_list))

def get_pointwise_data(trajectories_save_path):
    pointwise_data = []
    sample_num = 0
    for file in os.listdir(trajectories_save_path):
        # if sample_num==1000:
        #     break
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
                if node['depth']==depth:
                    if node['depth'] != 0:
                        prompt.append({'from': 'gpt', 'value':node['state']['action']})
                        prompt.append({'from': 'human', 'value':node['state']['observation']})
                    for (child_index, child) in enumerate(node['children']):
                        is_effective = 0
                        if child_index == 0:
                            continue
                        critique_prompt = copy.deepcopy(prompt)
                        original_observation = critique_prompt[-1]['value']
                        critique_prompt[-1]['value'] = original_observation + node['children'][child_index]['state']['regenerate_prompt']+ "\n" + original_observation
                        critique = [{'from': 'gpt', 'value':node['children'][child_index]['state']['critique']}]
                        
                        
                        if node['children'][child_index]['value'] > node['children'][child_index-1]['value']:
                            is_effective = 1
                            print("critique is effective")
                        else:
                            print("critique is ineffective")
                        pointwise_data.append({     
                            "critique_prompt": critique_prompt,
                            'critique': critique,
                            "is_effective": is_effective
                            })
                    break
                node = node['children'][best_trajectory_index]
                
    return pointwise_data


trajectories_save_path = "hotpot/trajectories-MCTS-critique-disable_early_stop_train_llama31_mcts_20iterations"

preference_data = get_pointwise_data(trajectories_save_path)
json.dump(preference_data, open("hotpot/hotpot-mcts_critic_data-iteration20.json", "w"), indent=4)

