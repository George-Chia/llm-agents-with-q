import json
import re
import copy

def parse_action(llm_output: str) -> str:
    llm_output = llm_output.strip()
    try:
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
    except:
        action = 'nothing'
    assert action is not None
    return action


# 加载JSON文件
with open('hotpot/trajectories-MCTS-critique_train_llama31_mcts_30iterations/89.json', 'r') as f:
    root = json.load(f)

with open('hotpot/trajectories-MCTS-critique-disable_early_stop_train_llama31_mcts_20iterations/89.json', 'r') as f:
    root = json.load(f)



node = copy.deepcopy(root)

best_trajectory_index_list = node['best_trajectory_index_list']

while node["is_terminal"]== False and node["children"] is not None:
    depth = node['depth']
    print(f'--------Depth: {depth}---------------')
    for index,child in enumerate(node['children']):
        # print(parse_action(child['state']["action"]), 'value: ', child['value'])
        print(child['state'])
        print('value: ', child['value'])
    node = node['children'][best_trajectory_index_list[depth]]
    