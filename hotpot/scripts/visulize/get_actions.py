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
with open('hotpot/trajectories-backup-MCTS_test_llama31-2_mcts_30iterations/39209.json', 'r') as f:
    root = json.load(f)

with open('hotpot/trajectories-backup-MCTS-conditional_test_llama31-3_mcts_30iterations/39209.json', 'r') as f:
    root = json.load(f)

node = copy.deepcopy(root)

best_trajectory_index_list = node['best_trajectory_index_list']

while node["is_terminal"]== False and node["children"] is not None:
    depth = node['depth']
    print(f'--------Depth: {depth}---------------')
    for child in node['children']:
        # print(parse_action(child['state']["action"]), 'value: ', child['value'])
        print(child['state'])
    node = node['children'][best_trajectory_index_list[depth]]
    