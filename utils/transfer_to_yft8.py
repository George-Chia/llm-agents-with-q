import json
import os

# 假设你的JSON文件名为example.json，位于当前工作目录
json_file_path = '/home/zhaiyuanzhao/LLM-Agents-with-Q/hotpot/trajectories_SIGIR/Iteration1/test/trajectories-Critique-MCTS-3n-policy-critique-iter1Epoch3-huan_test_llama31_mcts_30iterations/17307.json'

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 将数据写回到新的JSON文件中，确保使用UTF-8编码
new_json_file_path = os.path.splitext(json_file_path)[0] + '_utf8.json'
with open(new_json_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f'文件已保存为 {new_json_file_path}')