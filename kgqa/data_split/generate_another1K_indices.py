import random
import json

# 假设这是你提供的两个排除列表
with open("hotpot/data_split/test_indices.json", 'r') as file:
    excluded_list1 = json.load(file)
with open("hotpot/data_split/train_indices.json", 'r') as file:
    excluded_list2 = json.load(file)
with open("hotpot/data_split/valid_indices.json", 'r') as file:
    excluded_list3 = json.load(file)

# 合并两个列表并去重
excluded_numbers = set(excluded_list1 + excluded_list2 + excluded_list3)

# 计算可用的随机数范围
available_numbers = set(range(1, 90001)) - excluded_numbers

# 确保可用的随机数足够生成1000个不重复的随机数
if len(available_numbers) < 1000:
    raise ValueError("排除的数值太多，无法生成1000个不重复的随机数。")

# 生成1000个不重复的随机数
random_numbers = random.sample(available_numbers, 1000)

# 将随机数列表保存到JSON文件
with open('hotpot/data_split/train_another1K_indices.json', 'w', encoding='utf-8') as f:
    json.dump(random_numbers, f, ensure_ascii=False)