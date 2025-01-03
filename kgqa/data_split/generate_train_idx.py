import random
import json


# 设置随机数的范围
start = 0
end = 3000

with open('kgqa/data_split/test_indices.json', 'r') as f:
    existing_num_list = json.load(f)

# 假设existing_num_list是已经存在的数字列表
# existing_num_list = [i for i in range(1000, 2000)]  # 示例：1000到1999之间的数字

# 生成0到3000的所有数字的列表
all_numbers = list(range(start, end + 1))

# 移除existing_num_list中的数字
remaining_numbers = [num for num in all_numbers if num not in existing_num_list]

# 检查剩余数字是否足够生成1000个不重复的随机数
if len(remaining_numbers) < 1000:
    raise ValueError("Not enough unique numbers remaining to generate 1000 unique random numbers.")

# 从剩余的列表中随机选择1000个数字
train_indices = random.sample(remaining_numbers, 1000)

# 指定要保存的JSON文件名
filename = 'kgqa/data_split/train_indices.json'

# 将列表保存到JSON文件中
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(train_indices, f, ensure_ascii=False, indent=4)