import random
import json

# 确保生成的随机数数量不超过范围
max_value = 90000
count = 1200

# 生成一个包含1000个不重复随机数的列表
train_numbers = random.sample(range(max_value + 1), count)

# 将随机数列表保存到JSON文件
with open('hotpot/data_split/train_indices.json', 'w', encoding='utf-8') as f:
    json.dump(train_numbers[:1000], f, ensure_ascii=False)


with open('hotpot/data_split/valid_indices.json', 'w', encoding='utf-8') as f:
    json.dump(train_numbers[1000:1100], f, ensure_ascii=False)


with open('hotpot/data_split/test_indices.json', 'w', encoding='utf-8') as f:
    json.dump(train_numbers[1100:1200], f, ensure_ascii=False)
