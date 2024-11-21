import random
import json

random.seed(3)

'''
with open('webshop/data_split/From_ETO/test_indices.json', 'r') as file:
    my_list = json.load(file)
selected_elements = random.sample(my_list, 100)
with open('webshop/data_split/test_indices.json', 'w') as file:
    # 使用json.dump()方法将列表写入文件
    json.dump(selected_elements, file)
'''

with open('webshop/data_split/From_ETO/train_indices.json', 'r') as file:
    my_list = json.load(file)
selected_elements = random.sample(my_list, 1000)
with open('webshop/data_split/train_indices.json', 'w') as file:
    # 使用json.dump()方法将列表写入文件
    json.dump(selected_elements, file)