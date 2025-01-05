import random

# 生成一个包含100个不重复随机数的列表
random_numbers = random.sample(range(1001), 100)  # range(1001)生成0到1000的整数，包括1000

# 打印结果
print(random_numbers)