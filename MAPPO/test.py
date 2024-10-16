import pandas as pd
import random

# 设定随机数的范围
min_val = 0
max_val = 96

# 设定两个随机数加起来除以2的目标值列表
target_values = [92,
90,
94,
93,
90,
91,
90,
90,
90,
92,
90,
90,
85,
91,
91,
90,
88,
90,
92,
91,
92,
89,
90,
89,
86,
90,
91,
88,
89,
86,
90,
90,
93,
88,
89,
91,
89,
82,
90,
92,
83,
92,
89,
91,
90,
91,
90,
90,
90,
92,
90,
93,
91

]

# 创建一个空的DataFrame来存储结果
results = []

# 遍历目标值列表
for target in target_values:
    # 计算两个随机数的和应该是多少
    sum_target = target * 4

    # 确保生成的随机数之和不会超出我们设定的范围
    while True:
        num1 = random.randint(min_val, max_val)
        num2 = random.randint(min_val, max_val)
        num3 = random.randint(min_val, max_val)
        num4 = random.randint(min_val, max_val)
        # if num1 + num2 == sum_target:
        #     break
        if num1 + num2 + num3 + num4 == sum_target:
            break

            # 将结果添加到列表中
    results.append([num1, num2, num3, num4, target])

# 将结果列表转换为DataFrame
df = pd.DataFrame(results, columns=['Random1', 'Random2', "Random3", "Random4", 'Target'])

# 写入Excel文件
excel_path = 'results1.xlsx'
df.to_excel(excel_path, index=False, engine='openpyxl')

print(f'Results have been written to {excel_path}')
