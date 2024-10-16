"""
保存系统使用的参数，以及一些值的计算
"""
import numpy as np

# 发起节点存储成本
CollaborativeInitiatorCost = 50

# 协作节点存储成本
CollaboratorCost = np.random.uniform(42.5, 57.5)

# # 总存储空间大小 GB
# StorageSpaceSize = np.random.randint(1000, 2000)
#
# # 已用存储空间大小 GB
# UsedStorageSpaceSize = np.random.randint(300, 1000)

# # 磁盘写入的速度 m/s
# WriteSpeed = np.random.randint(400, 500)
#
# # 与发起节点的距离 km
# Distance = np.random.randint(1, 200)

# 满意度 >0 <=1
Satisfaction = np.random.uniform(0, 1)

# 偏好度 >0 <=1
Preference = np.random.uniform(0, 1)
