import os
import random

# 获取当前文件夹下的所有子文件夹的名称
subfolders = [f.name for f in os.scandir('.') if f.is_dir()]

# 计算70%、10%和20%的子文件夹数量
train_count = round(0.7 * len(subfolders))
valid_count = round(0.0 * len(subfolders))

# 随机选取子文件夹的名称
train_data = random.sample(subfolders, train_count)
valid_data = random.sample(set(subfolders) - set(train_data), valid_count)
test_data = list(set(subfolders) - set(train_data) - set(valid_data))

# 打印结果
print("Train data:", train_data)
print("Valid data:", valid_data)
print("Test data:", test_data)

# 将数据写入对应的文件
with open('train.txt', 'w') as f:
    f.write('\n'.join(train_data))
    
with open('valid.txt', 'w') as f:
    f.write('\n'.join(valid_data))
    
with open('test.txt', 'w') as f:
    f.write('\n'.join(test_data))