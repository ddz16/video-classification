from PIL import Image
import os
import numpy as np

# 父文件夹路径
parent_folder = "data"

# 获取所有子文件夹路径
sub_folders = [os.path.join(parent_folder, folder, 'frames') for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]
print(sub_folders)

# 初始化均值和方差列表
means = []
variances = []

# 遍历所有子文件夹
for sub_folder in sub_folders:
    # 获取子文件夹中所有图片路径
    image_files = [os.path.join(sub_folder, file) for file in os.listdir(sub_folder) if file.endswith(".jpg") or file.endswith(".png")]
    
    # 遍历所有图片
    for image_file in image_files:
        # 打开图片
        image = Image.open(image_file)
        
        # 将图片转换为numpy数组
        image_array = np.array(image)
        
        # 计算均值和方差
        mean = np.mean(image_array, axis=(0, 1))
        variance = np.var(image_array, axis=(0, 1))
        
        # 将均值和方差添加到列表中
        means.append(mean)
        variances.append(variance)

# 计算所有图片的平均均值和方差
mean_all = np.mean(means, axis=0)
variance_all = np.mean(variances, axis=0)

# 输出结果
print("均值：", mean_all)
print("方差：", variance_all)
