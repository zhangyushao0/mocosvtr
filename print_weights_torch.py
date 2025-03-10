import torch
import sys

# 从命令行参数获取权重文件路径
weights_path = sys.argv[1]

# 加载权重
state_dict = torch.load(weights_path)

# 打印权重结构
for item in state_dict["state_dict"].items():
    print(item[0])
