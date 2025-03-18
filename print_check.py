import torch

print(torch.cuda.is_available())  # 检查 CUDA 是否可用
print(torch.version.cuda)  # 打印 CUDA 版本
print(torch.cuda.get_device_name(0))  # 打印 GPU 名称
