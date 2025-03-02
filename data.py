import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class CorruptedImageDataset(Dataset):
    def __init__(self, label_file, img_dir, transform=None):
        """
        初始化数据集

        参数:
            label_file (string): 标签文件的路径
            img_dir (string): 图片目录的路径
            transform (callable, optional): 可选的图像变换
        """
        self.img_dir = img_dir
        self.transform = transform

        # 读取标签文件
        self.samples = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:  # 确保行有效
                    img_name = parts[0]
                    label = parts[1]
                    self.samples.append((img_name, label))

        # 创建类别到索引的映射
        self.class_to_idx = {
            cls: i
            for i, cls in enumerate(sorted(set([label for _, label in self.samples])))
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 读取图片
        image = Image.open(img_path).convert("RGB")

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 将标签转换为数字索引
        label_idx = self.class_to_idx[label]

        return image, label_idx


# 定义图像变换
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 创建训练和测试数据集
train_dataset = CorruptedImageDataset(
    label_file="./train/label.txt", img_dir="./Corrupted_Image", transform=transform
)

test_dataset = CorruptedImageDataset(
    label_file="./test/label.txt", img_dir="./Corrupted_Image", transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
