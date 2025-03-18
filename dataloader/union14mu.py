import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import lmdb
import io
import os
import random
from torchvision import transforms
import numpy as np


class LMDBImageDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, max_samples=None):
        """
        从LMDB数据库读取图像的数据集

        Args:
            lmdb_path: LMDB数据库路径
            transform: 应用于图像的变换
            max_samples: 最大样本数量（如果为None则使用全部）
        """
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.max_samples = max_samples

        # 防止PIL对截断图像的警告
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # 临时打开环境获取键列表
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        self.keys = []
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                self.keys.append(key)
                if max_samples is not None and len(self.keys) >= max_samples:
                    break
        env.close()

        # 这些变量将在每个worker中初始化
        self.env = None

    def _init_db(self):
        """初始化LMDB环境 - 仅在需要时调用"""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def __len__(self):
        return len(self.keys)

    def _load_image(self, key):
        self._init_db()  # 确保环境已初始化

        with self.env.begin() as txn:
            value = txn.get(key)
            if value is None:
                print(f"Key {key} not found in LMDB")
                # 创建一个空白图像作为替代
                image = Image.new("RGB", (256, 64), color=(0, 0, 0))
            else:
                try:
                    # 将字节数据转换为图像
                    image = Image.open(io.BytesIO(value)).convert("RGB")
                except Exception as e:
                    print(f"Error loading image for key {key}: {e}")
                    # 创建一个空白图像替代损坏的图像
                    image = Image.new("RGB", (256, 64), color=(0, 0, 0))

        return image

    def __getitem__(self, idx):
        key = self.keys[idx]

        # 加载图像
        image = self._load_image(key)

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 只返回图像
        return {"imgs": image}


# 工作进程初始化函数
def worker_init_fn(worker_id):
    # 为每个工作进程设置不同的随机种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# 图像变换示例
transform = transforms.Compose(
    [
        transforms.Resize((32, 100)),  # 调整大小以匹配模型输入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 使用示例
lmdb_dataset = LMDBImageDataset(
    lmdb_path="G:/download/Union14M-U/cc_lmdb",
    transform=transform,
    max_samples=None,  # 设置为None使用全部数据
)

# 使用worker_init_fn初始化每个工作进程
loader = DataLoader(
    lmdb_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    worker_init_fn=worker_init_fn,
    pin_memory=True,
)
