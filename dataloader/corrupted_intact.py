import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from mmocr.structures import TextRecogDataSample
from mmengine.structures import LabelData


# 字符到索引的映射，包括空白字符
def build_char_map(label_files=None):
    # 直接定义字符集：26个小写字母 + 26个大写字母 + 10个数字
    chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # 创建映射，0预留给空白符
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
    char_to_idx["<blank>"] = 0

    # 创建反向映射
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    return char_to_idx, idx_to_char


# 保留原有的数据集类
class CTCImageDataset(Dataset):
    def __init__(self, label_file, img_dir, char_to_idx, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.char_to_idx = char_to_idx

        # 读取标签文件
        self.samples = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    text = parts[1]  # 原始标签为文本
                    self.samples.append((img_name, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 读取图片时添加错误处理
        try:
            # 防止PIL对截断图像的警告
            from PIL import ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True

            image = Image.open(img_path).convert("RGB")
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            # 创建一个空白图像替代损坏的图像
            image = Image.new("RGB", (256, 64), color=(0, 0, 0))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 将文本标签转换为索引序列
        label = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

        # 创建符合MMEngine格式的数据结构
        data_sample = TextRecogDataSample()

        # 添加图像元信息
        img_meta = {
            "img_path": img_path,
            "ori_shape": image.shape if hasattr(image, "shape") else (64, 256, 3),
            "img_shape": image.shape if hasattr(image, "shape") else (64, 256, 3),
        }

        # 创建gt_text
        gt_text = LabelData(metainfo=img_meta)
        gt_text.item = text
        gt_text.indices = label
        gt_text.length = len(label)

        # 设置数据样本的gt_text字段
        data_sample.gt_text = gt_text

        # 返回的是(img, data_sample)，符合mmengine的数据格式
        return {"inputs": image, "data_samples": data_sample}


char_to_idx, idx_to_char = build_char_map(
    ["data/syth80k/train/labels_s.txt", "data/syth80k/test/labels_t.txt"]
)

num_chars = len(char_to_idx)


