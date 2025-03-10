import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from mmengine import load

class LMDBTextDataset(Dataset):
    def __init__(self, lmdb_file, img_dir, char_to_idx, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.char_to_idx = char_to_idx
        
        # 防止PIL对截断图像的警告
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        # 使用MMEngine的load函数从LMDB读取数据
        data = load(lmdb_file)
        
        # 解析数据结构
        self.samples = []
        if 'data_list' in data:
            for item in data['data_list']:
                img_path = item.get('img_path', '')
                instances = item.get('instances', [])
                
                if instances and img_path:
                    text = instances[0].get('text', '')
                    # 提取图像文件名
                    img_name = os.path.basename(img_path)
                    self.samples.append((img_name, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 读取图片时添加错误处理
        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            # 创建一个空白图像替代损坏的图像
            image = Image.new("RGB", (224, 64), color=(0, 0, 0))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 将文本标签转换为索引序列
        label = torch.tensor([self.char_to_idx.get(c, 0) for c in text], dtype=torch.long)

        # 返回与原数据集格式一致的字典
        return {"img": image, "text": text, "label": label, "length": len(label)}


# 图像变换
transform = transforms.Compose(
    [
        transforms.Resize((64, 224)),  # 调整为适合OCR的大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 创建数据集
train_dataset = LMDBTextDataset(
    label_file="data/syth80k/train/labels_s.txt",
    img_dir="data/syth80k/train/s_incomplete",
    char_to_idx=char_to_idx,
    transform=transform,
)
