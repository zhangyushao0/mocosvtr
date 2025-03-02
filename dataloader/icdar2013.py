import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


# 保留原有的字符映射功能
def build_char_map(label_files):
    chars = set()
    for file in label_files:
        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label = parts[1]
                    for char in label:
                        chars.add(char)

    # 添加空白字符（CTC需要）
    chars = sorted(list(chars))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # 0预留给空白符
    char_to_idx["<blank>"] = 0

    return char_to_idx, {idx: char for char, idx in char_to_idx.items()}


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
                    img_name = parts[0][:-1]  # 去除最后一个逗号
                    text = parts[1]  # 原始标签为文本下面去除"
                    text = text.replace('"', "")
                    self.samples.append((img_name, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 读取图片
        image = Image.open(img_path).convert("RGB")

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 将文本标签转换为索引序列
        label = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

        # 返回的是(img, label)，对应mmengine中的数据格式
        return {"img": image, "text": text, "label": label, "length": len(label)}


char_to_idx, idx_to_char = build_char_map(
    [
        "data/icdar2013/textrecog_imgs/train/gt.txt",
        "data/icdar2013/textrecog_imgs/test/gt.txt",
    ]
)

num_chars = len(char_to_idx)

# 图像变换
transform = transforms.Compose(
    [
        transforms.Resize((64, 224)),  # 调整为适合OCR的大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 创建数据集
train_dataset = CTCImageDataset(
    label_file="data/icdar2013/textrecog_imgs/train/gt.txt",
    img_dir="data/icdar2013/textrecog_imgs/train/",
    char_to_idx=char_to_idx,
    transform=transform,
)

test_dataset = CTCImageDataset(
    label_file="data/icdar2013/textrecog_imgs/test/gt.txt",
    img_dir="data/icdar2013/textrecog_imgs/test/",
    char_to_idx=char_to_idx,
    transform=transform,
)


# 创建DataLoader时的collate_fn来处理变长序列
def ctc_collate_fn(batch):
    imgs = []
    labels = []
    texts = []
    lengths = []

    for sample in batch:
        imgs.append(sample["img"])
        labels.extend(sample["label"].tolist())
        texts.append(sample["text"])
        lengths.append(sample["length"])

    # 将图像堆叠为一个批次
    imgs = torch.stack(imgs, 0)

    # 创建目标长度张量
    target_lengths = torch.tensor(lengths, dtype=torch.long)

    # 平展的标签张量
    targets = torch.tensor(labels, dtype=torch.long)

    return {
        "imgs": imgs,
        "labels": targets,
        "target_lengths": target_lengths,
        "texts": texts,
    }


# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    collate_fn=ctc_collate_fn,
)

test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=ctc_collate_fn
)
