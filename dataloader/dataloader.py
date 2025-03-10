import torch
from torch.utils.data import DataLoader, ConcatDataset
from .corrupted import CTCImageDataset, char_to_idx
from torchvision import transforms

# 图像变换
transform = transforms.Compose(
    [
        transforms.Resize((64, 256)),  # 调整为适合OCR的大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 创建完整数据集
train_F_dataset = CTCImageDataset(
    label_file="data/syth80k/train/labels_s.txt",
    img_dir="data/syth80k/train/s_f/s_f",
    char_to_idx=char_to_idx,
    transform=transform,
)



# 创建损坏数据集
train_C_dataset = CTCImageDataset(
    label_file="data/syth80k/train/labels_s.txt",
    img_dir="data/syth80k/train/s_incomplete",
    char_to_idx=char_to_idx,
    transform=transform,
)

test_C_dataset = CTCImageDataset(
    label_file="data/syth80k/test/labels_t.txt",
    img_dir="data/syth80k/test/t_incomplete",
    char_to_idx=char_to_idx,
    transform=transform,
)
# 合并数据集
combined_dataset = ConcatDataset([train_F_dataset, train_C_dataset])



# 创建DataLoader时的collate_fn来处理变长序列
def ctc_collate_fn(batch):
    imgs = []
    data_samples = []

    for sample in batch:
        imgs.append(sample["inputs"])
        data_samples.append(sample["data_samples"])

    # 将图像堆叠为一个批次
    imgs = torch.stack(imgs, 0)

    return {"inputs": imgs, "data_samples": data_samples}


# 创建数据加载器
train_loader = DataLoader(
    combined_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=12,
    collate_fn=ctc_collate_fn,
)

test_loader = DataLoader(
    test_C_dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=ctc_collate_fn
)