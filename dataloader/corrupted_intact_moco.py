import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import os
import random
from torchvision import transforms



class MoCoTextDataset(Dataset):
    def __init__(self, label_file, imgs_q_dir, imgs_k_dir, transform_q=None, transform_k=None, same_image=False):
        """
        初始化MoCo风格的数据集
        
        Args:
            label_file: 标签文件路径
            imgs_q_dir: 查询图像目录
            imgs_k_dir: 键图像目录
            char_to_idx: 字符到索引的映射
            transform_q: 查询图像的变换
            transform_k: 键图像的变换
            same_image: 是否使用相同的图像作为查询和键（如果为True，则忽略imgs_k_dir）
        """
        self.imgs_q_dir = imgs_q_dir
        self.imgs_k_dir = imgs_k_dir if not same_image else imgs_q_dir
        self.same_image = same_image
        self.transform_q = transform_q
        self.transform_k = transform_k

        # 读取标签文件
        self.samples = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    text = parts[1]  # 原始标签为文本
                    self.samples.append((img_name, text))
        # #随机重排samples
        # random.shuffle(self.samples)
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, img_dir, img_name):
        img_path = os.path.join(img_dir, img_name)
        
        try:
            # 防止PIL对截断图像的警告
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            
            image = Image.open(img_path).convert("RGB")
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            # 创建一个空白图像替代损坏的图像
            image = Image.new("RGB", (256, 64), color=(0, 0, 0))
            
        return image, img_path

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        
        # 加载查询图像
        image_q, img_path_q = self._load_image(self.imgs_q_dir, img_name)
        
        # 如果使用相同图像，则键图像就是查询图像
        # 否则从键图像目录加载
        if self.same_image:
            image_k, img_path_k = image_q, img_path_q
        else:
            image_k, img_path_k = self._load_image(self.imgs_k_dir, img_name)

        # 应用变换
        if self.transform_q:
            image_q = self.transform_q(image_q)
        
        if self.transform_k:
            image_k = self.transform_k(image_k)

        idx = torch.tensor(idx)


        # 返回一个包含查询图像和键图像的字典
        return {
            "imgs_q": image_q,
            "imgs_k": image_k,
            "idxs": idx
        }
    
class IndexMappedDataset(Dataset):
    def __init__(self, dataset, start_idx=0):
        self.dataset = dataset
        self.start_idx = start_idx
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['idxs'] = torch.tensor(self.start_idx + idx)
        return item

# 图像变换
transform = transforms.Compose(
    [
        transforms.Resize((64, 256)),  # 调整为适合OCR的大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 初始化syn80k数据集
syn80k_dataset = MoCoTextDataset(
        label_file="data/syth80k/train/labels_s.txt",
        imgs_q_dir="data/syth80k/train/s_incomplete",
        imgs_k_dir="data/syth80k/train/s_f/s_f",  # 可以使用相同或不同的目录
        transform_q=transform,
        transform_k=transform,
        same_image=False  # 如果为True，将忽略imgs_k_dir参数
    )
# 初始化IC13数据集
ic13_dataset = MoCoTextDataset(
        label_file="data/TII-ST/IC13/label_train.txt",
        imgs_q_dir="data/TII-ST/IC13/train_incomplete",
        imgs_k_dir="data/TII-ST/IC13/train_f",  # 可以使用相同或不同的目录
        transform_q=transform,
        transform_k=transform,
        same_image=False  # 如果为True，将忽略imgs_k_dir参数
    )

# 初始化IC15数据集
ic15_dataset = MoCoTextDataset(
        label_file="data/TII-ST/IC15/label.txt",
        imgs_q_dir="data/TII-ST/IC15/train_incomplete",
        imgs_k_dir="data/TII-ST/IC15/train_f",  # 可以使用相同或不同的目录
        transform_q=transform,
        transform_k=transform,
        same_image=False  # 如果为True，将忽略imgs_k_dir参数
    )

# 初始化IC17数据集
ic17_dataset = MoCoTextDataset(
        label_file="data/TII-ST/IC17/label.txt",
        imgs_q_dir="data/TII-ST/IC17/train_incomplete",
        imgs_k_dir="data/TII-ST/IC17/train_f",  # 可以使用相同或不同的目录
        transform_q=transform,
        transform_k=transform,
        same_image=False  # 如果为True，将忽略imgs_k_dir参数
    )


start_idx = 0
syn80k_mapped = IndexMappedDataset(syn80k_dataset, start_idx)
start_idx += len(syn80k_dataset)
ic13_mapped = IndexMappedDataset(ic13_dataset, start_idx)
start_idx += len(ic13_dataset)
ic15_mapped = IndexMappedDataset(ic15_dataset, start_idx)
start_idx += len(ic15_dataset)
ic17_mapped = IndexMappedDataset(ic17_dataset, start_idx)

combined_dataset = ConcatDataset([syn80k_mapped, ic13_mapped, ic15_mapped, ic17_mapped])


# 创建数据加载器
train_loader = DataLoader(
    combined_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=12
)