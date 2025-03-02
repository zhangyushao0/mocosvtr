import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms


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


# 修改数据集类来适应CTC任务
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

        # 读取图片
        image = Image.open(img_path).convert("RGB")

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 将文本标签转换为索引序列
        label = [self.char_to_idx[c] for c in text]

        return image, label


# 定义一个简单的CNN+RNN模型用于OCR
class CRNN(nn.Module):
    def __init__(self, num_chars):
        super(CRNN, self).__init__()

        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 计算CNN输出后的特征图大小
        self.time_steps = 28  # 这个值需要根据你的图像大小和CNN架构计算

        # 修改RNN部分: 不使用Sequential
        self.rnn1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(512, num_chars)

    def forward(self, x):
        # CNN特征提取
        conv = self.cnn(x)

        # 转换为RNN输入格式
        batch, c, h, w = conv.size()
        conv = conv.view(batch, c, -1)
        conv = conv.permute(0, 2, 1)  # [batch, width, channels]

        # RNN序列建模 - 修改为顺序调用LSTM
        rnn_out1, _ = self.rnn1(conv)
        rnn_out, _ = self.rnn2(rnn_out1)

        # 通过全连接层预测字符
        output = self.fc(rnn_out)

        # 对数Softmax
        log_probs = nn.functional.log_softmax(output, dim=2)

        # 转置为CTC要求的格式 [T, N, C]
        log_probs = log_probs.permute(1, 0, 2)

        return log_probs


# 创建DataLoader时的collate_fn来处理变长序列
def ctc_collate_fn(batch):
    images, labels = zip(*batch)

    # 将图像堆叠为一个批次
    images = torch.stack(images, 0)

    # 获取目标序列长度
    target_lengths = [len(label) for label in labels]

    # 将标签平展为一维张量
    flattened_targets = []
    for label in labels:
        flattened_targets.extend(label)

    # 转换为pytorch张量
    targets = torch.tensor(flattened_targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets, target_lengths


# 添加评估函数
def evaluate(model, data_loader, device, idx_to_char):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets, target_lengths in data_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # 前向传播获取预测结果
            log_probs = model(images)

            # 解码预测结果（贪婪解码）
            pred_sizes = torch.IntTensor([log_probs.size(0)] * batch_size)
            _, preds = log_probs.max(2)
            preds = preds.transpose(1, 0).contiguous().cpu()

            # 解码预测结果（移除重复字符和空白符）
            pred_strings = []
            target_strings = []

            # 处理预测结果
            for i, pred in enumerate(preds):
                # 获取当前样本的真实标签
                target_length = target_lengths[i].item()
                target_start = sum(target_lengths[:i].tolist()) if i > 0 else 0
                target = (
                    targets[target_start : target_start + target_length].cpu().tolist()
                )
                target_str = "".join([idx_to_char[idx] for idx in target])
                target_strings.append(target_str)

                # 解码预测结果（移除重复字符和空白符）
                pred_string = ""
                prev_char = -1
                for p in pred[: pred_sizes[i]]:
                    p = p.item()
                    if p != 0 and p != prev_char:  # 不是空白符且不重复
                        pred_string += idx_to_char[p]
                    prev_char = p
                pred_strings.append(pred_string)

            # 计算正确数量
            for pred_str, target_str in zip(pred_strings, target_strings):
                if pred_str == target_str:
                    correct += 1

            total += batch_size

    accuracy = correct / total
    return accuracy


def train_ocr_model():
    # 构建字符映射
    char_to_idx, idx_to_char = build_char_map(
        ["data/syth80k/train/label.txt", "data/syth80k/test/label.txt"]
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

    # 创建训练数据集
    train_dataset = CTCImageDataset(
        label_file="data/syth80k/train/label.txt",
        img_dir="data/syth80k/train/Corrupted_Image",
        char_to_idx=char_to_idx,
        transform=transform,
    )

    # 创建测试数据集
    test_dataset = CTCImageDataset(
        label_file="data/syth80k/test/label.txt",
        img_dir="data/syth80k/test/Corrupted_Image",
        char_to_idx=char_to_idx,
        transform=transform,
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=ctc_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=ctc_collate_fn,
    )

    # 创建模型
    model = CRNN(num_chars)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=5 / (10**4) * 2048 / 2048)

    # CTC损失函数
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean")

    # 训练循环
    num_epochs = 10
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            # 计算输入序列长度（假设所有序列长度相同）
            input_lengths = torch.full(
                (images.size(0),), model.time_steps, dtype=torch.long
            ).to(device)

            # 前向传播
            optimizer.zero_grad()
            log_probs = model(images)

            # 计算CTC损失
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        # 打印每个epoch的平均损失
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch: {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # 在测试集上评估
        accuracy = evaluate(model, test_loader, device, idx_to_char)
        print(f"Epoch: {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}")

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_ocr_model.pth")
            print(f"Saved best model with accuracy: {best_accuracy:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), "ocr_model.pth")
    print(f"最终测试准确率: {best_accuracy:.4f}")

    return model, char_to_idx, idx_to_char


train_ocr_model()
