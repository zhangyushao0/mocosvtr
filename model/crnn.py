import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch
from mmengine.model import BaseModel


# 定义CRNN网络结构，与原来一样，但作为MMEngine模型的backbone
class CRNNBackbone(nn.Module):
    def __init__(self, num_chars, input_channels=3):
        super(CRNNBackbone, self).__init__()

        # 使用ResNet风格的CNN特征提取
        self.cnn = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 引入残差连接的块
            self._make_residual_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._make_residual_block(128, 256),
            self._make_residual_block(256, 256),
            self._make_residual_block(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),
            self._make_residual_block(256, 512),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # 最后的特征提取
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 添加注意力机制
            SpatialAttention(),
        )

        # 使用更高级的RNN
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256), BidirectionalLSTM(256, 256, 512)
        )

        # 全连接层
        self.fc = nn.Linear(512, num_chars)

        # 时间步长，用于CTC计算
        self.time_steps = 28

    def _make_residual_block(self, in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        # CNN特征提取
        conv = self.cnn(x)

        # 转换为RNN输入格式
        batch, c, h, w = conv.size()
        conv = conv.view(batch, c, -1)
        conv = conv.permute(0, 2, 1)  # [batch, width, channels]

        # RNN序列建模
        output = self.rnn(conv)

        # 通过全连接层预测字符
        output = self.fc(output)

        # 对数Softmax
        log_probs = F.log_softmax(output, dim=2)

        # 转置为CTC要求的格式 [T, N, C]
        log_probs = log_probs.permute(1, 0, 2)

        return log_probs


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入输出通道不一致，添加1x1卷积进行调整
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 生成空间注意力图：最大池化和平均池化的特征融合
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)

        # 应用注意力
        return x * attention


# 封装双向LSTM
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output


# 基于MMEngine的CRNN模型
class MMCRNN(BaseModel):
    def __init__(self, num_chars):
        super(MMCRNN, self).__init__()
        self.backbone = CRNNBackbone(num_chars)

        # CTC损失函数
        self.loss_fn = nn.CTCLoss(blank=0, reduction="mean")
        # 添加梯度裁剪参数
        self.max_grad_norm = 5.0

    def forward(self, imgs, labels=None, target_lengths=None, texts=None, mode="tensor"):
        if self.training and mode == "loss":
            imgs = self.apply_augmentation(imgs)
        
        x = self.backbone(imgs)
        
        # 动态计算输入长度
        batch_size = imgs.size(0)
        seq_length = x.size(0)  # 应该是时间步长维度
        input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long).to(imgs.device)
        
        if mode == "loss":
            # 确保标签和长度有效
            valid_batch = True
            if labels is None or target_lengths is None:
                valid_batch = False
                print("警告: 缺少标签或目标长度")
            
            # 检查输入序列长度是否足够
            if torch.any(input_lengths < target_lengths):
                print("警告: 输入序列长度小于目标序列长度")
                valid_batch = False
                
            if valid_batch:
                loss = self.loss_fn(x, labels, input_lengths, target_lengths)
                
                # 安全检查
                if torch.isnan(loss) or torch.isinf(loss):
                    print("警告: 发现nan或inf")
                    loss = torch.tensor(10.0, device=loss.device, requires_grad=True)
            else:
                # 无效批次返回零损失
                loss = torch.tensor(0.0, device=imgs.device, requires_grad=True)
                
            return {"loss": loss}

        elif mode == "predict":
            # 预测模式，返回logits和labels用于评估
            return x, labels, target_lengths, texts

        # 默认只返回张量
        return x

    def apply_augmentation(self, imgs):
        # 添加简单的数据增强
        batch_size = imgs.shape[0]
        device = imgs.device
        
        # 随机应用增强，每个批次有50%的概率
        if torch.rand(1).item() < 0.5:
            return imgs
        
        # 随机亮度和对比度调整
        brightness = 0.8 + 0.4 * torch.rand(batch_size, 1, 1, 1, device=device)
        contrast = 0.8 + 0.4 * torch.rand(batch_size, 1, 1, 1, device=device)
        
        imgs = imgs * brightness
        imgs = (imgs - 0.5) * contrast + 0.5
        imgs = torch.clamp(imgs, 0, 1)
        
        return imgs
