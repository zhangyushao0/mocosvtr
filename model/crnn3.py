import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch
from mmengine.model import BaseModel


# 改进的CRNN网络结构
class CRNNBackbone(nn.Module):
    def __init__(self, num_chars, input_channels=3):
        super(CRNNBackbone, self).__init__()


        # 增强的CNN特征提取网络
        self.cnn_stage1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self._make_residual_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.cnn_stage2 = nn.Sequential(
            self._make_residual_block(64, 128),
            self._make_residual_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.cnn_stage3 = nn.Sequential(
            self._make_residual_block(128, 256),
            self._make_residual_block(256, 256),
            self._make_residual_block(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        
        self.cnn_stage4 = nn.Sequential(
            self._make_residual_block(256, 512),
            self._make_residual_block(512, 512),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 特征金字塔网络 (FPN)
        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 512)
        
        # 注意力机制: 结合空间和通道注意力
        self.attention = CBAM(512)

        # 高级双向LSTM，增加层数和隐藏单元
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, 256),
            BidirectionalLSTM(256, 256, 512)
        )

        # 全连接层
        self.fc = nn.Linear(512, num_chars)

    def _make_residual_block(self, in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        
        # 分阶段特征提取
        c1 = self.cnn_stage1(x)
        c2 = self.cnn_stage2(c1)
        c3 = self.cnn_stage3(c2)
        c4 = self.cnn_stage4(c3)
        
        # 特征金字塔处理多尺度特征
        features = self.fpn([c1, c2, c3, c4])
        
        # 应用注意力机制
        conv = self.attention(features)

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



# 特征金字塔网络，用于多尺度特征融合
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        
        # 用于调整各级特征通道数的横向连接
        self.lateral_convs = nn.ModuleList()
        # 用于平滑特征的卷积
        self.smooth_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.smooth_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
            
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        
        # 添加用于融合特征的1x1卷积
        self.fusion_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1)
    
    def forward(self, xs):
        # 自顶向下路径，将高层特征上采样并添加到低层特征
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, xs)]
        
        # 处理最后一个特征图直接作为输出
        prev_features = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            # 上采样前一层特征
            upsample = F.interpolate(prev_features, 
                                  size=laterals[i].shape[-2:],
                                  mode='nearest')
            # 特征融合
            laterals[i] = laterals[i] + upsample
            prev_features = laterals[i]
        
        # 平滑处理
        results = [smooth_conv(lateral) for smooth_conv, lateral in zip(self.smooth_convs, laterals)]
        
        # 获取目标尺寸（使用最后一层特征图的尺寸）
        target_height, target_width = results[-1].shape[-2:]
        
        # 调整所有特征到相同大小
        aligned_results = []
        for i, result in enumerate(results):
            # 使用自适应池化确保所有特征图具有相同的空间尺寸
            aligned_feature = F.adaptive_avg_pool2d(result, (target_height, target_width))
            aligned_results.append(aligned_feature)
        
        # 融合所有特征 - 在通道维度上连接
        concat_features = torch.cat(aligned_results, dim=1)
        
        # 使用1x1卷积将通道数降回out_channels
        output = self.fusion_conv(concat_features)
        
        return output


# 改进的残差块，使用瓶颈结构减少参数
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # 瓶颈设计，减少计算量
        bottleneck_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

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
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# CBAM (Convolutional Block Attention Module) 结合通道和空间注意力
class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# 通道注意力机制，专注于重要特征通道
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch, c, _, _ = x.size()
        
        # 平均池化
        avg_out = self.avg_pool(x).view(batch, c)
        avg_out = self.fc(avg_out).view(batch, c, 1, 1)
        
        # 最大池化
        max_out = self.max_pool(x).view(batch, c)
        max_out = self.fc(max_out).view(batch, c, 1, 1)
        
        # 融合注意力
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention


# 空间注意力机制，改进版本
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
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


# 改进双向LSTM，支持残差连接
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
        # 添加残差连接
        self.has_residual = (input_size == output_size)
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x if self.has_residual else 0
        
        self.rnn.flatten_parameters()  # 优化RNN内存使用
        output, _ = self.rnn(x)
        output = self.fc(output)
        
        if self.has_residual:
            output = output + residual
            
        output = self.dropout(output)
        return output


# Focal CTC Loss，增强对难例的学习
class FocalCTCLoss(nn.Module):
    def __init__(self, blank=0, reduction='mean', gamma=2.0):
        super(FocalCTCLoss, self).__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction='none')
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # 计算标准CTC损失
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        
        # 计算focal损失权重
        pt = torch.exp(-loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用权重
        focal_loss = focal_weight * loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 基于MMEngine的改进CRNN模型
class MMCRNN(BaseModel):
    def __init__(self, num_chars):
        super(MMCRNN, self).__init__()
        self.backbone = CRNNBackbone(num_chars)

        # 使用Focal CTC损失函数
        self.loss_fn = nn.CTCLoss(blank=0, reduction="mean")


    def forward(self, imgs, labels=None, target_lengths=None, texts=None, mode="tensor"):
        if mode == "loss":
            # 增强的数据增强，特别针对缺损文本
            imgs = self.apply_degradation_augmentation(imgs)
        
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

    def apply_degradation_augmentation(self, imgs):
        """增强的数据增强，针对缺损文本场景"""
        batch_size = imgs.shape[0]
        device = imgs.device
        
        # 不是所有图像都应用增强
        if torch.rand(1).item() < 0.3:
            return imgs
        
        # 模拟退化和缺损
        augmented_imgs = imgs.clone()
        
        # 随机选择增强类型，可以应用多种
        aug_probs = torch.rand(6, device=device)
        
        # 1. 随机亮度和对比度调整
        if aug_probs[0] < 0.5:
            brightness = 0.6 + 0.8 * torch.rand(batch_size, 1, 1, 1, device=device)
            contrast = 0.6 + 0.8 * torch.rand(batch_size, 1, 1, 1, device=device)
            augmented_imgs = augmented_imgs * brightness
            augmented_imgs = (augmented_imgs - 0.5) * contrast + 0.5
        
        # 2. 模拟噪声
        if aug_probs[1] < 0.4:
            noise = 0.1 * torch.randn_like(augmented_imgs)
            augmented_imgs = augmented_imgs + noise
        
        # 3. 模拟模糊
        if aug_probs[2] < 0.4:
            # 简单的高斯模糊模拟
            kernel_size = 3
            padding = kernel_size // 2
            augmented_imgs = F.avg_pool2d(augmented_imgs, kernel_size=kernel_size, 
                                          stride=1, padding=padding)
        
        # 4. 模拟遮挡
        if aug_probs[3] < 0.3:
            for i in range(batch_size):
                # 随机遮挡块
                h, w = augmented_imgs.shape[2:]
                occlusion_h = torch.randint(5, h//4, (1,)).item()
                occlusion_w = torch.randint(5, w//4, (1,)).item()
                
                # 随机位置
                y = torch.randint(0, h - occlusion_h, (1,)).item()
                x = torch.randint(0, w - occlusion_w, (1,)).item()
                
                # 应用遮挡
                augmented_imgs[i, :, y:y+occlusion_h, x:x+occlusion_w] = 0
        
        # 5. 模拟条纹
        if aug_probs[4] < 0.2:
            for i in range(batch_size):
                # 随机条纹
                h, w = augmented_imgs.shape[2:]
                stripe_width = torch.randint(1, 3, (1,)).item()
                num_stripes = torch.randint(1, 5, (1,)).item()
                
                for _ in range(num_stripes):
                    direction = torch.rand(1).item()
                    if direction > 0.5:  # 水平条纹
                        y = torch.randint(0, h, (1,)).item()
                        augmented_imgs[i, :, y:y+stripe_width, :] = 0
                    else:  # 垂直条纹
                        x = torch.randint(0, w, (1,)).item()
                        augmented_imgs[i, :, :, x:x+stripe_width] = 0
        
        # 6. 模拟透视变换
        if aug_probs[5] < 0.3:
            # 透视变换参数，轻微扭曲
            angle = 5 * (torch.rand(batch_size, device=device) * 2 - 1)  # ±5度
            
            for i in range(batch_size):
                # 旋转矩阵
                angle_rad = angle[i] * torch.pi / 180
                cos_val = torch.cos(angle_rad)
                sin_val = torch.sin(angle_rad)
                
                # 构建仿射变换矩阵
                theta = torch.tensor([[cos_val, -sin_val, 0],
                                     [sin_val, cos_val, 0]], device=device)
                
                theta = theta.unsqueeze(0)
                
                # 应用变换
                grid = F.affine_grid(theta, augmented_imgs[i:i+1].size(), align_corners=False)
                augmented_imgs[i:i+1] = F.grid_sample(augmented_imgs[i:i+1], grid, align_corners=False)

        # 确保像素值在有效范围内
        augmented_imgs = torch.clamp(augmented_imgs, 0, 1)
        
        return augmented_imgs