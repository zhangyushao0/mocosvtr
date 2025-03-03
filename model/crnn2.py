import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch
from mmengine.model import BaseModel


# 定义优化后的CRNN网络结构
class CRNNBackbone(nn.Module):
    def __init__(self, num_chars, input_channels=3):
        super(CRNNBackbone, self).__init__()

        # 使用多分支CNN特征提取，更适合缺损文本
        self.cnn = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 多尺度特征提取模块
            MultiScaleFeatureExtractor(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 增强的残差连接块
            self._make_residual_block(128, 256),
            self._make_residual_block(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # 双重注意力残差块
            DualAttentionResidualBlock(256, 512),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # 最后的特征提取
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 综合注意力机制
            CBAM(512),
        )

        # 使用改进的RNN
        self.rnn = nn.Sequential(
            ResidualBidirectionalLSTM(512, 256, 256),
            ResidualBidirectionalLSTM(256, 256, 512)
        )

        # 字符重建模块
        self.char_reconstructor = CharacterReconstructor(512, 256, num_chars)

        # 全连接层
        self.dropout = nn.Dropout(0.2)  # 增加dropout提高泛化能力
        self.fc = nn.Linear(512, num_chars)

    def _make_residual_block(self, in_channels, out_channels):
        return EnhancedResidualBlock(in_channels, out_channels)

    def forward(self, x):
        # CNN特征提取
        conv = self.cnn(x)

        # 转换为RNN输入格式
        batch, c, h, w = conv.size()
        conv = conv.view(batch, c, -1)
        conv = conv.permute(0, 2, 1)  # [batch, width, channels]

        # RNN序列建模
        rnn_output = self.rnn(conv)
        
        # 应用字符重建
        reconstructed = self.char_reconstructor(rnn_output)
        
        # 结合重建特征和RNN特征
        combined = rnn_output + 0.2 * reconstructed
        
        # 添加dropout
        combined = self.dropout(combined)
        
        # 通过全连接层预测字符
        output = self.fc(combined)

        # 对数Softmax
        log_probs = F.log_softmax(output, dim=2)

        # 转置为CTC要求的格式 [T, N, C]
        log_probs = log_probs.permute(1, 0, 2)

        return log_probs


# 多尺度特征提取模块 - 更好地处理缺损字符
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # 添加边缘检测分支 - 对缺损文本特别有效
        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # 使用Sobel算子提取边缘
        self.sobel_x = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        
        # 初始化Sobel核
        with torch.no_grad():
            self.sobel_x.weight[:, :, :, :] = 0
            self.sobel_y.weight[:, :, :, :] = 0
            sobel_kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            for i in range(in_channels):
                self.sobel_x.weight[0, i, :, :] = sobel_kernel_x
                self.sobel_y.weight[0, i, :, :] = sobel_kernel_y
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        
        # 边缘检测
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        edge_features = self.edge_branch(x) * edge_mag
        
        # 特征融合
        merged = torch.cat([branch1, branch2, branch3, edge_features], dim=1)
        return self.fusion(merged)


# 增强的残差块
class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 可分离卷积分支 - 更好地处理噪声和缺损
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 如果输入输出通道不一致，添加1x1卷积进行调整
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
            
        # 通道注意力
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x):
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 可分离卷积分支
        alt_path = self.depthwise(x if self.downsample is None else self.downsample(x))
        alt_path = self.pointwise(alt_path)
        alt_path = self.bn3(alt_path)
        alt_path = self.relu(alt_path)
        
        # 融合两个路径
        out = out + 0.3 * alt_path
        
        # 应用通道注意力
        out = self.channel_attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 双重注意力残差块
class DualAttentionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualAttentionResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 注意力机制
        self.cbam = CBAM(out_channels)
        
        # 下采样
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv_block(x)
        out = self.cbam(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention


# 卷积块注意力模块 (CBAM)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# 带残差连接的双向LSTM
class ResidualBidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ResidualBidirectionalLSTM, self).__init__()
        
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
        # 输入和输出维度不同时的投影层
        self.projection = None
        if input_size != output_size:
            self.projection = nn.Linear(input_size, output_size)
            
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, x):
        residual = x
        
        # LSTM处理
        output, _ = self.rnn(x)
        output = self.fc(output)
        
        # 残差连接
        if self.projection is not None:
            residual = self.projection(residual)
            
        output = output + residual
        
        # 层归一化
        output = self.layer_norm(output)
        
        return output


# 字符重建模块 - 专为缺损场景设计
class CharacterReconstructor(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super(CharacterReconstructor, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.char_context = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size)
        )
        
    def forward(self, x):
        # 计算时序注意力权重
        attn_weights = self.attention(x)
        
        # 应用注意力加权
        context = torch.sum(x * attn_weights, dim=1, keepdim=True)
        
        # 生成字符上下文特征
        char_context = self.char_context(context)
        
        # 扩展到与输入相同的时序长度
        batch_size, seq_len, _ = x.size()
        char_context = char_context.expand(-1, seq_len, -1)
        
        return char_context


# 基于MMEngine的优化后的CRNN模型
class MMCRNN(BaseModel):
    def __init__(self, num_chars):
        super(MMCRNN, self).__init__()
        self.backbone = CRNNBackbone(num_chars)

        # CTC损失函数
        self.loss_fn = nn.CTCLoss(blank=0, reduction="mean")
        # 添加梯度裁剪参数
        self.max_grad_norm = 5.0
        
        # 使用标签平滑
        self.label_smoothing = 0.1

    def forward(self, imgs, labels=None, target_lengths=None, texts=None, mode="tensor"):
        if mode == "loss":
            # 应用增强的数据增强
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
                print(f"警告: 输入序列长度({input_lengths[0]})小于目标序列长度({target_lengths[0]})")
                # 调整输入长度以匹配目标长度
                input_lengths = torch.maximum(input_lengths, target_lengths)
                valid_batch = True
                
            if valid_batch:
                # 应用CTC损失
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
        """专门针对缺损文本的数据增强"""
        batch_size = imgs.shape[0]
        device = imgs.device
        
        # 30%的概率不应用增强
        if torch.rand(1).item() < 0.3:
            return imgs
        
        # 创建增强后的图像副本
        augmented_imgs = imgs.clone()
        
        # 1. 随机亮度和对比度调整
        brightness = 0.7 + 0.6 * torch.rand(batch_size, 1, 1, 1, device=device)
        contrast = 0.7 + 0.6 * torch.rand(batch_size, 1, 1, 1, device=device)
        
        augmented_imgs = augmented_imgs * brightness
        augmented_imgs = (augmented_imgs - 0.5) * contrast + 0.5
        
        # # 2. 模拟缺损 (随机擦除)
        # for i in range(batch_size):
        #     # 50%的概率应用缺损模拟
        #     if torch.rand(1).item() < 0.5:
        #         h, w = imgs.shape[2], imgs.shape[3]
                
        #         # 创建1-5个随机擦除区域
        #         num_erases = torch.randint(1, 6, (1,)).item()
        #         for _ in range(num_erases):
        #             # 生成随机长方形区域
        #             erase_w = int(w * torch.rand(1).item() * 0.3)  # 最多擦除30%宽度
        #             erase_h = int(h * torch.rand(1).item() * 0.2)  # 最多擦除20%高度
                    
        #             # 随机位置
        #             x1 = int((w - erase_w) * torch.rand(1).item())
        #             y1 = int((h - erase_h) * torch.rand(1).item())
                    
        #             # 应用擦除 (设置为随机值或者背景色)
        #             if torch.rand(1).item() < 0.5:
        #                 # 随机杂点
        #                 noise = torch.rand(3, erase_h, erase_w, device=device)
        #                 augmented_imgs[i, :, y1:y1+erase_h, x1:x1+erase_w] = noise
        #             else:
        #                 # 背景色 (假设为白色或黑色)
        #                 value = torch.tensor(1.0 if torch.rand(1).item() < 0.5 else 0.0, device=device)
        #                 augmented_imgs[i, :, y1:y1+erase_h, x1:x1+erase_w] = value
        
        # 3. 模拟模糊 (20%概率) - 修复版本
        if torch.rand(1).item() < 0.2:
            # 使用内置的高斯模糊函数
            blur_radius = torch.randint(1, 3, (1,)).item() * 2 + 1  # 生成3或5
            sigma = 0.8 + 0.7 * torch.rand(1).item()  # 0.8-1.5之间的sigma
            
            # 逐批次应用模糊，避免维度问题
            for i in range(batch_size):
                img = augmented_imgs[i:i+1]
                # 使用torchvision的高斯模糊或者用自定义实现
                try:
                    # 尝试使用torchvision
                    import torchvision.transforms.functional as TF
                    img_blurred = TF.gaussian_blur(img, blur_radius, sigma)
                    augmented_imgs[i:i+1] = img_blurred
                except (ImportError, AttributeError):
                    # 回退到简单的均值模糊
                    padding = blur_radius // 2
                    unfold = F.unfold(img, kernel_size=blur_radius, padding=padding)
                    unfold = unfold.mean(dim=1, keepdim=True)
                    augmented_imgs[i:i+1] = F.fold(
                        unfold, 
                        output_size=img.shape[2:], 
                        kernel_size=1
                    )
        
        # 4. 添加噪声 (30%概率)
        if torch.rand(1).item() < 0.3:
            noise_level = 0.05 + 0.1 * torch.rand(1).item()
            noise = noise_level * torch.randn_like(augmented_imgs, device=device)
            augmented_imgs = augmented_imgs + noise
        
        # 确保像素值在有效范围内
        augmented_imgs = torch.clamp(augmented_imgs, 0, 1)
        
        return augmented_imgs