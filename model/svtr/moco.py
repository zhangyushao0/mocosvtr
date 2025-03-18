from .tps_preprocessor import STN
from .svtr_encoder import SVTREncoder
from mmengine.model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision import transforms


class TextDegradationTransform:
    """文本图像退化增强类
    
    为文本识别任务提供专门的数据增强，模拟各种文本缺损情况
    """
    def __init__(
        self,
        p_occlusion=0.3,       # 应用遮挡的概率
        p_noise=0.3,           # 应用噪声的概率
        p_blur=0.2,            # 应用模糊的概率
        p_dropout=0.3,         # 应用像素丢失的概率
        p_elastic=0.2,         # 应用弹性变换的概率
        occlusion_size=(0.05, 0.15), # 遮挡块大小范围(相对于图像大小)
        noise_level=(5, 20),   # 噪声强度范围
        blur_radius=(0.5, 1.5),  # 高斯模糊半径范围
        dropout_ratio=(0.05, 0.15), # 像素丢失比例范围
        elastic_alpha=(0.5, 1.5),  # 弹性变换alpha参数范围
        elastic_sigma=(0.05, 0.15), # 弹性变换sigma参数范围
    ):
        self.p_occlusion = p_occlusion
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_dropout = p_dropout
        self.p_elastic = p_elastic
        self.occlusion_size = occlusion_size
        self.noise_level = noise_level
        self.blur_radius = blur_radius
        self.dropout_ratio = dropout_ratio
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        
        # 高斯模糊变换
        self.blur_transform = transforms.GaussianBlur(
            kernel_size=5, 
            sigma=blur_radius
        )
    
    def apply_occlusion(self, img):
        """随机遮挡图像的一部分，模拟文字被遮挡的情况"""
        # 确保图像是4D张量 [B, C, H, W]
        is_batched = len(img.shape) == 4
        if not is_batched:
            img = img.unsqueeze(0)
        
        b, c, h, w = img.shape
        
        # 对每个图像单独处理
        for i in range(b):
            if random.random() < self.p_occlusion:
                # 随机确定遮挡块的大小
                block_h = int(random.uniform(*self.occlusion_size) * h)
                block_w = int(random.uniform(*self.occlusion_size) * w)
                
                # 随机确定遮挡块的位置
                top = random.randint(0, h - block_h)
                left = random.randint(0, w - block_w)
                
                # 应用遮挡（设置为随机值或固定值，这里用白色）
                img[i, :, top:top+block_h, left:left+block_w] = 1.0
        
        return img if is_batched else img.squeeze(0)
    
    def apply_noise(self, img):
        """添加随机噪声，模拟扫描文档或低质量图像"""
        # 确保图像是4D张量
        is_batched = len(img.shape) == 4
        if not is_batched:
            img = img.unsqueeze(0)
            
        b, c, h, w = img.shape
        
        # 对每个图像单独处理
        for i in range(b):
            if random.random() < self.p_noise:
                # 生成随机噪声
                noise_level = random.uniform(*self.noise_level) / 255.0
                noise = torch.randn_like(img[i]) * noise_level
                
                # 添加噪声并裁剪到合理范围
                img[i] = torch.clamp(img[i] + noise, 0, 1)
        
        return img if is_batched else img.squeeze(0)
    
    def apply_blur(self, img):
        """应用高斯模糊，模拟失焦或移动模糊"""
        # 确保图像是4D张量
        is_batched = len(img.shape) == 4
        if not is_batched:
            img = img.unsqueeze(0)
            
        b, c, h, w = img.shape
        
        # 对每个图像单独处理
        for i in range(b):
            if random.random() < self.p_blur:
                # 应用高斯模糊
                img[i] = self.blur_transform(img[i])
        
        return img if is_batched else img.squeeze(0)
    
    def apply_dropout(self, img):
        """随机丢弃像素，模拟文字缺失或墨水褪色"""
        # 确保图像是4D张量
        is_batched = len(img.shape) == 4
        if not is_batched:
            img = img.unsqueeze(0)
            
        b, c, h, w = img.shape
        
        # 对每个图像单独处理
        for i in range(b):
            if random.random() < self.p_dropout:
                # 确定丢弃像素的比例
                ratio = random.uniform(*self.dropout_ratio)
                
                # 生成随机掩码
                mask = torch.FloatTensor(c, h, w).uniform_() > ratio
                mask = mask.to(img.device)
                
                # 应用掩码
                img[i] = img[i] * mask 
        
        return img if is_batched else img.squeeze(0)
    
    def elastic_transform(self, img):
        """弹性变换，模拟纸张变形或字体变形"""
        # 目前仅支持CPU上的实现
        if img.is_cuda:
            return img
            
        # 确保图像是4D张量
        is_batched = len(img.shape) == 4
        if not is_batched:
            img = img.unsqueeze(0)
            
        b, c, h, w = img.shape
        result = img.clone()
        
        for i in range(b):
            if random.random() < self.p_elastic:
                # 随机确定变换参数
                alpha = random.uniform(*self.elastic_alpha)
                sigma = random.uniform(*self.elastic_sigma) * min(h, w)
                
                # 生成随机位移场
                dx = np.random.rand(h, w) * 2 - 1
                dy = np.random.rand(h, w) * 2 - 1
                
                # 应用高斯滤波使位移场更平滑
                from scipy.ndimage import gaussian_filter
                dx = gaussian_filter(dx, sigma) * alpha * w / 20
                dy = gaussian_filter(dy, sigma) * alpha * h / 20
                
                # 生成网格
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                
                # 应用位移
                indices_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
                indices_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
                
                # 转换为PyTorch网格采样格式
                grid_x = torch.from_numpy(indices_x / (w - 1) * 2 - 1)
                grid_y = torch.from_numpy(indices_y / (h - 1) * 2 - 1)
                grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0)
                
                # 执行网格采样
                for j in range(c):
                    result[i, j] = F.grid_sample(
                        img[i, j].unsqueeze(0).unsqueeze(0),
                        grid,
                        mode='bilinear',
                        padding_mode='border',
                        align_corners=True
                    ).squeeze()
        
        return result if is_batched else result.squeeze(0)
    
    def __call__(self, img):
        """应用随机文本退化增强"""
        # 随机应用各种退化变换
        img = self.apply_occlusion(img)
        img = self.apply_noise(img)
        img = self.apply_blur(img)
        img = self.apply_dropout(img)
        
        # 弹性变换相对较重，仅在CPU上实现
        if not img.is_cuda:
            img = self.elastic_transform(img)
            
        return img


class Backbone(nn.Module):
    """骨干网络，包含预处理器和编码器"""
    
    def __init__(self, preprocessor=None, encoder=None):
        super(Backbone, self).__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        
    def forward(self, img):
        if self.preprocessor is not None:
            img = self.preprocessor(img)
        return self.encoder(img)


class MoCoSVTR(BaseModel):
    """MoCoSVTR文本识别模型 (Scene Text Recognition with a Single Visual Model)

    Args:
        preprocessor: 预处理模块，如STN空间变换网络
        encoder: 特征提取编码器
        dictionary: 字典
        data_preprocessor: 数据预处理器
        dim: 投影空间维度
        K: 队列大小
        m: 动量更新系数
        T: 温度系数
        use_augmentation: 是否使用文本退化增强
    """

    def __init__(
        self,
        preprocessor=None,
        encoder=None,
        dictionary=None,
        data_preprocessor=None,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        use_augmentation=True,  # 是否使用文本退化增强
        init_cfg=None,
    ):
        super(MoCoSVTR, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # 创建backbone，包含preprocessor和encoder
        self.backbone_q = Backbone(preprocessor=preprocessor, encoder=encoder)
        self.dictionary = dictionary

        # MoCo 相关参数
        self.K = K  # 队列大小
        self.m = m  # 动量更新系数
        self.T = T  # 温度系数
        
        # 记录当前epoch和迭代次数
        self.current_epoch = 0
        self.current_iter = 0

        # 创建key编码器（动量编码器）- 与查询编码器相同架构
        self.backbone_k = self._build_momentum_encoder()

        # 冻结动量编码器的参数
        for param_k in self.backbone_k.parameters():
            param_k.requires_grad = False

        # 创建投影头（将特征投影到低维空间）
        feat_dim = self.backbone_q.encoder.out_channels  # 获取编码器输出维度
        self.projector_q = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, dim),
        )
        self.projector_k = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, dim),
        )

        # 初始化参数
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 初始化队列和指针
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)  # 归一化队列
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # 新增：初始化索引队列，用于记录每个特征对应的图像索引
        self.register_buffer("index_queue", torch.full((K,), -1, dtype=torch.long))
        
        # 初始化数据增强
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.text_augmenter = TextDegradationTransform()

    def _build_momentum_encoder(self):
        """构建与查询编码器相同的动量编码器"""
        # 直接创建一个与backbone结构完全相同的副本
        import copy
        backbone_k = copy.deepcopy(self.backbone_q)
        
        # 冻结参数
        for param_k in backbone_k.parameters():
            param_k.requires_grad = False

        return backbone_k

    def extract_feat(self, img, encoder_type="q"):
        """提取特征

        Args:
            img: 输入图像
            encoder_type: 使用的编码器类型，'q'表示查询编码器，'k'表示键编码器
        """
        if encoder_type == "q":
            return self.backbone_q(img)
        else:
            return self.backbone_k(img)

    def apply_augmentation(self, img):
        """应用文本退化增强

        Args:
            img: 输入图像 [B, C, H, W]

        Returns:
            aug_img: 增强后的图像
        """
        if self.use_augmentation and self.training:
            return self.text_augmenter(img)
        return img

    def forward(self, imgs_q=None, imgs_k=None, idxs=None, mode="tensor"):
        """前向传播函数，符合BaseModel的接口约定

        Args:
            imgs_q: 查询图像
            imgs_k: 键图像
            idxs: 图像索引
            mode: 运行模式: 'tensor', 'loss', 或 'predict'

        Returns:
            根据mode返回相应结果:
            - tensor: 返回特征张量
            - loss: 返回损失字典
            - predict: 返回预测结果
        """
        if mode == "loss":
            # 对查询图像应用增强
            if self.training and self.use_augmentation:
                imgs_q = self.apply_augmentation(imgs_q)
                
                # 对于键图像，我们应用不同的随机增强以提高多样性
                imgs_k = self.apply_augmentation(imgs_k)
            
            # 提取查询特征
            feats_q = self.extract_feat(imgs_q, encoder_type="q")  # [B, C, H, W]

            # 对特征进行全局平均池化
            b, c, h, w = feats_q.shape
            feats_q = feats_q.view(b, c, -1).mean(dim=2)  # [B, C]

            # 投影到对比学习空间
            q = self.projector_q(feats_q)  # [B, dim]
            q = F.normalize(q, dim=1)  # 归一化

            # 使用动量编码器提取键特征
            with torch.no_grad():
                # 动量更新键编码器
                self._momentum_update_key_encoder()

                # 提取键特征
                feats_k = self.extract_feat(imgs_k, encoder_type="k")  # [B, C, H, W]

                # 对特征进行全局平均池化
                feats_k = feats_k.view(b, c, -1).mean(dim=2)  # [B, C]

                # 投影到对比学习空间
                k = self.projector_k(feats_k)  # [B, dim]
                k = F.normalize(k, dim=1)  # 归一化

            # 计算对比损失，传入索引以排除相同样本
            loss = self._get_contrastive_loss(q, k, idxs)

            # 更新队列，同时更新索引队列
            self._dequeue_and_enqueue(k, idxs)

            # 返回损失字典，符合BaseModel接口约定
            return {"loss": loss}

        elif mode == "predict":
            # 对于预训练模型的评估，我们只需要查询编码器的特征
            feats = self.extract_feat(imgs_q, encoder_type="q")

            # 对特征进行全局平均池化
            b, c, h, w = feats.shape
            feats = feats.view(b, c, -1).mean(dim=2)  # [B, C]

            # 投影到对比学习空间
            q = self.projector_q(feats)  # [B, dim]
            q = F.normalize(q, dim=1)  # 归一化

            # 返回预测结果，用于评估
            return q

        elif mode == "tensor":
            # 返回特征张量
            return self.extract_feat(imgs_q)

        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新键编码器"""
        # 更新backbone
        for param_q, param_k in zip(
            self.backbone_q.parameters(), self.backbone_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        # 更新投影头
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, idxs=None):
        """更新队列及其对应的索引

        Args:
            keys: 特征向量
            idxs: 图像索引
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 替换队列的旧数据
        if ptr + batch_size <= self.K:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            if idxs is not None:
                self.index_queue[ptr : ptr + batch_size] = idxs
        else:
            # 处理队列循环
            remain = self.K - ptr
            self.queue[:, ptr:] = keys[:remain].T
            self.queue[:, : batch_size - remain] = keys[remain:].T
            
            if idxs is not None:
                self.index_queue[ptr:] = idxs[:remain]
                self.index_queue[: batch_size - remain] = idxs[remain:]

        # 更新指针
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def _get_contrastive_loss(self, q, k, idxs=None):
        """计算对比损失，排除队列中与当前批次相同索引的样本

        Args:
            q: 查询特征 [N, C]
            k: 键特征 [N, C]
            idxs: 当前批次图像的索引 [N]

        Returns:
            contrastive_loss: 对比损失
        """
        # 计算正样本相似度
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # [N, 1]

        # 计算负样本相似度
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])  # [N, K]
        
        # 如果提供了索引，则创建掩码排除队列中与当前批次相同索引的样本
        if idxs is not None:
            # 为每个查询创建掩码
            queue_idxs = self.index_queue.clone().detach()  # [K]
            mask = torch.ones_like(l_neg, dtype=torch.bool)  # 初始化为全1掩码
            
            # 对于每个样本，标记队列中所有相同索引的位置
            for i, idx in enumerate(idxs):
                # 找出队列中与当前样本索引相同的位置
                same_idx = (queue_idxs == idx)
                if same_idx.sum() > 0:  # 如果存在相同索引
                    mask[i, same_idx] = False  # 将这些位置标记为需要排除
            
            # 将要排除的位置设为一个非常小的值（等效于负无穷）
            l_neg = torch.where(mask, l_neg, torch.tensor(-1e9, device=l_neg.device))

        # 合并logits
        logits = torch.cat([l_pos, l_neg], dim=1)  # [N, 1+K]

        # 除以温度系数
        logits /= self.T

        # 对比损失，第0个位置为正样本
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def before_train_epoch(self, runner):
        """每个训练epoch开始前的回调函数，用于准备队列预热

        Args:
            runner: 训练运行器
        """
        # 记录当前epoch
        self.current_epoch = runner.epoch
        # 重置迭代计数器
        self.current_iter = 0

    def train_step(self, data, optim_wrapper):
        """训练步骤

        Args:
            data: 数据批次，应包含查询图像和键图像
            optim_wrapper: 优化器包装器

        Returns:
            log_vars: 损失日志字典
        """
        # 更新迭代计数
        self.current_iter += 1
        
        # 使用数据预处理器处理数据
        data = self.data_preprocessor(data, training=True)

        # 解包数据，预期是包含查询图像和键图像的元组或列表
        if isinstance(data, (tuple, list)) and len(data) >= 3:
            imgs_q, imgs_k, idxs = data[:3]
        else:
            # 如果数据格式不符合预期，尝试使用字典格式访问
            if isinstance(data, dict) and "imgs_q" in data and "imgs_k" and "idxs" in data:
                imgs_q = data["imgs_q"]
                imgs_k = data["imgs_k"]
                idxs = data["idxs"]
            else:
                raise ValueError("数据格式不正确，需要包含查询图像和键图像")

        # 前向传播计算损失
        losses = self(imgs_q=imgs_q, imgs_k=imgs_k, idxs=idxs, mode="loss")
        
        # 解析损失
        parsed_losses, log_vars = self.parse_losses(losses)
        
        # 更新参数
        optim_wrapper.update_params(parsed_losses)
        
        return log_vars

    def val_step(self, data, optim_wrapper=None):
        """验证步骤

        Args:
            data: 数据批次
            optim_wrapper: 优化器包装器(不会使用)

        Returns:
            outputs: 模型输出
        """
        # 使用数据预处理器处理数据
        data = self.data_preprocessor(data, training=False)

        # 解包数据，只需要查询图像
        if isinstance(data, (tuple, list)) and len(data) >= 1:
            imgs_q = data[0]
        else:
            # 如果数据格式不符合预期，尝试使用字典格式访问
            if isinstance(data, dict) and "imgs" in data:
                imgs_q = data["imgs"]
            else:
                raise ValueError("数据格式不正确")

        # 前向传播计算预测
        outputs = self(imgs_q=imgs_q, mode="predict")

        return outputs

    def test_step(self, data, optim_wrapper=None):
        """测试步骤，与验证步骤相同"""
        return self.val_step(data, optim_wrapper)


pretrained = "work_dir/epoch_20.pth"

MoCoSVTRModel = MoCoSVTR(    
    preprocessor=STN(
        in_channels=3, resized_image_size=(48, 192), output_image_size=(48, 192)
    ),
    encoder=SVTREncoder(
        img_size=[48, 192],
        in_channels=3,
        out_channels=192,
        embed_dims=[64, 128, 256],
        depth=[3, 6, 3],
        num_heads=[2, 4, 8],
        mixer_types=["Local"] * 6 + ["Global"] * 6,
        window_size=[[7, 11], [7, 11], [7, 11]],
        merging_types="Conv",
        prenorm=False,
        max_seq_len=25,
    ),
    dim=256,
    use_augmentation=True,     # 启用文本退化增强
    init_cfg=dict(type='Pretrained', checkpoint=pretrained),
)