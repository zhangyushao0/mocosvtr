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
    def __init__(
        self,
        p_occlusion=0.3,
        p_noise=0.3,
        p_blur=0.2,
        p_dropout=0.3,
        p_elastic=0.2,
        p_contrast=0.3,
        p_sharpen=0.3,
        p_perspective=0.3,
    ):
        self.p_occlusion = p_occlusion
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_dropout = p_dropout
        self.p_elastic = p_elastic
        self.p_contrast = p_contrast
        self.p_sharpen = p_sharpen
        self.p_perspective = p_perspective
        self.blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.0))

    def apply_occlusion(self, img):
        is_batched = len(img.shape) == 4
        if not is_batched:
            img = img.unsqueeze(0)
        b, c, h, w = img.shape
        for i in range(b):
            if random.random() < self.p_occlusion:
                block_h = int(random.uniform(0.05, 0.15) * h)
                block_w = int(random.uniform(0.05, 0.15) * w)
                top = random.randint(0, h - block_h)
                left = random.randint(0, w - block_w)
                img[i, :, top : top + block_h, left : left + block_w] = 1.0
        return img if is_batched else img.squeeze(0)

    def apply_noise(self, img):
        is_batched = len(img.shape) == 4
        if not is_batched:
            img = img.unsqueeze(0)
        b, c, h, w = img.shape
        for i in range(b):
            if random.random() < self.p_noise:
                noise_level = random.uniform(5, 20) / 255.0
                noise = torch.randn_like(img[i]) * noise_level
                img[i] = torch.clamp(img[i] + noise, 0, 1)
        return img if is_batched else img.squeeze(0)

    def apply_blur(self, img):
        is_batched = len(img.shape) == 4
        if not is_batched:
            img = img.unsqueeze(0)
        b, c, h, w = img.shape
        for i in range(b):
            if random.random() < self.p_blur:
                img[i] = self.blur_transform(img[i])
        return img if is_batched else img.squeeze(0)

    def apply_dropout(self, img):
        is_batched = len(img.shape) == 4
        if not is_batched:
            img = img.unsqueeze(0)
        b, c, h, w = img.shape
        for i in range(b):
            if random.random() < self.p_dropout:
                ratio = random.uniform(0.05, 0.15)
                mask = torch.FloatTensor(c, h, w).uniform_() > ratio
                mask = mask.to(img.device)
                img[i] = img[i] * mask
        return img if is_batched else img.squeeze(0)

    def elastic_transform(self, img):
        if img.is_cuda:
            return img
        is_batched = len(img.shape) == 4
        if not is_batched:
            img = img.unsqueeze(0)
        b, c, h, w = img.shape
        result = img.clone()
        for i in range(b):
            if random.random() < self.p_elastic:
                alpha = random.uniform(0.5, 1.5)
                sigma = random.uniform(0.05, 0.15) * min(h, w)
                from scipy.ndimage import gaussian_filter

                dx = (
                    gaussian_filter(np.random.rand(h, w) * 2 - 1, sigma)
                    * alpha
                    * w
                    / 20
                )
                dy = (
                    gaussian_filter(np.random.rand(h, w) * 2 - 1, sigma)
                    * alpha
                    * h
                    / 20
                )
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                indices_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
                indices_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
                grid_x = torch.from_numpy(indices_x / (w - 1) * 2 - 1)
                grid_y = torch.from_numpy(indices_y / (h - 1) * 2 - 1)
                grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0)
                for j in range(c):
                    result[i, j] = F.grid_sample(
                        img[i, j].unsqueeze(0).unsqueeze(0),
                        grid,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=True,
                    ).squeeze()
        return result if is_batched else result.squeeze(0)

    def apply_contrast(self, img):
        if random.random() < self.p_contrast:
            alpha = random.uniform(0.5, 1.0)
            img = torch.clamp(0.5 + alpha * (img - 0.5), 0, 1)
        return img

    def apply_sharpen(self, img):
        if random.random() < self.p_sharpen:
            alpha = random.uniform(0.0, 0.5)
            lightness = random.uniform(0.0, 0.5)
            sharpened = torch.clamp(
                img * (1 + lightness) - img.mean() * lightness, 0, 1
            )
            img = torch.clamp(img * (1 - alpha) + sharpened * alpha, 0, 1)
        return img

    def apply_perspective(self, img):
        if random.random() < self.p_perspective:
            b, c, h, w = img.shape if len(img.shape) == 4 else (1, *img.shape)
            scale = random.uniform(0.01, 0.02)
            points1 = torch.tensor(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=torch.float32
            )
            points2 = points1 + torch.randn(4, 2) * scale * min(h, w)
            transform = transforms.RandomPerspective(distortion_scale=scale, p=1.0)
            img = transform(img)
        return img

    def apply_crop(self, img):
        if random.random() < 0.5:
            b, c, h, w = img.shape if len(img.shape) == 4 else (1, *img.shape)
            vert_crop = random.uniform(0.0, 0.4) * h
            horz_crop = random.uniform(0.0, 0.02) * w
            top = int(vert_crop // 2)
            left = int(horz_crop // 2)
            img = img[
                ..., top : h - int(vert_crop - top), left : w - int(horz_crop - left)
            ]
            img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
        return img

    def __call__(self, img):
        img = self.apply_contrast(img)
        img = self.apply_noise(img)
        img = self.apply_blur(img)
        img = self.apply_dropout(img)
        img = self.apply_sharpen(img)
        img = self.apply_perspective(img)
        img = self.apply_crop(img)
        if not img.is_cuda:
            img = self.elastic_transform(img)
        return img


class Backbone(nn.Module):
    def __init__(self, preprocessor=None, encoder=None):
        super(Backbone, self).__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(self, img):
        if self.preprocessor is not None:
            img = self.preprocessor(img)
        return self.encoder(img)


class SeqCLR(BaseModel):
    def __init__(
        self,
        preprocessor=None,
        encoder=None,
        dictionary=None,
        data_preprocessor=None,
        dim=128,
        num_blocks=5,
        instance_mapping_type="window-to-instance",
        T=0.07,
        projection_head_type="mlp",
        use_augmentation=True,
        init_cfg=None,
    ):
        super(SeqCLR, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg
        )
        self.backbone = Backbone(preprocessor=preprocessor, encoder=encoder)
        self.dictionary = dictionary
        self.num_blocks = num_blocks
        self.instance_mapping_type = instance_mapping_type
        self.T = T
        feat_dim = self.backbone.encoder.out_channels
        self.projection_head_type = projection_head_type

        if projection_head_type == "mlp":
            self.global_projector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, dim),
            )
            self.local_projector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, dim),
            )
        elif projection_head_type == "bilstm":
            self.global_projector = nn.LSTM(
                feat_dim, dim // 2, num_layers=1, bidirectional=True, batch_first=True
            )
            self.local_projector = nn.LSTM(
                feat_dim, dim // 2, num_layers=1, bidirectional=True, batch_first=True
            )
        else:
            raise ValueError(f"Unknown projection_head_type: {projection_head_type}")

        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.text_augmenter = TextDegradationTransform()

    def extract_feat(self, img):
        return self.backbone(img)

    def apply_augmentation(self, img):
        if self.use_augmentation and self.training:
            return self.text_augmenter(img)
        return img

    def _apply_instance_mapping(self, feats):
        b, c, h, w = feats.shape
        global_feats = feats.mean(dim=[2, 3])

        if self.instance_mapping_type == "all-to-instance":
            return [global_feats]
        elif self.instance_mapping_type == "frame-to-instance":
            local_feats = feats.view(b, c, -1).permute(0, 2, 1)
            return [local_feats[:, i] for i in range(local_feats.size(1))]
        elif self.instance_mapping_type == "window-to-instance":
            feats_seq = feats.view(b, c, -1)
            local_feats = F.adaptive_avg_pool1d(feats_seq, self.num_blocks)
            return [local_feats[:, :, i] for i in range(self.num_blocks)]
        else:
            raise ValueError(
                f"Unknown instance_mapping_type: {self.instance_mapping_type}"
            )

    def _apply_projection(self, feats, projector):
        if self.projection_head_type == "mlp":
            return F.normalize(projector(feats), dim=1)
        elif self.projection_head_type == "bilstm":
            feats_seq = feats.unsqueeze(1)
            output, _ = projector(feats_seq)
            return F.normalize(output.squeeze(1), dim=1)

    def _get_contrastive_loss(
        self, global_proj1, local_projs1, global_proj2, local_projs2
    ):
        b = global_proj1.shape[0]
        loss = 0.0

        for local_proj1 in local_projs1:
            pos_sim = torch.einsum("nc,nc->n", [global_proj1, local_proj1]).unsqueeze(
                -1
            )
            # 使用正确的 einsum 表达式处理 3D 张量
            stacked_local_projs2 = torch.stack(
                local_projs2, dim=1
            )  # [b, num_blocks, dim]
            neg_sim = torch.einsum("nc,nkc->nk", [global_proj1, stacked_local_projs2])
            logits = torch.cat([pos_sim, neg_sim], dim=1) / self.T
            labels = torch.zeros(b, dtype=torch.long, device=logits.device)
            loss += F.cross_entropy(logits, labels)

        for local_proj2 in local_projs2:
            pos_sim = torch.einsum("nc,nc->n", [global_proj2, local_proj2]).unsqueeze(
                -1
            )
            # 同样处理
            stacked_local_projs1 = torch.stack(
                local_projs1, dim=1
            )  # [b, num_blocks, dim]
            neg_sim = torch.einsum("nc,nkc->nk", [global_proj2, stacked_local_projs1])
            logits = torch.cat([pos_sim, neg_sim], dim=1) / self.T
            labels = torch.zeros(b, dtype=torch.long, device=logits.device)
            loss += F.cross_entropy(logits, labels)

        return loss / (len(local_projs1) + len(local_projs2))

    def forward(self, imgs=None, mode="tensor"):
        if mode == "loss":
            imgs_aug1 = self.apply_augmentation(imgs)
            imgs_aug2 = self.apply_augmentation(imgs)
            feats1 = self.extract_feat(imgs_aug1)
            feats2 = self.extract_feat(imgs_aug2)

            global_proj1 = self._apply_projection(
                feats1.mean(dim=[2, 3]), self.global_projector
            )
            local_projs1 = [
                self._apply_projection(feat, self.local_projector)
                for feat in self._apply_instance_mapping(feats1)
            ]
            global_proj2 = self._apply_projection(
                feats2.mean(dim=[2, 3]), self.global_projector
            )
            local_projs2 = [
                self._apply_projection(feat, self.local_projector)
                for feat in self._apply_instance_mapping(feats2)
            ]

            loss = self._get_contrastive_loss(
                global_proj1, local_projs1, global_proj2, local_projs2
            )
            return {"loss": loss}

        elif mode == "predict":
            feats = self.extract_feat(imgs)
            global_proj = self._apply_projection(
                feats.mean(dim=[2, 3]), self.global_projector
            )
            return global_proj

        elif mode == "tensor":
            return self.extract_feat(imgs)

        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def train_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data, training=True)
        if isinstance(data, (tuple, list)) and len(data) >= 1:
            imgs = data[0]
        elif isinstance(data, dict) and "imgs" in data:
            imgs = data["imgs"]
        else:
            raise ValueError("数据格式不正确，需要包含图像")
        losses = self(imgs=imgs, mode="loss")
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data, optim_wrapper=None):
        data = self.data_preprocessor(data, training=False)
        if isinstance(data, (tuple, list)) and len(data) >= 1:
            imgs = data[0]
        elif isinstance(data, dict) and "imgs" in data:
            imgs = data["imgs"]
        else:
            raise ValueError("数据格式不正确")
        outputs = self(imgs=imgs, mode="predict")
        return outputs

    def test_step(self, data, optim_wrapper=None):
        return self.val_step(data, optim_wrapper)


pretrained = "work_dir/epoch_20.pth"

SeqCLRModel = SeqCLR(
    preprocessor=STN(
        in_channels=3, resized_image_size=(32, 100), output_image_size=(32, 100)
    ),
    encoder=SVTREncoder(
        img_size=[32, 100],
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
    num_blocks=5,
    instance_mapping_type="window-to-instance",
    projection_head_type="mlp",
    use_augmentation=True,
    # init_cfg=dict(type="Pretrained", checkpoint=pretrained),
)
