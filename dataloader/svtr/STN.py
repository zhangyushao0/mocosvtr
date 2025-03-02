from mmengine.model import BaseModule

class STN(BaseModule):
    """Implement STN module in ASTER: An Attentional Scene Text Recognizer with
    Flexible Rectification
    (https://ieeexplore.ieee.org/abstract/document/8395027/)

    Args:
        in_channels (int): The number of input channels.
        resized_image_size (Tuple[int, int]): The resized image size. The input
            image will be downsampled to have a better recitified result.
        output_image_size: The size of the output image for TPS. Defaults to
            (32, 100).
        num_control_points: The number of control points. Defaults to 20.
        margins: The margins for control points to the top and down side of the
            image for TPS. Defaults to [0.05, 0.05].
    """

    def __init__(self,
                 in_channels: int,
                 resized_image_size: Tuple[int, int] = (32, 64),
                 output_image_size: Tuple[int, int] = (32, 100),
                 num_control_points: int = 20,
                 margins: Tuple[float, float] = [0.05, 0.05],
                 init_cfg: Optional[Union[Dict, List[Dict]]] = [
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d'),
                 ]):
        super().__init__(init_cfg=init_cfg)
        self.resized_image_size = resized_image_size
        self.num_control_points = num_control_points
        self.tps = TPStransform(output_image_size, num_control_points, margins)
        self.stn_convnet = nn.Sequential(
            ConvModule(in_channels, 32, 3, 1, 1, norm_cfg=dict(type='BN')),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(32, 64, 3, 1, 1, norm_cfg=dict(type='BN')),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(64, 128, 3, 1, 1, norm_cfg=dict(type='BN')),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(128, 256, 3, 1, 1, norm_cfg=dict(type='BN')),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(256, 256, 3, 1, 1, norm_cfg=dict(type='BN')),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(256, 256, 3, 1, 1, norm_cfg=dict(type='BN')),
        )

        self.stn_fc1 = nn.Sequential(
            nn.Linear(2 * 256, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.stn_fc2 = nn.Linear(512, num_control_points * 2)
        self.init_stn(self.stn_fc2)

    def init_stn(self, stn_fc2: nn.Linear) -> None:
        """Initialize the output linear layer of stn, so that the initial
        source point will be at the top and down side of the image, which will
        help to optimize.

        Args:
            stn_fc2 (nn.Linear): The output linear layer of stn.
        """
        margin = 0.01
        sampling_num_per_side = int(self.num_control_points / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom],
                                     axis=0).astype(np.float32)
        stn_fc2.weight.data.zero_()
        stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward function of STN.

        Args:
            img (Tensor): The input image tensor.

        Returns:
            Tensor: The rectified image tensor.
        """
        resize_img = F.interpolate(
            img, self.resized_image_size, mode='bilinear', align_corners=True)
        points = self.stn_convnet(resize_img)
        batch_size, _, _, _ = points.size()
        points = points.view(batch_size, -1)
        img_feat = self.stn_fc1(points)
        points = self.stn_fc2(0.1 * img_feat)
        points = points.view(-1, self.num_control_points, 2)

        transformd_image = self.tps(img, points)
        return transformd_image