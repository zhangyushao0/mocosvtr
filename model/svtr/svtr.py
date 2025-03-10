from .tps_preprocessor import STN
from .svtr_encoder import SVTREncoder
from .svtr_decoder import SVTRDecoder
from mmengine.model import BaseModel
from mmocr.models.common.dictionary import Dictionary

class SVTR(BaseModel):
    """SVTR文本识别模型 (Scene Text Recognition with a Single Visual Model)

    Args:
        preprocessor: 预处理模块，如STN空间变换网络
        encoder: 特征提取编码器
        decoder: 解码器
        dictionary: 字典
        data_preprocessor: 数据预处理器
    """

    def __init__(
        self,
        preprocessor=None,
        encoder=None,
        decoder=None,
        dictionary=None,
        data_preprocessor=None,
    ):
        super(SVTR, self).__init__(data_preprocessor=data_preprocessor)

        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.dictionary = dictionary

    def extract_feat(self, img):
        """提取特征"""
        if self.preprocessor is not None:
            img = self.preprocessor(img)
        return self.encoder(img)

    def forward(self, inputs, data_samples=None, mode="tensor"):
        """前向传播函数

        Args:
            inputs: 输入图像
            data_samples: 数据样本
            mode: 运行模式: 'tensor', 'loss', 或 'predict'

        Returns:
            tensor、loss字典或预测结果，取决于mode
        """
        if mode == "loss":
            return self.forward_train(inputs, data_samples)
        elif mode == "predict":
            return self.forward_test(inputs, data_samples)
        elif mode == "tensor":
            return self.extract_feat(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def forward_train(self, inputs, data_samples=None):
        """训练模式前向传播"""
        feats = self.extract_feat(inputs)
        return self.decoder.loss(out_enc=feats, data_samples=data_samples)

    def forward_test(self, inputs, data_samples=None):
        """测试模式前向传播"""
        feats = self.extract_feat(inputs)
        probs = self.decoder.forward_test(
            feat=None, out_enc=feats, data_samples=data_samples
        )
        return probs


dictionary = Dictionary(
    dict_file="model/svtr/dicts/lower_english_digits.txt",
    with_padding=True,
    with_unknown=True,
)



SVTRModel = SVTR(
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
    decoder=SVTRDecoder(
        in_channels=192,
        module_loss=dict(type="CTCModuleLoss", letter_case="lower", zero_infinity=True),
        postprocessor=dict(type="CTCPostProcessor"),
        dictionary=dictionary,
    ),
)
