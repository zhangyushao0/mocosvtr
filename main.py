from torch.optim import AdamW
from mmengine.runner import Runner
from dataloader.corrupted import train_loader, test_loader
from dataloader.corrupted import num_chars, idx_to_char
from model.crnn import MMCRNN
from test import OCRAccuracy, OCRCharAccuracy

runner = Runner(
    # 用以训练和验证的模型，需要满足特定的接口需求
    model=MMCRNN(num_chars),
    # 工作路径，用以保存训练日志、权重文件信息
    work_dir="./work_dir",
    # 训练数据加载器，需要满足 PyTorch 数据加载器协议
    train_dataloader=train_loader,
    # 优化器包装，用于模型优化，并提供 AMP、梯度累积等附加功能
    optim_wrapper=dict(optimizer=dict(type=AdamW, lr=5 / (10**4) * 1024 / 2048)),
    # 训练配置，用于指定训练周期、验证间隔等信息
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    # 验证数据加载器，需要满足 PyTorch 数据加载器协议
    val_dataloader=test_loader,
    # 验证配置，用于指定验证所需要的额外参数
    val_cfg=dict(),
    # 用于验证的评测器，这里使用默认评测器，并评测指标
    val_evaluator=dict(type=OCRAccuracy, idx_to_char=idx_to_char),
)

runner.train()
