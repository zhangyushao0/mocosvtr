from torch.optim import AdamW, SGD
from mmengine.runner import Runner
from dataloader.union14mu import loader
import argparse
from model.svtr.moco import MoCoSVTRModel
import torch


# torch.set_float32_matmul_precision('high')
def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    optimizer = dict(type=AdamW, lr=5e-4, weight_decay=1e-4)
    optim_wrapper = dict(optimizer=optimizer)
    param_scheduler = [
        dict(
            type="LinearLR",
            start_factor=0.2,
            end_factor=1.0,
            end=2,
            verbose=False,
            convert_to_iter_based=True,
        ),
        # dict(
        #     type="CosineAnnealingLR",
        #     T_max=78,
        #     end=80,
        #     eta_min=1e-6,
        #     verbose=False,
        #     convert_to_iter_based=True,
        # ),
    ]
    default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=20))
    runner = Runner(
        # 用以训练和验证的模型，需要满足特定的接口需求
        model=MoCoSVTRModel,
        # 工作路径，用以保存训练日志、权重文件信息
        work_dir="./work_dir",
        # 训练数据加载器，需要满足 PyTorch 数据加载器协议
        train_dataloader=loader,
        # 优化器包装，用于模型优化，并提供 AMP、梯度累积等附加功能
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        # 训练配置，用于指定训练周期、验证间隔等信息
        train_cfg=dict(by_epoch=True, max_epochs=80, val_interval=-1),
        # # 验证数据加载器，需要满足 PyTorch 数据加载器协议
        # val_dataloader=test_loader,
        # 验证配置，用于指定验证所需要的额外参数
        # val_cfg=dict(),
        # 用于验证的评测器，这里使用默认评测器，并评测指标
        # val_evaluator=dict(type=OCRAccuracy, dictionary=dictionary),
        default_hooks=default_hooks,
        launcher=args.launcher,
    )
    runner.train()


if __name__ == "__main__":
    main()
