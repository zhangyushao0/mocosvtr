from mmocr.datasets import RecogLMDBDataset

mjsynth_lmdb_textrecog_data_root = 'data/mjsynth_lmdb'

mjsynth_lmdb_textrecog_train = dict(
    type='RecogLMDBDataset',
    data_root=mjsynth_lmdb_textrecog_data_root,
    ann_file='textrecog_train.lmdb',
    pipeline=None)

mjsynth_lmdb_sub_textrecog_train = dict(
    type='RecogLMDBDataset',
    data_root=mjsynth_lmdb_textrecog_data_root,
    ann_file='subset_textrecog_train.lmdb',
    pipeline=None)


synthtext_lmdb_textrecog_data_root = 'data/synthtext_lmdb'

synthtext_lmdb_textrecog_train = dict(
    type='RecogLMDBDataset',
    data_root=synthtext_lmdb_textrecog_data_root,
    ann_file='textrecog_train.lmdb',
    pipeline=None)

synthtext_lmdb_sub_textrecog_train = dict(
    type='RecogLMDBDataset',
    data_root=synthtext_lmdb_textrecog_data_root,
    ann_file='subset_textrecog_train.lmdb',
    pipeline=None)

synthtext_lmdb_an_textrecog_train = dict(
    type='RecogLMDBDataset',
    data_root=synthtext_lmdb_textrecog_data_root,
    ann_file='alphanumeric_textrecog_train.lmdb',
    pipeline=None)









train_pipeline = [
    dict(type="LoadImageFromNDArray", ignore_empty=True, min_size=5),
    dict(type="LoadOCRAnnotations", with_text=True),
    dict(
        type="RandomApply",
        prob=0.4,
        transforms=[
            dict(
                type="TextRecogGeneralAug",
            ),
        ],
    ),
    dict(
        type="RandomApply",
        prob=0.4,
        transforms=[
            dict(
                type="CropHeight",
            ),
        ],
    ),
    dict(
        type="ConditionApply",
        condition='min(results["img_shape"])>10',
        true_transforms=dict(
            type="RandomApply",
            prob=0.4,
            transforms=[
                dict(
                    type="TorchVisionWrapper",
                    op="GaussianBlur",
                    kernel_size=5,
                    sigma=1,
                ),
            ],
        ),
    ),
    dict(
        type="RandomApply",
        prob=0.4,
        transforms=[
            dict(
                type="TorchVisionWrapper",
                op="ColorJitter",
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1,
            ),
        ],
    ),
    dict(
        type="RandomApply",
        prob=0.4,
        transforms=[
            dict(
                type="ImageContentJitter",
            ),
        ],
    ),
    dict(
        type="RandomApply",
        prob=0.4,
        transforms=[
            dict(
                type="ImgAugWrapper",
                args=[dict(cls="AdditiveGaussianNoise", scale=0.1**0.5)],
            ),
        ],
    ),
    dict(
        type="RandomApply",
        prob=0.4,
        transforms=[
            dict(
                type="ReversePixels",
            ),
        ],
    ),
    dict(type="Resize", scale=(256, 64)),
    dict(
        type="PackTextRecogInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
    ),
]

train_list = [
    mjsynth_lmdb_textrecog_train,
    synthtext_lmdb_textrecog_train,
]

train_dataloader = dict(
    batch_size=256,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="ConcatDataset", datasets=train_list, pipeline=train_pipeline
    ),
)