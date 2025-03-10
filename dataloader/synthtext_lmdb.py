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
