_base_ = [
    "base_fcos_default.py", "../_base_/schedules/schedule_60k.py"
]

data_root = 'data/split_ss_dota_v15/'
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter',
           'container-crane')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        sup=dict(
            ann_file=data_root + 'train_30_labeled/annfiles/',
            img_prefix=data_root + 'train_30_labeled/images/',
            classes=classes,
        ),
        unsup=dict(
            ann_file=data_root + 'train_30_unlabeled/images/',
            img_prefix=data_root + 'train_30_unlabeled/images/',
            classes=classes,
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

model = dict(
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=6400,
        unsup_weight=4.0
    )
)
