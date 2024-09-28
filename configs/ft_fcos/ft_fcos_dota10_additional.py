_base_ = [
    "base_fcos_default.py", "../_base_/schedules/schedule_180k.py"
]

data_root = 'data/split_ss_1024/'
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter')
           
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        sup=dict(
            ann_file=data_root + 'trainval/annfiles/',
            img_prefix=data_root + 'trainval/images/',
            classes=classes,
        ),
        # for debug only
        unsup=dict(
            ann_file='data/semi_dota/train/empty_annfiles/',
            img_prefix='data/semi_dota/train/images/',
            classes=classes,
        ),
    ),
    val=dict(
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        classes=classes),
    test=dict(
        ann_file=data_root + 'test_gap200/images/',
        img_prefix=data_root + 'test_gap200/images/',
        classes=classes),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

model = dict(
    model=dict(bbox_head=dict(num_classes=15)),
    semi_loss=dict(type='RotatedFTLoss', cls_channels=15),
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=25600,
        unsup_weight=4.0
    )
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.9996, interval=1, start_steps=12800),
]
