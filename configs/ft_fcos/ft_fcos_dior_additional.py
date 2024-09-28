_base_ = [
    "base_fcos_default_800.py", "../_base_/schedules/schedule_180k.py"
]

data_root = 'data/dior_r/'
classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt','bridge',
            'chimney', 'Expressway-Service-area', 'Expressway-toll-station', 'dam',
            'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
            'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')
           
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
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
        classes=classes),
    test=dict(
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
        classes=classes),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

model = dict(
    model=dict(bbox_head=dict(num_classes=20), test_cfg=dict(nms=dict(iou_thr=0.5))),
    semi_loss=dict(type='RotatedFTLoss', cls_channels=20),
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