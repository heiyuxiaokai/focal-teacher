# evaluation
evaluation = dict(type="SubModulesDistEvalHook", interval=5000, metric='mAP',
                  save_best='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 10,
    step=[40000, 55000])
runner = dict(type="IterBasedRunner", max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=2)
