_base_ = [
    '../_base_/datasets/voc0712_coco.py',
    '../_base_/schedules/voc_schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='GFLLwf',
    ori_config_file='configs/gfl_incre_voc/gfl_r50_fpn_1x_voc_first_10_cats.py',
    #ori_checkpoint_file='/data-nas/ss/model_zoo/mmdet/gfl_incre/gfl_r50_fpn_1x_coco_first_40_cats/epoch_12.pth',
    ori_num_classes=10,
    dist_loss_weight=1.0,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHeadLwf',
        num_classes=20, #80 
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=2,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    ori_checkpoint_file='work_dirs/gfl_r50_fpn_1x_voc_first_10_cats/latest.pth',
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

data_root = 'data/VOCdevkit/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
        ann_file=data_root + 'anns_coco_fmt/voc0712_trainval_only_sel_last_10_cats.json',)
    ),
    val=dict(

        ann_file=data_root + 'anns_coco_fmt/voc07_test.json',
    ),
    test=dict(

        ann_file=data_root + 'anns_coco_fmt/voc07_test.json',
    ))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

custom_hooks = []