modality = 'km'
graph = 'coco_new'
work_dir = f'./work_dirs/k400/km'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='GCN_Module',
        gcn_ratio=0.125,
        gcn_ctr='T',
        gcn_ada='T',
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02)),
    cls_head=dict(type='SimpleHead', data_cfg='k400', num_classes=400, in_channels=384))

memcached = True
mc_cfg = ('localhost', 22077)
dataset_type = 'PoseDataset'
ann_file = '/data/k400/k400_hrnet.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
box_thr = 0.5
valid_ratio = 0.0

train_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='Kinetics_Transform'),
    dict(type='GenSkeFeat', dataset='coco_new', feats=[modality]),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='Kinetics_Transform'),
    dict(type='GenSkeFeat', dataset='coco_new', feats=[modality]),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='Kinetics_Transform'),
    dict(type='GenSkeFeat', dataset='coco_new', feats=[modality]),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=16,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline,
        box_thr=box_thr, valid_ratio=valid_ratio, memcached=memcached, mc_cfg=mc_cfg),
    val=dict(
        type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline,
        box_thr=box_thr, memcached=memcached, mc_cfg=mc_cfg),
    test=dict(
        type=dataset_type, ann_file=ann_file, split='val', pipeline=test_pipeline,
        box_thr=box_thr, memcached=memcached, mc_cfg=mc_cfg))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 150
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
