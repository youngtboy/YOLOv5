_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/smoke.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
load_from = "checkpoints/cascade_rcnn_r101_fpn_20e_smoke.pth"
model = dict(
    pretrained=None,
    backbone=dict(
        depth=101,
        init_cfg=None))

lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4)
