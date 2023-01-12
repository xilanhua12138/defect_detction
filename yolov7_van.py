_base_ = './configs/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py'
data_root = 'data/balloon/'

train_batch_size_per_gpu = 1
train_num_workers = 1
max_epochs = 10
save_epoch_intervals = 1
num_classes = 1
metainfo = {
    'CLASSES': ('balloon', ),
    'PALETTE': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='val/'),
        ann_file='val.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val.json')

test_evaluator = val_evaluator

default_hooks = dict(
    logger=dict(interval=1),
    visualization=dict(draw=True, interval=1))

# 导入 mmcls.models 使得可以调用 mmcls 中注册的模块
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/van/van-base_8xb128_in1k_20220501-6a4cc31b.pth'  # noqa
widen_factor = 1.0
in_channels = [128, 320, 512]
out_channels = [128, 256, 512]
in_channels_head = [256, 512, 1024]
strides = [8, 16, 32]
model = dict(
    backbone=dict(
        _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
        type='mmcls.VAN', # 使用 mmcls 中的 MobileNetV3
        arch='b',
        out_indices=(1, 2, 3), # 修改 out_indices
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,
            prefix='backbone.')), # MMCls 中主干网络的预训练权重含义 prefix='backbone.'，为了正常加载权重，需要把这个 prefix 去掉。
    neck=dict(
        type='YOLOv7PAFPN',
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.5,
            block_ratio=0.25,
            num_blocks=4,
            num_convs_in_block=1),
        upsample_feats_cat_first=False,
        in_channels=in_channels,
        # The real output channel will be multiplied by 2
        out_channels=out_channels,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            in_channels=in_channels_head, # head 部分输入通道也要做相应更改
            featmap_strides=strides,
            num_base_priors=3))
)

