# from mmcls.models import VAN
# import torch
# model = VAN(arch='b')
# inputs = torch.rand(1, 3, 640, 640)
# level_outputs = model(inputs)
# for level_out in level_outputs:
#     print(tuple(level_out.shape))
# import torch
# from mmyolo.models import PPYOLOECSPResNet
# from mmyolo.utils import register_all_modules

# # 注册所有模块
# register_all_modules()

# imgs = torch.randn(1, 3, 640, 640)
# out_indices=(2, 3, 4)
# model = PPYOLOECSPResNet(arch='P5', widen_factor=1.0, out_indices=out_indices)
# out = model(imgs)
# out_shapes = [out[i].shape for i in range(len(out_indices))]
# print(out_shapes)