import torch
num_classes = 2
model_coco = torch.load("checkpoints/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth")

a = model_coco["state_dict"]
model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"][:num_classes,:]
model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"][:num_classes,:]
model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"][:num_classes,:]

model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"][:num_classes]

torch.save(model_coco,"checkpoints/cascade_rcnn_r101_fpn_20e_smoke.pth")

