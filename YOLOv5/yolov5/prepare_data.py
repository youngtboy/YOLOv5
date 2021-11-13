import os
import shutil

image_path = "/home/data/323/SmokeDataset/images"
img_list = os.listdir(image_path)
for i in img_list:
    shutil.move(os.path.join(image_path,i),"/home/data/323/")

ann_path = "/home/data/323/SmokeDataset/annotations"
ann_list = os.listdir(ann_path)
for i in ann_list:
    shutil.move(os.path.join(ann_path,i),"/home/data/323/")
