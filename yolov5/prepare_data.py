import os
import shutil

image_path = "/home/data/323/SmokeDatset/images"
img_list = os.listdir(image_path)
for i in img_list:
    shutil.move(os.path.join(image_path,i),"/home/data/323/")

ann_list