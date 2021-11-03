import os
img_info = os.listdir("/home/data/323")
img_info.sort()
train_txt_path = "../train.txt"
with open(train_txt_path,"w") as fn:   
    for i in range(len(img_info)//2):
        fn.write(img_info[2*i].split(".")[0]+"\n")
