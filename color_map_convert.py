import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import random

# img = Image.open(img_name).convert('RGB')
# Image.fromarray(mask).save(png_name)


# img_list = os.listdir('/home/sy/dataset/ArsUDD/labels')
# random.shuffle(img_list)

# total_len = len(img_list)

# ninety_percent = int(0.9 * total_len)

# train_list = img_list[:ninety_percent]
# val_list = img_list[ninety_percent:]

# train_list_name = "/home/sy/dataset/ArsUDD/train.txt"
# val_list_name = "/home/sy/dataset/ArsUDD/val.txt"

# read_test_obj =  open('/home/sy/dataset/ArsUDD/train.txt')
# lines = read_test_obj.readlines()

# def writeList(input_list,filename):
#     file_write_obj = open(filename, 'a')
#     for var in input_list:
#         file_write_obj.write(var)
#         file_write_obj.write('\n')

#     file_write_obj.close()


# writeList(train_list,train_list_name)
# writeList(val_list,val_list_name)



test_label_path = "/home/sy/dataset/ArsUDD/labels/000001_001.png"
test_label_path2 = "/home/sy/dataset/UDD/UDD5/train/gt/000001.png"
img = Image.open(test_label_path)
img = img.convert('P', palette=Image.ADAPTIVE, colors=8)
img = np.array(img)
cnt = np.bincount(img.reshape(-1))
#cnt2 = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
cnt2 = np.unique(img.reshape(-1),return_counts=True)

# Aeroscape Dataset
rgb2label_aeroscape = {}
rgb2label_aeroscape[(0,0,0)] = 0 # Background
rgb2label_aeroscape[(192,128,128)] = 1 # Person
rgb2label_aeroscape[(0,128,0)] = 2 # Bike
rgb2label_aeroscape[(0,64,0)] = 3 # Vegetation
rgb2label_aeroscape[(192,0,0)] = 4 # Obstacle
rgb2label_aeroscape[(0,128,128)] = 5 # Sky
rgb2label_aeroscape[(128,128,0)] = 6 # Road
rgb2label_aeroscape[(192,0,128)] = 7 # Animal
rgb2label_aeroscape[(192,128,0)] = 8 # building
rgb2label_aeroscape[(128,128,128)] = 9 # car
rgb2label_aeroscape[(128,0,0)] = 10 # drone
rgb2label_aeroscape[(0,0,128)] = 11 # boat

label2rgb_aeroscape = dict((v,k) for k,v in rgb2label_aeroscape.items())


# UDD5 Dataset
rgb2label_UDD5= {}
rgb2label_UDD5[(107,142,35)]=0     # Vegetation
rgb2label_UDD5[(102,102,156)]=1    # Building
rgb2label_UDD5[(128,64,128)]=2     # Road
rgb2label_UDD5[(0,0,142)]=3        # Vehicle
rgb2label_UDD5[(0,0,0)]=4          # Other


label2rgb_ArsUDD = {}
label2rgb_ArsUDD[0]=(0,64,0) # Vegetation
label2rgb_ArsUDD[1]=(192,128,0) # Building
label2rgb_ArsUDD[2]=(128,128,0) # Road
label2rgb_ArsUDD[3]=(128,128,128)  # car or bike
label2rgb_ArsUDD[4]=(0,0,128) # Other
label2rgb_ArsUDD[5]=(0,0,0) # Background
label2rgb_ArsUDD[6]=(192,128,128) # Person or animal
label2rgb_ArsUDD[7]=(0,128,128) # sky


label_aeroscape2ArsUDD = {0:5,1:6,2:3,3:0,4:4,5:7,6:2,7:6,8:1,9:3,10:4,11:4}


rgb_aeroscape2ArsUDD = dict((k,label2rgb_ArsUDD[label_aeroscape2ArsUDD[v]]) for k,v in rgb2label_aeroscape.items())

def my_func(val):
    global rgb_aeroscape2ArsUDD
    return rgb_aeroscape2ArsUDD[val]

aeros_scape_laebl_path = "/home/sy/dataset/aeroscapes/Visualizations"
ArsUDD_label_path = "/home/sy/dataset/ArsUDD/labels"





for f in tqdm(os.listdir(aeros_scape_laebl_path)):
    img_name = os.path.join(aeros_scape_laebl_path,f)
    img = Image.open(img_name).convert('RGB')
    img = np.array(img)
    cvt_img = np.zeros_like(img)
    for rgb_src, rgb_dst in rgb_aeroscape2ArsUDD.items():
        rgb_src = np.array(rgb_src)
        rgb_dst = np.array(rgb_dst)
        idx = np.where(np.all(img == rgb_src, axis=-1))
        cvt_img[idx] = rgb_dst
    dst_img_name = os.path.join(ArsUDD_label_path,f)
    Image.fromarray(cvt_img).save(dst_img_name)








