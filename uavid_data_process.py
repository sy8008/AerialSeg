from PIL import Image
import numpy as np
import os


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


src_train_base_path = "/home/sy/Dataset/uavid/uavid_train"
dst_train_base_path = "/home/sy/Dataset/uavid_processed/uavid_train"


src_val_base_path = "/home/sy/Dataset/uavid/uavid_val"
dst_val_base_path = "/home/sy/Dataset/uavid_processed/uavid_val"

img_type_suffix = 'png'



# src_img_name_list = os.listdir(src_img_path)
# src_img_name_list.sort(key=lambda x:int(x.split('.')[0][-4:]))


def processUavidFolder(src_train_base_path,dst_train_base_path,img_type_suffix='png'):
    train_seq_list = os.listdir(src_train_base_path)

    train_seq_list.sort(key=lambda x:int(x.split('seq')[1]))

    cnt = 0
    for seq in train_seq_list:
        img_base_path = os.path.join(src_train_base_path,seq,"Images")
        label_base_path = os.path.join(src_train_base_path,seq,"Labels")
        img_name_list = os.listdir(img_base_path)
        img_name_list.sort(key=lambda x:int(x.split('.')[0]))
        for img_name in img_name_list:
            img_path = os.path.join(img_base_path,img_name)
            label_path = os.path.join(label_base_path,img_name)
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('RGB')
            dst_img = img.resize((768,576),Image.LINEAR)
            dst_label = label.resize((768,576),Image.LINEAR)
            dst_img_name = "{:0>8}.{}".format(cnt,img_type_suffix)

            dst_img_path = os.path.join(dst_train_base_path,"Images",dst_img_name)
            dst_img.save(dst_img_path)

            dst_label_path = os.path.join(dst_train_base_path,"Labels",dst_img_name)
            dst_label.save(dst_label_path)
            cnt +=1



processUavidFolder(src_train_base_path,dst_train_base_path)

processUavidFolder(src_val_base_path,dst_val_base_path)


