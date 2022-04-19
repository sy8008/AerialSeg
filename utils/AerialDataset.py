from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from .utils import mask2label
from .AerialTransforms import TrainAug,EvalAug
import numpy as np

class AerialDataset(Dataset):
    def __init__(self,data_path,dataset,mode,crop_size=512):
        self.crop_size = crop_size
        self.dataset = dataset
        self.mode = mode
        assert self.mode in ['train','val']
        self.img_list,self.gt_list = [],[]
        if dataset=='Potsdam':
            self.img_path=os.path.join(data_path,'2_Ortho_RGB')
            self.gt_path=os.path.join(data_path,'Potsdam_label')
            if self.mode=='train':
                self.list=os.path.join(data_path,'Potsdam_train.txt')
            else:
                self.list=os.path.join(data_path,'Potsdam_val.txt')
            with open(self.list) as f:
                for each_file in f:
                    file_name = each_file.strip()
                    img = os.path.join(self.img_path,file_name)
                    file_name = file_name[:-7]+"label.png"
                    gt = os.path.join(self.gt_path,file_name)
                    assert os.path.isfile(img),"Images %s cannot be found!" %img
                    assert os.path.isfile(gt),"Ground truth %s cannot be found!" %gt
                    self.img_list.append(img)
                    self.gt_list.append(gt)
        elif dataset=='UDD5':
            self.img_path=data_path
            self.gt_path=data_path
            if self.mode=='train':
                self.list=os.path.join(data_path,'metadata/train.txt')
            else:
                self.list=os.path.join(data_path,'metadata/val.txt')
            with open(self.list) as f:
                for each_file in f:
                    img_name,gt_name = each_file.strip().split(' ')[0],each_file.strip().split(' ')[1]
                    img = os.path.join(self.img_path,img_name)
                    gt = os.path.join(self.gt_path,gt_name)
                    assert os.path.isfile(img),"Images %s cannot be found!" %img
                    assert os.path.isfile(gt),"Ground truth %s cannot be found!" %gt
                    self.img_list.append(img)
                    self.gt_list.append(gt)
        elif dataset=='UDD6':
            self.img_path=data_path
            self.gt_path=data_path
            if self.mode=='train':
                self.list=os.path.join(data_path,'metadata/train.txt')
            else:
                self.list=os.path.join(data_path,'metadata/val.txt')
            with open(self.list) as f:
                for each_file in f:
                    img_name,gt_name = each_file.strip().split(' ')[0],each_file.strip().split(' ')[1]
                    img = os.path.join(self.img_path,img_name)
                    gt = os.path.join(self.gt_path,gt_name)
                    assert os.path.isfile(img),"Images %s cannot be found!" %img
                    assert os.path.isfile(gt),"Ground truth %s cannot be found!" %gt
                    self.img_list.append(img)
                    self.gt_list.append(gt)
        elif dataset=='Custom':
            self.list = os.listdir(data_path)
            for each_file in self.list:
                each_file = os.path.join(data_path,each_file)
                self.img_list.append(each_file)
                self.gt_list.append(each_file)
        
        elif dataset == 'ArsUDD':
            self.img_path= os.path.join(data_path,'JPEGImages')
            self.gt_path= os.path.join(data_path,'labels')
            if self.mode=='train':
                self.list=os.path.join(data_path,'train.txt')
            else:
                self.list=os.path.join(data_path,'val.txt')
            with open(self.list) as f:
                for each_file in f.readlines():
                    gt_name = each_file.strip()
                    img_name = gt_name.split('.')[0] + '.jpg'
                    img = os.path.join(self.img_path,img_name)
                    gt = os.path.join(self.gt_path,gt_name)
                    assert os.path.isfile(img),"Images %s cannot be found!" %img
                    assert os.path.isfile(gt),"Ground truth %s cannot be found!" %gt
                    self.img_list.append(img)
                    self.gt_list.append(gt)            
        
        else:
            raise NotImplementedError
        print(f"{len(self.img_list)} pairs to {self.mode}...")
        if self.mode=='train':
            self.augtrans = TrainAug(self.crop_size)
        

        
    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self,index):
        if self.mode=='train':
            img = Image.open(self.img_list[index]).convert('RGB')
            gt = mask2label(np.array(Image.open(self.gt_list[index]).convert('RGB')),self.dataset) 
            #or more efficiently, directly load label map
            # gt = Image.open(self.gt_list[index]).convert('RGB').convert('P') # the original code  using 'P' mode to load UDD gt 

            # img_test = np.array(img)
            # gt_test = np.array(gt)
            # test_bin = np.bincount(gt_test.reshape(-1))
            # test_bin2 = np.unique(gt_test.reshape(-1), axis=0)
            #Trans from PIL pair to tensor pair
            # gt = Image.fromarray(gt)
            # palette = gt.getpalette()
            # res = np.array(palette).reshape(-1,3)
            gt = Image.fromarray(gt).convert('L')
            return self.augtrans(img,gt)
        else:
            sample = {'img':self.img_list[index],'gt':self.gt_list[index]}
            return sample
        
if __name__ == "__main__":
   print("AerialDataset.py")
