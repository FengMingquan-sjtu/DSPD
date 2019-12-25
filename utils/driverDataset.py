from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import torch
import numpy as np
import os
#import scipy.misc as misc
import imageio

class DriverDataset(data.Dataset):
    def __init__(self,task_name,LR_root,GT_root,isTrain):
        
        self.LR_root=LR_root
        self.GT_root=GT_root
        self.isTrain=isTrain
        imgs_name = self._fileList(LR_root)
        self.imgs_name=imgs_name
        if task_name=="SR":
            self.load_and_transform=self.SR_load_and_transform
        elif task_name=="edge":
            self.load_and_transform=self.edge_load_and_transform
        
    def __getitem__(self, index):
        name=  self.imgs_name[index]
        lr_path=os.path.join(self.LR_root, name)
        lr = self.load_and_transform(lr_path)
        
        gt_path=os.path.join(self.GT_root, name)
        gt = self.load_and_transform(gt_path)
        
        
        if self.isTrain:
            return (lr,gt)
        else:
            return (lr,gt,name)

    def __len__(self):
        return len(self.imgs_name)
    
    def _fileList(self,path):
        ret_list=[]
        for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith("png"):
                    ret_list.append(name)
        return ret_list

    def SR_load_and_transform(self,path):# for EDSR,SAN
        img = imageio.imread(path)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).float()
        return img

    def edge_load_and_transform(self,path):
        pass


if __name__ == '__main__':
    task_name="SR"
    LR_root="../data/temp/test_LR"
    GT_root="../data/temp/test_GT/SR"
    d=DriverDataset(task_name=task_name,LR_root=LR_root,GT_root=GT_root,isTrain=True)
    n=d.__getitem__(0)
    print(n[0].shape)
    print(n[1].shape)
    print(torch.sum(n[1]))