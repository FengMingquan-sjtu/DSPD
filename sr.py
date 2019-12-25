from utils.driverDataset import DriverDataset
from torch.utils.data import DataLoader
import torch
import cv2
import utils.common as common
from utils.dataPrepare import HR_H,HR_W
import os.path as osp

import model.san as san
import model.edsr as edsr
import model.mdsr as mdsr

#import scipy.misc as misc
import imageio

class SR(object):
    """docstring for SR"""
    def __init__(self, model_name,scale_factor,LR_root,GT_root,output_root,weight_path=None):
        self.model_name=model_name
        self.scale_factor=scale_factor
        self.LR_root=LR_root
        self.GT_root=GT_root
        self.output_root=output_root
        common.mkdir(output_root)

        if model_name!="cv2":
            data = DriverDataset(task_name="SR",LR_root=LR_root,GT_root=GT_root,isTrain=False)
            self.dataloader = DataLoader(dataset=data,shuffle=True, batch_size=4)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            if model_name=="SAN":
                n_resgroups=20
                n_resblocks=10
                n_feats=64
                self.model=san.SAN(n_resgroups,n_resblocks,n_feats,scale_factor)
            elif model_name=="EDSR":
                n_resblocks=32
                n_feats=256
                self.model=edsr.EDSR(n_resblocks,n_feats,scale_factor)
            elif model_name=="MDSR":
                n_resblocks=80
                n_feats=64
                self.model=mdsr.MDSR(n_resblocks,n_feats,scale_factor)

            self.model=self.model.to(self.device)
            self.model.load_state_dict(torch.load(weight_path,map_location=self.device))
    
    def run(self):
        if self.model_name=="cv2":
            LR_files=common.fileList(self.LR_root)
            for path,name in LR_files:
                img=cv2.imread(path)
                img=cv2.resize(img,(HR_H,HR_W),interpolation=cv2.INTER_CUBIC)
                output_path=osp.join(self.output_root,name)
                cv2.imwrite(output_path,img)
        else:
            self.model.eval()
            for lr,gt,name in self.dataloader:
                lr=lr.to(self.device)
                output=self.model(lr)
                for img_idx,img in enumerate(output.cpu().data):
                    n=name[img_idx]
                    p=osp.join(self.output_root,n)
                    output=img
                    output=output.clamp(0, 255).round()
                    output=output.byte().permute(1, 2, 0)
                    #output=output.transpose(1,2,0)
                    output=output.numpy()
                    imageio.imsave(p, output)




if __name__ == '__main__':
    LR_root="./data/temp/test_LR"
    GT_root="./data/temp/test_GT/SR"
    output_root="./data/output/cv2_SR"
    model_name="cv2"
    weight_path=None
    #output_root="./data/output/SAN_SR"
    #model_name="SAN"
    #weight_path="./weight/SAN_BI2X.pt"
    #output_root="./data/output/EDSR_SR"
    #model_name="EDSR"
    #weight_path="./weight/EDSR_x2.pt"

    #output_root="./data/output/MDSR_SR"
    #model_name="MDSR"
    #weight_path="./weight/MDSR.pt"

    scale_factor=2

    
    

    sr=SR(model_name=model_name,scale_factor=scale_factor,LR_root=LR_root,GT_root=GT_root,output_root=output_root,weight_path=weight_path)
    sr.run()


        