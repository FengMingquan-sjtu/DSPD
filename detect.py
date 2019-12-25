from utils.driverDataset import DriverDataset
from torch.utils.data import DataLoader
import torch
import cv2
import utils.common as common
from utils.dataPrepare import HR_H,HR_W
import os.path as osp

import model.san as san


class Detect(object):
    def __init__(self,task_name, model_name,LR_root,GT_root,output_root,weight_path=None):
        self.task_name=task_name
        self.model_name=model_name
        self.LR_root=LR_root
        self.GT_root=GT_root
        self.output_root=output_root
        common.mkdir(output_root)

        if model_name!="cv2":
            data = DriverDataset(task_name=task_name,LR_root=LR_root,GT_root=GT_root,isTrain=False)
            self.dataloader = DataLoader(dataset=data,shuffle=True, batch_size=4)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            if model_name=="":#model name
                pass
            self.model=self.model.to(self.device)
            self.model.load_state_dict(torch.load(weight_path,map_location=self.device))
    
    def run(self):
        if self.model_name=="cv2":
            LR_files=common.fileList(self.LR_root)
            for path,name in LR_files:
                img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                if self.task_name=="edges":
                    res=common.cv2_edgeDetect(img)
                elif self.task_name=="corners":
                    res=common.cv2_cornersDetect(img)
                output_path=osp.join(self.output_root,name)
                cv2.imwrite(output_path,res)
        else:
            self.model.eval()
            for lr,gt,name in self.dataloader:
                lr=lr.to(self.device)
                output=self.model(lr)
                for img_idx,img in enumerate(output.cpu().data):
                    n=name[img_idx]
                    p=osp.join(self.output_root,n)
                    




if __name__ == '__main__':
    input_root="./data/output/EDSR_SR"
    GT_root="./data/temp/test_GT/detect/edges"
    output_root="./data/output/cv2_edges"
    model_name="cv2"
    task_name="edges"
    output_root="./data/output/cv2_corners"
    model_name="cv2"
    task_name="corners"


    
    

    d=Detect(task_name=task_name,model_name=model_name,LR_root=input_root,GT_root=GT_root,output_root=output_root)
    d.run()


        