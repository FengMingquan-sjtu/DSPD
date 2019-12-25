import cv2
import os
import numpy as np
from torchvision import transforms as T
from PIL import Image

import sys
sys.path.append('../')
import utils.common as common

HR_H=100
HR_W=100
ratio=1.00
crop_H=int(ratio*HR_H)
crop_W=int(ratio*HR_W)
stride=1.00
stride_H=int(stride*HR_H)
stride_W=int(stride*HR_W)



class DataPrepare:
    def __init__(self,input_HR_path,output_GT_root,output_LR_root,scale_factor,detectors_order,detectors):
        self.input_HR_path=input_HR_path
        self.output_LR_root=output_LR_root
        self.output_GT_root=output_GT_root
        self.detectors=detectors
        self.detectors_order=detectors_order
        self.scale_factor=scale_factor

        self.transform=self.default_transform()

        self.LR_size=(int(HR_W/self.scale_factor),int(HR_H/self.scale_factor))

        if not os.path.exists(input_HR_path):
            raise IOError("input_HR_path= %s not exists"%input_HR_path)


        
        self.output_detect_GT_paths=[os.path.join(output_GT_root,"detect",d) for d in detectors_order]
        for p in self.output_detect_GT_paths:
            common.mkdir(p)

        self.output_SR_GT_path=os.path.join(output_GT_root,"SR")
        common.mkdir(self.output_SR_GT_path)
        

        common.mkdir(self.output_LR_root)



    def dataPrepare(self):
        ## prepare SR_GT
        #print("NOTE: no spliting")
        self.pre_process_spliting()

        
        HR_files=common.fileList(self.output_SR_GT_path) #the splited HR files

        ## prepare detect_GT
        self.process_and_save(HR_files,self.output_detect_GT_paths)


        ## prepare LR
        for path,name in HR_files:
            color=cv2.imread(path,cv2.IMREAD_COLOR)
            LR_img=cv2.resize(color,self.LR_size,interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.output_LR_root,name),LR_img)

        print("Finish Preparation %d imgs"%(len(HR_files)))

    def process_and_save(self,files,img_paths):
        
        for path,name in files:
            gray=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            #print(gray,path)
            GT=self.detect(gray)
            for idx in range(len(self.detectors_order)):
                file=os.path.join(img_paths[idx],name)
                cv2.imwrite(file,GT[idx])


    def pre_process_spliting(self):
        ##extract files
        input_HR_files=common.fileList(self.input_HR_path)

        ##pre-process(split into fixed size, random transform)
        
        for path,name in input_HR_files:
            splited=self.split(path)
            for cnt,img in enumerate(splited):
                splited_name=str(cnt)+"_"+name
                img=self.transform(img)
                p=os.path.join(self.output_SR_GT_path,splited_name)
                img.save(p)
        print("pre-process spliting finished")

    def split(self,path):
        img=Image.open(path)
        w,h=img.size
        ret_list=list()
        
        for upper in range(0, h-crop_H ,stride_H):
            lower=upper+crop_H
            for left in range(0, w-crop_W, stride_W):
                right=left+crop_W
                box=(left, upper, right, lower)
                croped=img.crop(box)
                ret_list.append(croped)
        return ret_list


    def default_transform(self):
        transforms = T.Compose([
            T.RandomRotation(10), #random rotate in (-10,10),may lead to furry edge
            T.CenterCrop((HR_H,HR_W)),
            T.RandomHorizontalFlip(), #horizontal flip  with p=0.5
            T.RandomVerticalFlip(), #Vertical flip with p=0.5
            ])
        return transforms

    def detect(self,img):
        return [self.detectors[d](img) for d in self.detectors_order]







def _main():
    input_HR_path="./data/input/test_input_small"
    output_GT_root="./data/temp/test_GT"
    output_LR_root="./data/temp/test_LR"
    scale_factor=2
    detector_num=2
    detectors_order,detectors=common.get_detectors(detector_num)
    

    data=DataPrepare(input_HR_path,output_GT_root,output_LR_root,scale_factor,detectors_order,detectors)
    data.dataPrepare()

if __name__ == '__main__':
    _main()
        



