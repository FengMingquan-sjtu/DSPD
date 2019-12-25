import cv2
import numpy as np
import os
import torch.nn.functional as F
import torch
import sys
sys.path.append('../')
from utils.dataPrepare import get_detectors,fileList,GRAY_MAX
from utils.common import confusion_mat


class Baseline:
    def __init__(self,test_LR_path,test_GT_path,loss_func,baseline_output_path,scale_factor,detectors_order):
        self.test_LR_path=test_LR_path
        self.test_GT_path=test_GT_path
        self.loss_func=loss_func
        self.baseline_output_path=baseline_output_path
        self.detectors_order=detectors_order
        self.scale_factor=scale_factor
        _,self.detectors=get_detectors(len(detectors_order))
        self.baseline_output_paths=[os.path.join(baseline_output_path,d) for d in self.detectors_order]
        self.baseline_output_paths.append(os.path.join(baseline_output_path,"subCorner"))
        for p in self.baseline_output_paths:
            if not os.path.exists(p):
                os.makedirs(p)
    def detect(self):
        for LR_path,name in fileList(self.test_LR_path):
            LR_img=cv2.imread(LR_path,cv2.IMREAD_GRAYSCALE)
            x,y=LR_img.shape
            ## NOTICE: x,y order in cv2.resize is reverted!
            HR_img=cv2.resize(LR_img,(y*self.scale_factor,x*self.scale_factor),cv2.INTER_CUBIC) 
            for det_idx,det_name in enumerate(self.detectors_order):
                p=os.path.join(self.baseline_output_paths[det_idx],name)
                result=self.detectors[det_name](HR_img)
                cv2.imwrite(p,result)
            
            #subCorner
            p=os.path.join(self.baseline_output_paths[-1],name)
            result=self.cv2SubCorner(LR_img)
            cv2.imwrite(p,result)

    def calculate_loss(self):
        for det_idx,det_name in enumerate(self.detectors_order):
            p=self.baseline_output_paths[det_idx]
            print("evaluate:",det_name)
            total_conf_mat=np.zeros([5])
            total_loss=0
            for X_path,name in fileList(p):
                X=cv2.imread(X_path,cv2.IMREAD_GRAYSCALE)
                #print(X_path)
                Y_path=os.path.join(self.test_GT_path,det_name,name)
                #print(Y_path)
                Y=cv2.imread(Y_path,cv2.IMREAD_GRAYSCALE)
                loss=self.loss_func(torch.tensor(X,dtype=torch.float).div(255),torch.tensor(Y,dtype=torch.float).div(255))
                total_loss+=loss.item()
                conf=confusion_mat(X,Y)
                total_conf_mat=total_conf_mat+conf
            print("avg_loss=",total_loss/len(fileList(p)))
            print("avg_confusion[Acc,Precis,Recall,Specif,F1]=",total_conf_mat/len(fileList(p)))
        
        #det_name="subCorner"
        det_name=False
        if det_name:
            p=self.baseline_output_paths[-1]
            print("evaluate:",det_name)
            total_conf_mat=np.zeros([5])
            total_loss=0
            for X_path,name in fileList(p):
                X=cv2.imread(X_path,cv2.IMREAD_GRAYSCALE)
                Y_path=os.path.join(test_GT_path,"corners",name)
                Y=cv2.imread(Y_path,cv2.IMREAD_GRAYSCALE)
                loss=self.loss_func(torch.tensor(X,dtype=torch.float).div(255),torch.tensor(Y,dtype=torch.float).div(255))
                total_loss+=loss.item()
                conf=confusion_mat(X,Y)
                total_conf_mat=total_conf_mat+conf
            print("avg_loss=",total_loss/len(fileList(p)))
            print("avg_confusion[Acc,Precis,Recall,Specif,F1]=",total_conf_mat/len(fileList(p)))
                





    def cv2SubCorner(self,gray):
        x,y=gray.shape
        x*=self.scale_factor
        y*=self.scale_factor
        matrix_corners=np.zeros((x,y))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
        corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 5)
        if corners is not None:
            sub_corners=cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
        

            for c in sub_corners:
                y,x=c.flatten()# notice the order of x,y is changed
                x=int(x*scale_factor)
                y=int(y*scale_factor)
                try:
                    matrix_corners[x,y]=GRAY_MAX
                except:
                    continue

            
        return matrix_corners




scale_factor=4
detectors_order=["corners","edges"]
loss_func = F.mse_loss

def test0():
    test_LR_path="../data/temp/test_LR/img/origin"
    test_GT_path="../data/temp/test_GT/img"
    baseline_output_path="../data/output/baseline"
    b=Baseline(test_LR_path,test_GT_path,loss_func,baseline_output_path,scale_factor,detectors_order)
    #b.detect()
    b.calculate_loss()

def test1():
    test_LR_path="../data/temp/SR_LR"
    test_GT_path="../data/temp/SR_temp_GT/img"
    baseline_output_path="../data/output/baseline_SR"
    b=Baseline(test_LR_path,test_GT_path,loss_func,baseline_output_path,scale_factor,detectors_order)
    b.detect()
    b.calculate_loss()
def test2():
    test_LR_path="../data/temp/SR_LR"
    test_GT_path="../data/temp/SR_temp_GT/img"
    baseline_output_path="../data/temp/SR_temp_result/img"
    b=Baseline(test_LR_path,test_GT_path,loss_func,baseline_output_path,scale_factor,detectors_order)
    b.calculate_loss()

if __name__ == '__main__':
    print("baseline:")
    test1()
    print("EDSR")
    test2()
