from detect import Detect
from sr import SR
import utils.common as common
import os.path as osp
import cv2


def generate_sr():
    scale_factor=2
    LR_root="./data/temp/test_LR"
    GT_root="./data/temp/test_GT/SR"
    output_root="./data/output/cv2_SR"
    model_name="cv2"
    cv2_sr=SR(model_name=model_name,scale_factor=scale_factor,LR_root=LR_root,GT_root=GT_root,output_root=output_root)
    cv2_sr.run()

    output_root="./data/output/EDSR_SR"
    model_name="EDSR"
    weight_path="./weight/EDSR_x2.pt"

    
    sr=SR(model_name=model_name,scale_factor=scale_factor,LR_root=LR_root,GT_root=GT_root,output_root=output_root,weight_path=weight_path)
    sr.run()


def generate_detect():
    input_root="./data/output/EDSR_SR"
    GT_root=""
    model_name="cv2"

    output_root="./data/output/EDSR_cv2_edges"
    task_name="edges"
    d=Detect(task_name=task_name,model_name=model_name,LR_root=input_root,GT_root=GT_root,output_root=output_root)
    d.run()

    output_root="./data/output/EDSR_cv2_corners"
    task_name="corners"
    d=Detect(task_name=task_name,model_name=model_name,LR_root=input_root,GT_root=GT_root,output_root=output_root)
    d.run()

    input_root="./data/output/cv2_SR"
    output_root="./data/output/cv2_cv2_edges"
    task_name="edges"
    d=Detect(task_name=task_name,model_name=model_name,LR_root=input_root,GT_root=GT_root,output_root=output_root)
    d.run()

    input_root="./data/output/cv2_SR"
    output_root="./data/output/cv2_cv2_corners"
    task_name="corners"
    d=Detect(task_name=task_name,model_name=model_name,LR_root=input_root,GT_root=GT_root,output_root=output_root)
    d.run()

def compare(result_root,GT_root):
    X_img_list=[]
    Y_img_list=[]
    for path,name in common.fileList(result_root):
        x=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        X_img_list.append(x)
        y_path=osp.join(GT_root,name)
        y=cv2.imread(y_path,cv2.IMREAD_GRAYSCALE)
        Y_img_list.append(y)
    res=common.confusion_mat(X_img_list,Y_img_list)
    print(res)



if __name__ == '__main__':
    generate_sr()
    generate_detect()
    corners_GT="./data/temp/test_GT/detect/corners"
    edges_GT="./data/temp/test_GT/detect/edges"
    EDSR_cv2_edges="./data/output/EDSR_cv2_edges"
    EDSR_cv2_corners="./data/output/EDSR_cv2_corners"
    cv2_cv2_edges="./data/output/cv2_cv2_edges"
    cv2_cv2_corners="./data/output/cv2_cv2_corners"
    print("[Acc,Precis,Recall,Specif,F1]")
    print("EDSR_cv2_edges")
    compare(EDSR_cv2_edges,edges_GT)
    print("cv2_cv2_edges")
    compare(cv2_cv2_edges,edges_GT)

    print("EDSR_cv2_corners")
    compare(EDSR_cv2_corners,corners_GT)
    print("cv2_cv2_corners")
    compare(cv2_cv2_corners,corners_GT)
    

    #SubCorners="./data/output/SubCorners"
    #print("SubCorners")
    #compare(SubCorners,corners_GT)

    #HessianResponse="./data/output/HessianResponse"
    #print("HessianResponse")
    #compare(HessianResponse,corners_GT)

