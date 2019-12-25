import numpy as np
import os
import cv2

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fileList(path):
    ret_list=[]
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".png"): #modify to support other img types
                ret_list.append((os.path.join(root, name),name))
    return ret_list

def cv2_cornersDetect(gray):
    max_corners=100
    qualityLevel = 0.1
    minDistance = 1
    corners=cv2.goodFeaturesToTrack(gray, max_corners, qualityLevel, minDistance)
    matrix_corners=np.zeros(gray.shape)
    if corners is not None:
        for i in corners:
            # notice the order of x,y is changed
            y=int(i[0,0])
            x=int(i[0,1])
            matrix_corners[x,y]=255
    return matrix_corners


def cv2_circlesDetect(gray):
    matrix_circles=np.zeros(gray.shape)
    circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20)
    if isinstance(circles, np.ndarray):
        circles=circles[0]
        for i in circles:
            cv2.circle(matrix_circles,(i[0],i[1]),i[2],255,1)
    return matrix_circles

def cv2_edgeDetect(gray):
    matrix_edges = cv2.Canny(gray,100,200)
    return matrix_edges

def get_detectors(detector_num):
    detectors={"corners":cv2_cornersDetect,"edges":cv2_edgeDetect,"circles":cv2_circlesDetect}
    detectors_order=["corners","edges","circles"]
    return detectors_order[:detector_num],detectors


def confusion_mat(X_img_list,Y_img_list):
    Acc,Precis,Recall,Specif,F1=[0]*5

    for i in range(len(X_img_list)):
        X_img=X_img_list[i]
        Y_img=Y_img_list[i]
        X=X_img/255
        X_neg=1-X
        Y=Y_img/255
        Y_neg=1-Y
        TP=np.multiply(X,Y).sum()+0.0001
        FN=np.multiply(X_neg,Y).sum()+0.001
        FP=np.multiply(X,Y_neg).sum()+0.001
        TN=np.multiply(X_neg,Y_neg).sum()+0.0001

        Acc+=(TP+TN)/(TP+TN+FP+FN)
        Precis+=TP/(TP+FP)
        Recall+=TP/(TP+FN)
        Specif+=TN/(TN+FP)

    result=np.array([Acc,Precis,Recall,Specif,F1])/len(X_img_list)
    F1=2*result[1]*result[2]/(result[1]+result[2])
    result[4]=F1

    return result

if __name__ == '__main__':
    X=np.random.rand(3,3)
    Y=np.random.rand(3,3)
    confusion_mat([X],[Y])
