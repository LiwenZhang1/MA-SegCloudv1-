import cv2
import numpy as np
import os

#预测结果路径
pred_path = "I:\lunwen\deep3/result_acc0.9704night"
#标签路径
lab_path = r"I:\lunwen\deep3\dataset2\datasets\night\png"


def tpcount(imgp,imgl):
    n = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgp[i,j][0] == 255 and imgl[i,j] == 255:
                n = n+1
    return n

def fncount (imgp,imgl):
    n = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i,j] == 255 and imgp[i,j][0] == 0:
                n = n+1
    return n

def fpcount(imgp,imgl):
    n = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i,j] == 0 and imgp[i,j][0] == 255:
                n+=1
    return n

def tncount(imgp,imgl):
    n=0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i,j] == 0 and imgp[i,j][0] == 0:
                n += 1
    return n




imgs = os.listdir(pred_path)
a = len(imgs)
TP = 0
FN = 0
FP = 0
TN = 0
c = 0
for name in imgs:

    imgp = cv2.imread(pred_path + '/' + name, -1)
    imgp = np.array(imgp)

    imgl = cv2.imread(lab_path + '/' + name, -1)
    imgl = np.array(imgl)

    WIDTH = imgl.shape[0]
    HIGTH = imgl.shape[1]

    TP += tpcount(imgp, imgl)
    FN += fncount(imgp, imgl)
    FP += fpcount(imgp, imgl)
    TN += tncount(imgp, imgl)

    c += 1
    print('已经计算：'+str(c) + ',剩余数目：'+str(a-c))

print('TP:'+str(TP))
print('FN:'+str(FN))
print('FP:'+str(FP))
print('TN:'+str(TN))


#准确率  accuracy
zq = (int(TN)+int(TP))/(int(WIDTH)*int(HIGTH)*int(len(imgs)))
#精确率    precision
jq = int(TP)/(int(TP)+int(FP))
#召回率    recall
zh = int(TP)/(int(TP)+int(FN))
#F1     F-score
f1 = int(TP)*2/(int(TP)*2+int(FN)+int(FP))
# Miou
Miou = ((int(TP)/(int(TP)+int(FP)+int(FN)))+(int(TN)/(int(TN)+int(FP)+int(FN))))
# Error Rate
ER = (int(FP)+int(FN))/(int(TP)+int(FP)+int(TN)+int(FN))
# FPR
FPR = int(FP)/(int(FP)+int(TN))
# TPR
TPR = int(TP) / (int(TP) + int(FN))

print('准确率(accuracy)：'+ str(zq))
print('精确率(precision)：'+ str(jq))
print('召回率(recall)：'+ str(zh))
print('F1值(F-score)：'+ str(f1))
print('Miou:'+ str(0.5*(Miou)))
print('ER:'+ str(ER))

