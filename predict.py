import copy
import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# 修改后的网络
from nets_0.deeplab import Deeplabv3

# # 原始网络
# from deeplab_Mobile.nets_yuan.deeplab import Deeplabv3


if __name__ == "__main__":
    #---------------------------------------------------#
    #   定义了输入图片的颜色，当我们想要去区分两类的时候
    #   我们定义了两个颜色，分别用于背景和斑马线
    #   [0,0,0], [0,255,0]代表了颜色的RGB色彩
    #---------------------------------------------------#
    class_colors = [[0,0,0],[255,255,255]]
    #---------------------------------------------#
    #   定义输入图片的高和宽，以及种类数量
    #---------------------------------------------#
    HEIGHT = 320
    WIDTH = 320
    #---------------------------------------------#
    #   背景 + 云 = 2
    #---------------------------------------------#
    NCLASSES = 2

    #---------------------------------------------#
    #   载入模型
    #---------------------------------------------#
    model = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,3))
    #--------------------------------------------------#
    #   载入权重，训练好的权重会保存在logs文件夹里面
    #   我们需要将对应的权重载入
    #   修改model_path，将其对应我们训练好的权重即可
    #   下面只是一个示例
    #--------------------------------------------------#
    model.load_weights("I:\lunwen\deep3\logs\ep251-loss0.012-val_loss0.141.h5")

    # 打开文件夹 验证集
    image_ids = open(r"I:\lunwen\deep3\dataset2\datasets\day\val.txt", 'r').read().splitlines()
    #--------------------------------------------------#
    #   对imgs文件夹进行一个遍历
    #--------------------------------------------------#

    # imgs = os.listdir("./img/")
    for image_id in tqdm(image_ids):
        #--------------------------------------------------#
        #   打开imgs文件夹里面的每一个图片
        #--------------------------------------------------#
        image_path = r"I:\lunwen\deep3\dataset2\datasets\day\jpg/" + image_id + ".jpg"
        img = Image.open(image_path)
        # img = Image.open("./img/"+jpg)
        
        old_img = copy.deepcopy(img)
        orininal_h = np.array(img).shape[0]
        orininal_w = np.array(img).shape[1]

        #--------------------------------------------------#
        #   对输入进来的每一个图片进行Resize
        #   resize成[HEIGHT, WIDTH, 3]
        #--------------------------------------------------#
        img = img.resize((WIDTH,HEIGHT), Image.BICUBIC)
        img = np.array(img) / 255
        img = img.reshape(-1, HEIGHT, WIDTH, 3)

        #--------------------------------------------------#
        #   将图像输入到网络当中进行预测
        #--------------------------------------------------#
        pr = model.predict(img)[0]
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        # pr = pr.argmax(axis=-1).reshape([self.model_image_size[0], self.model_image_size[1]])
        pr = pr.argmax(axis=-1).reshape((int(HEIGHT), int(WIDTH)))

        #------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        #------------------------------------------------#
        seg_img = np.zeros((int(HEIGHT), int(WIDTH), 3))
        for c in range(NCLASSES):
            seg_img[:, :, 0] += ((pr[:,: ] == c) * class_colors[c][0]).astype('uint8')
            seg_img[:, :, 1] += ((pr[:,: ] == c) * class_colors[c][1]).astype('uint8')
            seg_img[:, :, 2] += ((pr[:,: ] == c) * class_colors[c][2]).astype('uint8')

        #------------------------------------------------#
        #   将新图片转换成Image的形式
        #------------------------------------------------#
        seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        #------------------------------------------------#
        #   将新图片和原图片混合
        #------------------------------------------------#
        # image = Image.blend(old_img,seg_img,0.3)
        
        seg_img.save(r"I:\lunwen\deep3\result_acc0.9704day/"+image_id+'.png')


