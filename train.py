import time

import keras
import os
import numpy as np
import cv2
from random import shuffle
from utils.metrics import f_score

from keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                             TensorBoard)
from utils.utils import ModelCheckpoint
from keras.optimizers import Adam,SGD
from keras.utils.data_utils import get_file
from PIL import Image

import tensorflow as tf

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)

from keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

# 修改后的网络
from nets_0.deeplab import MASegCloud


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def letterbox_image(image, label , size):
    label = Image.fromarray(np.array(label))
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    return new_image, new_label



#-------------------------------------------------------------#
#   定义了一个生成器，用于读取datasets2文件夹里面的图片与标签
#-------------------------------------------------------------#
def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            #-------------------------------------#
            #   读取输入图片并进行归一化和resize
            #-------------------------------------#
            name = lines[i].split()[0]
            # 从文件中读取图像
            # img = Image.open("./dataset2/jpg/" + name)

            # 从文件中读取图像
            img = Image.open(r"N:\zhang\deep3\dataset2\datasets\all\jpg" + '/' + name + ".jpg")

            img = img.resize((WIDTH, HEIGHT), Image.BICUBIC)
            img = np.array(img)/255
            X_train.append(img)

            #-------------------------------------#
            #   读取标签图片并进行resize
            #-------------------------------------#

            # name = lines[i].split(';')[1].split()[0]
            # label = Image.open("./dataset2/png/" + name)
            label = Image.open(r"N:\zhang\deep3\dataset2\datasets\all\png" + '/' + name + ".png")
            label = label.resize((int(WIDTH),int(HEIGHT)), Image.NEAREST)
            if len(np.shape(label)) == 3:
                label = np.array(label)[:,:,0]
            label = np.reshape(np.array(label), [-1])
            one_hot_label = np.eye(NCLASSES)[np.array(label, np.int32)]
            Y_train.append(one_hot_label)

            i = (i+1) % n
        yield (np.array(X_train), np.array(Y_train))


if __name__ == "__main__":
    #---------------------------------------------#
    #   定义输入图片的高和宽，以及种类数量
    #---------------------------------------------#
    inputs_size = [320, 320, 3]
    HEIGHT = 320
    WIDTH = 320
    #---------------------------------------------#
    #   背景 + 云 = 2
    #---------------------------------------------#
    NCLASSES = 2

    log_dir = r"N:\zhang\deep3\logs/"
    model = MASegCloud(classes=NCLASSES, input_shape=(HEIGHT, WIDTH, 3))


    weights_path = "N:\zhang\deep3\logs\ep072-loss0.036-val_loss0.092.h5"
    model.load_weights(weights_path,by_name=True,skip_mismatch=True)

    # 打开训练集的txt
    with open(r"N:\zhang\deep3\dataset2\datasets\all\train.txt","r") as f:
        train_lines = f.readlines()

    # 打开验证集的txt
    with open(r"N:\zhang\deep3\dataset2\datasets\all\val.txt","r") as f:
        val_lines = f.readlines()

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                monitor='val_loss', save_weights_only=True, save_best_only=False,verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)


    if True:
        lr = 1e-3
        batch_size = 8
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), batch_size))

        # gen = Generator(batch_size, train_lines, inputs_size, NCLASSES, aux_branch).generate()
        # gen_val = Generator(batch_size, val_lines, inputs_size, NCLASSES, aux_branch).generate(False)

        gen = generate_arrays_from_file(train_lines, batch_size)
        gen_val =generate_arrays_from_file(val_lines, batch_size)

        model.fit_generator(gen,
                steps_per_epoch=max(1, len(train_lines)//batch_size),
                validation_data=gen_val,
                validation_steps=max(1, len(val_lines)//batch_size),
                epochs=300,
                initial_epoch=0,
                callbacks=[checkpoint, tensorboard, early_stopping])     #改 callbacks=[checkpoint, reduce_lr, tensorboard])
        model.save_weights(log_dir+'last1.h5')
