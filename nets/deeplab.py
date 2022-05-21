import tensorflow as tf
from functools import wraps
from keras import backend as K
from functools import reduce
from keras import layers
from keras.activations import relu
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import *
from keras import backend
from keras.models import Model
from keras.utils.data_utils import get_file
from nets.mobilenetV2 import (BasicRFB, _inverted_res_block, squeeze)
from nets.mobilenetV3 import MobileNetV3
from nets.resnet import ResNet50

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def _activation(x, name='relu'):
    if name == 'relu':
        return Activation('relu')(x)
    elif name == 'hardswish':
        return hard_swish(x)

def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])
def hard_sigmoid(x):
    return backend.relu(x + 3.0, max_value=6.0) / 6.0

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

from nets.mobilenetV2 import mobilenetV2

#--------------------------------------------------#
#   单次卷积DarknetConv2D
#   正则化系数为5e-4
#   如果步长为2则自己设定padding方式。
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + Relu6
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Activation(relu6))


def relu6(x):
    return K.relu(x, max_value=6)


# ---------------------------------------------------#
#   深度可分离卷积块
#   DepthwiseConv2D + BatchNormalization + Relu6
# ---------------------------------------------------#
def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha=1,
                          depth_multiplier=1, strides=(1, 1), activation='relu', block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False)(inputs)

    x = BatchNormalization()(x)
    x = _activation(x, activation)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = _activation(x, activation)
    return x



def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, activation='relu', epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    
    if not depth_activation:
        x = _activation(x, activation)

    # 首先使用3x3的深度可分离卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = _activation(x, activation)

    # 利用1x1卷积进行通道数调整
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = _activation(x, activation)

    return x

def Deeplabv3(input_shape=(320, 320, 3), classes=2, alpha=1):
    img_input = Input(shape=input_shape)
    #                           改1  v2               改2  v3
    # skip1                  320 320 64               80 80 24
    # feat1                  160 160 128                40 40 40
    # feat2                  80 80 256                20 20 112
    # feat3                  40 40 512               10 10 160
    # feat4                  20 20 1024
    skip1, feat1, feat2, feat3 = ResNet50(img_input) #, alpha)

                                                                # 改1                                                 改2
    P3 = BasicRFB(feat3, 512, 512, stride=1, name=1)                   #   40 40 512-> 40 40 512
    P3_Up = UpSampling2D(size=(2, 2), name="p3upsampled")(P3)           # 40 40 512--> 80 80 512                   10 10 160 -> 20 20 160
    P2 = BasicRFB(feat2, 256, 256, stride=1,  name=2)                   # 80 80 256 ---> 80 80 256                      20 20 112 ->  20 20 112
    P1 = BasicRFB(feat1, 128, 128, stride=1,  name=3)                   # 160 160 128 ->160 160 128

    P2 = Concatenate()([P2, P3_Up])                                  # 80 80 256 + 512 ---> 80 80 768                   20 20 112 -> 20 20 272

    P2 = _depthwise_conv_block(P2, 256, strides=(1, 1), activation='hardswish')              # 80 80 768 -> 80 80 256

    P2_Up = UpSampling2D(size=(2, 2), name="p2upsampled")(P2)                # 80 80 256 -> 160 160 256                # 20 20 272 -> 40 40 272

    P1 = Concatenate()([P1, P2_Up])                                  # 160 160 256 + 256  ---> 160 160 512              #    40 40 40 -> 40 40 40+ 40 40 272 = 40 40 312
    P1 = _depthwise_conv_block(P1, 256, strides=(1, 1), activation='hardswish')             # 160 160 512  -> 160 160 256
    P1_down = _depthwise_conv_block(P1, 256, strides=(2, 2), activation='hardswish')        #160 160 256 -> 80 80 256                      # 40 40 312  -> 20 20 312

    P2 = Concatenate()([P2, P1_down])                  #     80 80 256+256 -> 80 80 512                           #   20 20 272 -> 20 20 312 + 20 20 272 = 20 20 584
    P2  = _depthwise_conv_block(P2, 256, strides=(1, 1), activation='hardswish')        # 80 80 512  -> 80 80 256
    P2_down = _depthwise_conv_block(P2, 256, strides=(2, 2), activation='hardswish')     #80 80 256 -> 40 40 256                               # 20 20 584 -> 10 10 584

    P3 = Concatenate()([P2_down, P3])                   #      40 40 256+512 ->   40 40 768                         # 10 10 584 -> 10 10 584 + 10 10 160 =10 10 724

    P3 = _depthwise_conv_block(P3, 256, strides=(1, 1), activation='hardswish')                 # 40 40 768-> 40 40 256                        #  10 10 416-> 10 10 128
    P3 = _depthwise_conv_block(P3, 256, strides=(1, 1), activation='hardswish')

    # 防止过拟合
    x = Dropout(0.1)(P3)
#                     加skip1
                                                # 40 40 256 -> 320 320 256               10 10 256 -> 80 80 256
    # skip1 = DarknetConv2D_BN_Leaky(24, (1, 1), strides=(2, 2))(skip1)
    x = Lambda(lambda xx: tf.image.resize_images(x, skip1.shape[1:3]))(x)

    # 320 320 64 ---> 320 320 64                # 80 80 24 -> 80 80 48                  # 80 80 24 > 80 80 48
    dec_skip1 = Conv2D(64, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)

    #   # 320 320 64  + 320 320 256 = 320 320 320          # 80 80 256 + 80 80 48 = 80 80 304
    x = Concatenate()([x, dec_skip1])
###############################


    # 320 320 320  -> 320 320 256 -> 80 80 256
    x = SepConv_BN(x, 256, 'decoder_conv0', depth_activation=True, activation='hardswish', epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True, activation='hardswish', epsilon=1e-5)

    # 320 320 256-> 320 320 2 ->  320 320 2
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Conv2D(classes, (1, 1), padding='same')(x)
    x = BatchNormalization(name='feature_projection1_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Lambda(lambda xx:tf.image.resize_images(xx, size_before3[1:3]))(x)

    #---------------------------------------------------#
    #   获得每一个像素点属于每一个类的概率
    #---------------------------------------------------#
    # x = Activation("softmax", name="main")(x)

    x = Reshape((-1,classes))(x)
    x = Softmax()(x)

    model = Model(img_input, x,  name='deeplabv3plus')
    return model

