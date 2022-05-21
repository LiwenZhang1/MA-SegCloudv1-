import tensorflow as tf
from functools import wraps
from keras import backend as K
from functools import reduce


from keras.models import Model
from keras.utils.data_utils import get_file
from nets.mobilenetV2 import (BasicRFB, _inverted_res_block, squeeze)
from nets.mobilenetV3 import MobileNetV3
from nets_0.resnet import ResNet50
from keras.layers import *
from keras import backend

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


################   cbam注意力机制   ###############
def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

##############  空间注意力  #################
def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

###################  通道注意力  #####################
def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

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
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False)(inputs)

    x = BatchNormalization()(x)
    x = _activation(x, 'hardswish')

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1))(x)
    x = BatchNormalization()(x)
    return Activation(relu6)(x)



def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
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
        x = _activation(x, 'hardswish')

    # 首先使用3x3的深度可分离卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = _activation(x, 'hardswish')

    # 利用1x1卷积进行通道数调整
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = _activation(x, 'hardswish')

    return x

def _activation(x, name='relu'):
    if name == 'relu':
        return Activation('relu')(x)
    elif name == 'hardswish':
        return hard_swish(x)
def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])

def hard_sigmoid(x):
    return backend.relu(x + 3.0, max_value=6.0) / 6.0

def cov_rate(input_tensor,filters2,kernel_size,dilation_rate,name,activation,se):
    # -------------------------------#
    #   利用3x3卷积进行特征提取
    # -------------------------------#
    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation_rate, name=name+'2a')(input_tensor)
    x = BatchNormalization(name=name + '2b')(x)
    if se:
        x = cbam_block(x)
    x = _activation(x, activation)
    return x
def Deeplabv3(input_shape=(320, 320, 3), classes=2, alpha=1):
    img_input = Input(shape=input_shape)
    #                           改1  v2               改2  v3
    # skip1                  80 80 64               80 80 24
    # feat1                  80 80 128                40 40 40
    # feat2                  40 40 256                20 20 112
    # feat3                  20 20 512               10 10 160
    skip1, feat1, feat2, feat3 = ResNet50(img_input) #, alpha)
                                                                # 改1                                                 改2
    P3 = BasicRFB(feat3, 512, 512, stride=1, name=1)                   #   20 20 512-> 20 20 512
    print("P3:",K.int_shape(P3))
    P3_Up = UpSampling2D(size=(2, 2), name="p3upsampled")(P3)           # 20 20 512--> 40 40 512                   10 10 160 -> 20 20 160
    print("P3_UP:", K.int_shape(P3_Up))
    P2 = BasicRFB(feat2, 256, 256, stride=1,  name=2)                   # 40 40 256 ---> 40 40 256                      20 20 112 ->  20 20 112
    print("P2:", K.int_shape(P2))
    P1 = BasicRFB(feat1, 128, 128, stride=1,  name=3)                   # 80 80 128 ->80 80 128
    print("P1:", K.int_shape(P1))
    P2 = Concatenate()([P2, P3_Up])                                  # 40 40 256 + 512 ---> 40 40 768                   20 20 112 -> 20 20 272
    print("P2:", K.int_shape(P2))
    P2 = cov_rate(P2, 256, 3, dilation_rate=(1, 1), name='re_chane_1', activation='hardswish', se=True)
    P2 = cov_rate(P2, 256, 3, dilation_rate=(2, 2), name='re_chane_2', activation='hardswish', se=True)
    # P2 = _depthwise_conv_block(P2, 256, strides=(1, 1))              # 40 40 768 -> 40 40 256
    print("P2:", K.int_shape(P2))
    P2_Up = UpSampling2D(size=(2, 2), name="p2upsampled")(P2)                # 40 40 256 -> 80 80 256                # 20 20 272 -> 40 40 272
    print("P2_UP:", K.int_shape(P2_Up))
    P1 = Concatenate()([P1, P2_Up])                                  # 80 80 256 + 256  ---> 80 80 512              #    40 40 40 -> 40 40 40+ 40 40 272 = 40 40 312
    print("P1:", K.int_shape(P1))
    P1 = cov_rate(P1, 256, 3, dilation_rate=(4, 4), name='re_chane_3', activation='hardswish', se=True)
    P1 = cov_rate(P1, 256, 3, dilation_rate=(1, 1), name='re_chane_4', activation='hardswish', se=True)
    # P1 = _depthwise_conv_block(P1, 512, strides=(1, 1))             # 80 80 512  -> 80 80 512
    print("P1:", K.int_shape(P1))
    P1_down = _depthwise_conv_block(P1, 256, strides=(2, 2))        #80 80 512 -> 40 40 512                      # 40 40 312  -> 20 20 312
    print("P1_down:", K.int_shape(P1_down))
    P2 = Concatenate()([P2, P1_down])                  #     40 40 256+256 -> 40 40 256                           #   20 20 272 -> 20 20 312 + 20 20 272 = 20 20 584
    # P2  = _depthwise_conv_block(P2, 256, strides=(1, 1))        # 40 40 584  -> 40 40 256
    print("P2:", K.int_shape(P2))
    P2 = cov_rate(P2, 256, 3, dilation_rate=(2, 2), name='re_chane_5', activation='hardswish', se=True)
    P2 = cov_rate(P2, 256, 3, dilation_rate=(4, 4), name='re_chane_6', activation='hardswish', se=True)
    print("P2:", K.int_shape(P2))
    P2_down = _depthwise_conv_block(P2, 256, strides=(2, 2))     #40 40 256 -> 20 20 256                               # 20 20 584 -> 10 10 584
    print("P2_down:", K.int_shape(P2_down))
    P3 = Concatenate()([P2_down, P3])                   #      20 20 256+512 ->   20 20 768                         # 10 10 584 -> 10 10 584 + 10 10 160 =10 10 724
    print("P3:", K.int_shape(P3))
    # P3 = _depthwise_conv_block(P3, 256, strides=(1, 1))                 # 20 20 768-> 20 20 256                        #  10 10 416-> 10 10 128
    # P3 = _depthwise_conv_block(P3, 256, strides=(1, 1))
    P3 = cov_rate(P3, 256, 3, dilation_rate=(1, 1), name='re_chane_7', activation='hardswish', se=True)
    P3 = cov_rate(P3, 256, 3, dilation_rate=(2, 2), name='re_chane_8', activation='hardswish', se=True)
    print("P3:", K.int_shape(P3))
    # 防止过拟合
    x = Dropout(0.1)(P3)
#                     加skip1
                                                # 20 20 256 -> 80 80 256               10 10 256 -> 80 80 256
    # skip1 = DarknetConv2D_BN_Leaky(24, (1, 1), strides=(2, 2))(skip1)
    x = Lambda(lambda xx: tf.image.resize_images(x, skip1.shape[1:3]))(x)
    print("x:", K.int_shape(x))
    # 160 160 24 ---> 160 160 48                # 80 80 24 -> 80 80 48                  # 80 80 24 > 80 80 48
    dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    print("dec_skip1:", K.int_shape(dec_skip1))
    #   # 80 80 48 + 80 80 256 = 80 80 304          # 80 80 256 + 80 80 48 = 80 80 304
    x = Concatenate()([x, dec_skip1])
    print("x:", K.int_shape(x))
###############################


    # 80 80 304 -> 80 80 256-> 80 80 256
    x = SepConv_BN(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)
    print("x:", K.int_shape(x))
    # 80 80 256-> 80 80 2 ->  320 320 2
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Conv2D(classes, (1, 1), padding='same')(x)
    x = BatchNormalization(name='feature_projection1_BN', epsilon=1e-5)(x)
    x = _activation(x, 'hardswish')
    x = Lambda(lambda xx:tf.image.resize_images(xx, size_before3[1:3]))(x)
    print("x:", K.int_shape(x))
    #---------------------------------------------------#
    #   获得每一个像素点属于每一个类的概率
    #---------------------------------------------------#
    # x = Activation("softmax", name="main")(x)

    x = Reshape((-1,classes))(x)
    x = Softmax()(x)
    print("x:", K.int_shape(x))
    model = Model(img_input, x,  name='deeplabv3plus')
    return model

