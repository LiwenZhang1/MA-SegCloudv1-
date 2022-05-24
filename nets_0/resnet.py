from __future__ import print_function
from keras import backend
import keras.backend as K
from keras import layers
from keras.layers import *


def _activation(x, name='relu'):
    if name == 'relu':
        return Activation('relu')(x)
    elif name == 'hardswish':
        return hard_swish(x)

def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])
def hard_sigmoid(x):
    return backend.relu(x + 3.0, max_value=6.0) / 6.0

def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

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

def identity_block(input_tensor, kernel_size, filters, stage, block, activation, dilation_rate, se):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), padding='same', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = _activation(x, activation)

    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation_rate, name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = _activation(x, activation)

    x = Conv2D(filters3, (1, 1), padding='same', name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    if se:
        x = cbam_block(x)
    x = layers.add([x, input_tensor])
    x = _activation(x, activation)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, activation, dilation_rate, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), padding='same', strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = _activation(x, activation)

    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation_rate, name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), padding='same', name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), padding='same', strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(img_input):
    x = ZeroPadding2D((3, 3))(img_input)

    # 320,320,3 -> 160,160,64
    x = Conv2D(64, (3, 3), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # 160,160,64 -> 80,80,64
    x = Conv2D(64, (3, 3), strides=(2, 2), name='conv2')(x)
    x = BatchNormalization(name='bn_conv2')(x)
    x = Activation('relu')(x)
    skip1 = x

    # 80,80,64 -> 80,80,128
    x = conv_block(x, 3, [64, 64, 128], stage=2, block='a', activation='relu', dilation_rate=(1, 1), strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 128], stage=2, block='b', activation='relu', dilation_rate=(2, 2), se=False)
    x = identity_block(x, 3, [64, 64, 128], stage=2, block='c', activation='relu', dilation_rate=(4, 4), se=True)
    x = identity_block(x, 3, [64, 64, 128], stage=2, block='d', activation='hardswish', dilation_rate=(1, 1), se=True)
    feat1 = x

    # 80,80,128 -> 40,40,256
    x = conv_block(x, 3, [128, 128, 256], stage=3, block='a', activation='hardswish', dilation_rate=(2, 2), strides=(2, 2))
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='b', activation='hardswish', dilation_rate=(4, 4), se=False)
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='c', activation='hardswish', dilation_rate=(1, 1), se=True)
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='d', activation='hardswish', dilation_rate=(2, 2), se=True)
    feat2 = x

    # 40,40,256 -> 20,20,512
    x = conv_block(x, 3, [256, 256, 512], stage=4, block='a', activation='hardswish', dilation_rate=(4, 4), strides=(2, 2))
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='b', activation='hardswish', dilation_rate=(1, 1), se=True)
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='c', activation='hardswish', dilation_rate=(2, 2), se=True)
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='d', activation='hardswish', dilation_rate=(4, 4), se=True)
    feat3 = x

    return skip1, feat1, feat2, feat3


