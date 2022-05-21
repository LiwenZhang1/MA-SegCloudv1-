from keras import backend
from keras import backend as K
from keras.layers import (Activation, Add, BatchNormalization,
                                     Conv2D, Dense, DepthwiseConv2D,
                                     GlobalAveragePooling2D, Input, Multiply,
                                     Reshape, GlobalMaxPooling2D, Lambda, Permute,
                                     Concatenate, multiply)
from keras.models import Model
from nets.mobilenetV2 import BasicRFB
from keras.activations import relu


def _activation(x, name='relu'):
    if name == 'relu':
        return Activation('relu')(x)
    elif name == 'hardswish':
        return hard_swish(x)


def hard_sigmoid(x):
    return backend.relu(x + 3.0, max_value=6.0) / 6.0


def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#---------------------------------------#
#   激活函数 relu6
#---------------------------------------#
def relu6(x):
    return relu(x, max_value=6)

#################   cbam注意力机制   ###############
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

def _bneck(inputs, expansion, alpha, out_ch, kernel_size, stride, se_ratio, dilation_rate, activation,
           block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    out_channels = _make_divisible(out_ch * alpha, 8)
    exp_size = _make_divisible(in_channels * expansion, 8)
    x = inputs
    prefix = 'expanded_conv/'
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = Conv2D(exp_size,
                   kernel_size=1,
                   padding='same',
                   use_bias=False,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis,
                               name=prefix + 'expand/BatchNorm')(x)
        x = _activation(x, activation)

    x = DepthwiseConv2D(kernel_size,
                        strides=stride,
                        padding='same',
                        dilation_rate=dilation_rate,
                        use_bias=False,
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis,
                           name=prefix + 'depthwise/BatchNorm')(x)
    x = _activation(x, activation)

    if se_ratio:
        ############原始注意力#############
        reduced_ch = _make_divisible(exp_size * se_ratio, 8)
        y = GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(x)
        y = Reshape([1, 1, exp_size], name=prefix + 'reshape')(y)
        y = Conv2D(reduced_ch,
                   kernel_size=1,
                   padding='same',
                   use_bias=True,
                   name=prefix + 'squeeze_excite/Conv')(y)
        y = Activation("relu", name=prefix + 'squeeze_excite/Relu')(y)
        y = Conv2D(exp_size,
                   kernel_size=1,
                   padding='same',
                   use_bias=True,
                   name=prefix + 'squeeze_excite/Conv_1')(y)
        x = Multiply(name=prefix + 'squeeze_excite/Mul')([Activation(hard_sigmoid)(y), x])

    x = Conv2D(out_channels,
               kernel_size=1,
               padding='same',
               use_bias=False,
               name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis,
                           name=prefix + 'project/BatchNorm')(x)

    if in_channels == out_channels and stride == 1:
        x = Add(name=prefix + 'Add')([inputs, x])
    return x


def MobileNetV3(inputs, alpha=1.0, kernel=5, se_ratio=0.25):
    if alpha not in [0.75, 1.0]:
        raise ValueError('Unsupported alpha - `{}` in MobilenetV3, Use 0.75, 1.0.'.format(alpha))
    # 416,416,3 -> 208,208,16                     # 320 320 3   -> 160 160 16
    x = Conv2D(16, kernel_size=3, strides=(2, 2), padding='same',
               use_bias=False,
               name='Conv')(inputs)
    x = BatchNormalization(axis=-1,
                           epsilon=1e-3,
                           momentum=0.999,
                           name='Conv/BatchNorm')(x)
    x = Activation(hard_swish)(x)

    # 208,208,16 -> 208,208,16               # 160 160 16 -> 160 160 16
    x = _bneck(x, 1, 16, alpha, 3, 1, None, 1, 'relu', 0)

    # 208,208,16 -> 104,104,24                # 160 160 16 -> 80 80 24
    x = _bneck(x, 4, 24, alpha, 3, 2, None, 1, 'relu', 1)
    x = _bneck(x, 3, 24, alpha, 3, 1, None, 2, 'relu', 2)
    skip1 = x

    # 104,104,24 -> 52,52,40                  # 80 80 24 -> 40 40 40
    x = _bneck(x, 3, 40, alpha, kernel, 2, None, 1, 'relu', 3)
    x = _bneck(x, 3, 40, alpha, kernel, 1, se_ratio, 2, 'relu', 4)
    x = _bneck(x, 3, 40, alpha, kernel, 1, se_ratio, 4, 'relu', 5)
    feat1 = x
    # 52,52,40 -> 26,26,112                     # 40 40 40 -> 20 20 112
    x = _bneck(x, 6, 80, alpha, 3, 2, None, 1, 'hardswish', 6)
    x = _bneck(x, 2.5, 80, alpha, 3, 1, None, 2, 'hardswish', 7)
    x = _bneck(x, 2.3, 80, alpha, 3, 1, None, 4, 'hardswish', 8)
    # feat1 = x
    #                                            40 40 40 -> 20 20 112
    x = _bneck(x, 2.3, 80, alpha, 3, 1, None, 1, 'hardswish', 9)
    x = _bneck(x, 6, 112, alpha, 3, 1, se_ratio, 2, 'hardswish', 10)
    x = _bneck(x, 6, 112, alpha, 3, 1, se_ratio, 4, 'hardswish', 11)
    feat2 = x

    # 26,26,112 -> 13,13,160                    # 20 20 112  -> 10 10 160
    x = _bneck(x, 6, 160, alpha, kernel, 2, None, 1, 'hardswish', 12)
    x = _bneck(x, 6, 160, alpha, kernel, 1, se_ratio, 2, 'hardswish', 13)
    x = _bneck(x, 6, 160, alpha, kernel, 1, se_ratio, 4, 'hardswish', 14)
    x = BasicRFB(x, 160, 160, name=1)
    feat3 = x

    return skip1, feat1, feat2, feat3
