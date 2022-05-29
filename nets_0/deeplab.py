import tensorflow as tf
from functools import wraps
from functools import reduce


from keras.models import Model
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

def squeeze(inputs):
    input_channels = int(inputs.shape[-1])
    x = GlobalAveragePooling2D()(inputs)

    x = Dense(int(input_channels/4))(x)
    x = Activation(relu6)(x)

    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)

    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x

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

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def relu6(x):
    return K.relu(x, max_value=6)

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

    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = _activation(x, 'hardswish')
    if attention:
        x = squeeze(x)

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
    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation_rate, name=name+'2a')(input_tensor)
    x = BatchNormalization(name=name + '2b')(x)
    if se:
        x = cbam_block(x)
    x = _activation(x, activation)
    return x

def conv2d_bn(x,filters,num_row,num_col,padding='same',stride=1,dilation_rate=1,relu=True):
    x = Conv2D(
        filters, (num_row, num_col),
        strides=(stride,stride),
        padding=padding,
        dilation_rate=(dilation_rate, dilation_rate),
        use_bias=False)(x)
    x = BatchNormalization()(x)
    if relu:
        x = Activation("relu")(x)
    return x

def MACM(x, input_filters, output_filters, stride=1, map_reduce=8, name=1):
    input_filters_div = input_filters // map_reduce

    branch_0 = conv2d_bn(x, input_filters_div * 2, 1, 1, stride=stride)
    branch_0 = SepConv_BN(branch_0, input_filters_div * 2, 'RFB_0_%d'%name, depth_activation=False)

    branch_1 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_1 = SepConv_BN(branch_1, input_filters_div*2, 'RFB_1_1_%d'%name, stride=stride, depth_activation=True)
    branch_1 = SepConv_BN(branch_1, input_filters_div * 2, 'RFB_1_2_%d'%name, rate=3, depth_activation=False, attention=True)

    branch_2 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_2 = conv2d_bn(branch_2, (input_filters_div // 2)*3, 1, 5)
    branch_2 = conv2d_bn(branch_2, input_filters_div*2, 5, 1, stride=stride)
    # branch_2 = SepConv_BN(branch_2, (input_filters_div // 2) * 3, 'RFB_2_1_%d'%name, depth_activation=True, attention=True)
    # branch_2 = SepConv_BN(branch_2, input_filters_div * 2, 'RFB_2_2_%d'%name, stride=stride, depth_activation=True)
    branch_2 = SepConv_BN(branch_2, input_filters_div * 2, 'RFB_2_3_%d'%name, rate=5, depth_activation=False, attention=True)

    branch_3 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_3 = conv2d_bn(branch_3, (input_filters_div // 2) * 3, 1, 7)
    branch_3 = conv2d_bn(branch_3, input_filters_div * 2, 7, 1, stride=stride)
    branch_3 = SepConv_BN(branch_3, input_filters_div * 2, 'RFB_3_1_%d'%name, rate=7, depth_activation=False, attention=True)

    out = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    out = conv2d_bn(out, output_filters, 1, 1, relu=False)

    short = conv2d_bn(x, output_filters, 1, 1, stride=stride, relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out, short])
    out = Activation("relu")(out)
    return out

def MASegCloud(input_shape=(320, 320, 3), classes=2, alpha=1):
    img_input = Input(shape=input_shape)
    skip1, feat1, feat2, feat3 = ResNet50(img_input) #, alpha)

    P3 = MACM(feat3, 512, 512, stride=1, name=1)
    P3_Up = UpSampling2D(size=(2, 2), name="p3upsampled")(P3)

    P2 = MACM(feat2, 256, 256, stride=1,  name=2)
    P1 = MACM(feat1, 128, 128, stride=1,  name=3)
    P2 = Concatenate()([P2, P3_Up])

    P2 = cov_rate(P2, 256, 3, dilation_rate=(1, 1), name='re_chane_1', activation='hardswish', se=True)
    P2 = cov_rate(P2, 256, 3, dilation_rate=(2, 2), name='re_chane_2', activation='hardswish', se=True)

    P2_Up = UpSampling2D(size=(2, 2), name="p2upsampled")(P2)

    P1 = Concatenate()([P1, P2_Up])
    P1 = cov_rate(P1, 256, 3, dilation_rate=(4, 4), name='re_chane_3', activation='hardswish', se=True)
    P1 = cov_rate(P1, 256, 3, dilation_rate=(1, 1), name='re_chane_4', activation='hardswish', se=True)

    P1_down = _depthwise_conv_block(P1, 256, strides=(2, 2))
    P2 = Concatenate()([P2, P1_down])

    P2 = cov_rate(P2, 256, 3, dilation_rate=(2, 2), name='re_chane_5', activation='hardswish', se=True)
    P2 = cov_rate(P2, 256, 3, dilation_rate=(4, 4), name='re_chane_6', activation='hardswish', se=True)
    P2_down = _depthwise_conv_block(P2, 256, strides=(2, 2))

    P3 = Concatenate()([P2_down, P3])

    P3 = cov_rate(P3, 256, 3, dilation_rate=(1, 1), name='re_chane_7', activation='hardswish', se=True)
    P3 = cov_rate(P3, 256, 3, dilation_rate=(2, 2), name='re_chane_8', activation='hardswish', se=True)

    # 防止过拟合
    x = Dropout(0.1)(P3)

    x = Lambda(lambda xx: tf.image.resize_images(x, skip1.shape[1:3]))(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)

    x = Concatenate()([x, dec_skip1])

    x = SepConv_BN(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Conv2D(classes, (1, 1), padding='same')(x)
    x = BatchNormalization(name='feature_projection1_BN', epsilon=1e-5)(x)
    x = _activation(x, 'hardswish')
    x = Lambda(lambda xx:tf.image.resize_images(xx, size_before3[1:3]))(x)

    x = Reshape((-1,classes))(x)
    x = Softmax()(x)
    model = Model(img_input, x,  name='MASegCloud')
    return model

