from keras import layers
from keras import backend
from keras.activations import relu
from keras.layers import *
from keras.models import Model

#---------------------------------------#
#   通道注意力机制单元
#   利用两次全连接算出每个通道的比重
#---------------------------------------#
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

def BasicRFB(x, input_filters, output_filters, stride=1, map_reduce=8, name=1):
    # -------------------------------------------------------#
    #   BasicRFB模块是一个残差结构
    #   主干部分使用不同膨胀率的卷积进行特征提取
    #   残差边只包含一个调整宽高和通道的1x1卷积
    # -------------------------------------------------------#
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

    # -------------------------------------------------------#
    #   将不同膨胀率的卷积结果进行堆叠
    #   利用1x1卷积调整通道数
    # -------------------------------------------------------#
    out = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    out = conv2d_bn(out, output_filters, 1, 1, relu=False)

    # -------------------------------------------------------#
    #   残差边也需要卷积，才可以相加
    # -------------------------------------------------------#
    short = conv2d_bn(x, output_filters, 1, 1, stride=stride, relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out, short])
    out = Activation("relu")(out)
    return out


##########调节维度##########
def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #-------------------------------#
    #   利用1x1卷积进行通道数的下降
    #-------------------------------#
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用3x3卷积进行特征提取
    #-------------------------------#
    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用1x1卷积进行通道数的上升
    #-------------------------------#
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


################### 加深网络 ##################
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #-------------------------------#
    #   利用1x1卷积进行通道数的下降
    #-------------------------------#
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用3x3卷积进行特征提取
    #-------------------------------#
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用1x1卷积进行通道数的上升
    #-------------------------------#
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    #-------------------------------#
    #   将残差边也进行调整
    #   才可以进行连接
    #-------------------------------#
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
#---------------------------------------#
#   激活函数 relu6
#---------------------------------------#
def relu6(x):
    return relu(x, max_value=6)

#---------------------------------------#
#   利用relu函数乘上x模拟sigmoid
#---------------------------------------#
def hard_swish(x):
    return x * relu(x + 3.0, max_value=6.0) / 6.0

#---------------------------------------#
#   用于判断使用哪个激活函数
#---------------------------------------#
def return_activation(x, activation):
    if activation == 'HS':
        x = Activation(hard_swish)(x)
    if activation == 'RE':
        x = Activation(relu6)(x)
    return x

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1, attention=False):
    in_channels = inputs.shape[-1].value  # inputs._keras_shape[-1]
    pointwise_filters = _make_divisible(int(filters * alpha), 8)
    prefix = 'expanded_conv_{}_'.format(block_id)

    x = inputs
    # 利用1x1卷积进行通道数的扩张
    if block_id:
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # 利用3x3深度可分离卷积提取特征
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)


    #---------------------------------#
    #   引入注意力机制
    #---------------------------------#
    if attention:
        x = cbam_block(x)

    # 利用1x1卷积进行通道数的下降
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    return x

# 空洞卷积
def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3, attention = False):
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
        x = Activation('relu')(x)

    # 首先使用3x3的深度可分离卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    #---------------------------------#
    #   引入注意力机制
    #---------------------------------#
    if attention:
        x = squeeze(x)

    # 利用1x1卷积进行通道数调整
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

# 用于计算padding的大小
def correct_pad(inputs, kernel_size):
    img_dim = 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def mobilenetV2(inputs, alpha=1):

    if alpha not in [0.5, 0.75, 1.0, 1.3]:
        raise ValueError('Unsupported alpha - `{}` in MobilenetV2, Use 0.5, 0.75, 1.0, 1.3'.format(alpha))
    # stem部分
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D(padding=correct_pad(inputs, 3),
                             name='Conv1_pad')(inputs)
    # 512,512,3 -> 256,256,32       # 320 320 3 ---> 160 160 32
    x = Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv1')(x)
    x = BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name='bn_Conv1')(x)
    x = Activation(relu6, name='Conv1_relu')(x)

    # 256,256,32 -> 256,256,16            # 160 160 32 ---> 160 160 16
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False, attention=False)
    # 256,256,16 -> 256,256,24             # 160 160 16  ---> 160 160 24
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=1, skip_connection=False, attention=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True, attention=False)

    skip1 = x

    # 256,256,24 -> 128,128,32          # 160 160 24 ---> 80 80 32
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False, attention=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True, attention=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True, attention=True)
    feat1 = x

    # 128,128,32 -> 64,64,96            # 80 80 24 ---> 40 40 96
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6, skip_connection=False, attention=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True, attention=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True, attention=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=9, skip_connection=True, attention=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=10, skip_connection=False, attention=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=11, skip_connection=True, attention=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=12, skip_connection=True, attention=True)
    feat2 = x

    # 64,64,96 -> 32,32,320             # 40 40 96 ---> 20 20 320
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13, skip_connection=False, attention=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=14, skip_connection=True, attention=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True, attention=True)
    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False, attention=True)
    x = BasicRFB(x, 320, 320, name=1)
    feat3 = x
    return [skip1, feat1, feat2, feat3]
