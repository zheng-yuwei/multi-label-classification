# -*- coding: utf-8 -*-
"""
File resnet18_v2.py
@author:ZhengYuwei
"""
from tensorflow import keras


def _bn_relu(input_x):
    """ 预激活 """
    bn = keras.layers.BatchNormalization(axis=ResNet18_v2.CHANNEL_AXIS, momentum=ResNet18_v2.BATCH_NORM_DECAY,
                                         gamma_regularizer=keras.regularizers.l2(ResNet18_v2.L2_GAMMA_DECAY),
                                         epsilon=ResNet18_v2.BATCH_NORM_EPSILON)(input_x)
    return keras.layers.Activation(activation='relu')(bn)


def _conv(**conv_params):
    """ 单层卷积
    :param conv_params: 包含参数：
        filters: 整数，输出的channel数
        kernel_size: 卷积核大小，(width, height)
        strides: 步长，(width, height)
        padding: 填充方式
        kernel_initializer: 初始化方式
        kernel_regularizer: 正则化罚项
    """
    filters = conv_params['filters']
    kernel_size = conv_params['kernel_size']
    strides = conv_params.setdefault('strides', (1, 1))
    padding = conv_params.setdefault('padding', 'same')
    kernel_initializer = conv_params.setdefault('kernel_initializer', 'he_normal')
    kernel_regularizer = conv_params.setdefault('kernel_regularizer', keras.regularizers.l2(ResNet18_v2.L2_WEIGHT))

    def f(input_x):
        conv = keras.layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding=padding,
                                   use_bias=False,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer)(input_x)
        return conv

    return f


def _shortcut(identity, residual, is_nin=False):
    """
    合并单位分支和残差分支的shortcut
    :param identity: shortcut的单位量分支
    :param residual: shortcut的残差量分支
    :param is_nin: 是否对单位量实施NIN操作
    :return: 合并结果tensor
    """
    identity_shape = keras.backend.int_shape(identity)
    residual_shape = keras.backend.int_shape(residual)
    stride_width = int(round(identity_shape[ResNet18_v2.ROW_AXIS] / residual_shape[ResNet18_v2.ROW_AXIS]))
    stride_height = int(round(identity_shape[ResNet18_v2.COL_AXIS] / residual_shape[ResNet18_v2.COL_AXIS]))

    if is_nin:
        identity = _conv(filters=residual_shape[ResNet18_v2.CHANNEL_AXIS],
                         kernel_size=(1, 1),
                         strides=(stride_width, stride_height),
                         padding='valid')(identity)

    return keras.layers.add(inputs=[identity, residual])


def residual_v2_module(input_x, **conv_params):
    """
    一个resnet v2残差块：
    input-> bn+relu-> conv-> bn+relu-> conv-> add-> bn+relu-> conv-> bn+relu-> conv-> add->
                   |----> conv(1 X 1)+bn ---->| |------------------------------------->|
    :param input_x: 该残差块的输入
    :param conv_params: 包含参数：
        filters: 整数，输出的channel数
        kernel_size: 卷积核大小，(width, height)
        strides: 步长，(width, height)
        padding: 填充方式
        kernel_initializer: 初始化方式
        kernel_regularizer: 正则化罚项
    :return:
    """
    filters = conv_params['filters']
    kernel_size = conv_params.setdefault('kernel_size', (3, 3))
    strides = conv_params.setdefault('strides', (1, 1))
    padding = conv_params.setdefault('padding', 'same')

    pre_activation = _bn_relu(input_x)
    residual = _conv(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(pre_activation)
    residual = _bn_relu(residual)
    residual = _conv(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding)(residual)
    identity = _shortcut(pre_activation, residual, is_nin=True)

    residual = _bn_relu(identity)
    residual = _conv(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding)(residual)
    residual = _bn_relu(residual)
    residual = _conv(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding)(residual)
    identity = _shortcut(identity, residual, is_nin=False)
    return identity


class ResNet18_v2(object):

    L2_WEIGHT = 5.e-4
    L2_GAMMA_DECAY = 1.e-3
    BATCH_NORM_DECAY = 0.9
    BATCH_NORM_EPSILON = 1e-5
    BATCH_SIZE_AXIS = 0
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3

    @staticmethod
    def build(input_x):
        """
        构造resnet18 v2基础网络，接受layers.Input，输出全1000的连接层
        :param input_x: layers.Input对象
        :return: layers.Dense对象，1000神经元全连接输出
        """
        net = _conv(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_x)
        net = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(net)

        # 4 * 残差模块
        net = residual_v2_module(net, filters=64)
        net = residual_v2_module(net, filters=128, strides=(2, 2))
        net = residual_v2_module(net, filters=256, strides=(2, 2))
        net = residual_v2_module(net, filters=384, strides=(2, 2))
        net = _bn_relu(net)
        
        # 全连接层：先做全局平均池化，然后flatten，然后再全连接层
        net = keras.layers.GlobalAveragePooling2D()(net)
        net = keras.layers.Flatten()(net)
        # net = keras.layers.Dropout(rate=0.5)(net)
        # net = keras.layers.Dense(units=200, activation="relu",
        #                          kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))(net)
        # net = keras.layers.Dropout(rate=0.5)(net)
        return net
