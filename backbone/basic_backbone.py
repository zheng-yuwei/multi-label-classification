# -*- coding: utf-8 -*-
"""
File basic_backbone.py
@author:ZhengYuwei
"""
from tensorflow import keras


class BasicBackbone(object):
    """ 骨干网络基础类，其他骨干网络类需要继承该类 """
    L2_CONV_DECAY = 5.e-4  # 卷积层W权重衰减系数
    BN_L2_GAMMA_DECAY = 1.e-5  # BN层gamma系数的权重衰减系数
    BN_MOMENTUM = 0.9  # BN层mean、std的指数平滑动量系数
    BN_EPSILON = 1e-5
    BATCH_SIZE_AXIS = 0  # tensorflow backend的维度顺序(N, H, W, C)
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
    
    @classmethod
    def convolution(cls, input_x, filters, **conv_params):
        """
        卷积运算
        :param input_x: 卷积运算的输入
        :param filters: 卷积核数量，输出channel数
        :param conv_params: 可缺省的默认参数：
            kernel_size: 卷积核大小，(width, height)，默认(3, 3)
            strides: 步长，(width, height)，默认(1, 1)
            padding: 填充方式，默认 same
            use_bias: 是否使用偏置b，默认不使用
            kernel_initializer: 卷积核初始化方式，默认 he_normal
            kernel_regularizer: 卷积核正则化项，默认 L2正则化，衰减权重系数为L2_CONV_DECAY
        :return: 卷积运算的输出
        """
        conv_params.setdefault('filters', filters)
        conv_params.setdefault('kernel_size', (3, 3))
        conv_params.setdefault('strides', (1, 1))
        conv_params.setdefault('padding', 'same')
        conv_params.setdefault('use_bias', False)
        conv_params.setdefault('kernel_initializer', 'he_normal')
        conv_params.setdefault('kernel_regularizer', keras.regularizers.l2(cls.L2_CONV_DECAY))
        conv = keras.layers.Conv2D(**conv_params)(input_x)
        return conv
    
    @classmethod
    def depthwise_conv(cls, input_x, **conv_params):
        """
        深度可分离卷积
        :param input_x: 卷积运算的输入
        :param conv_params: 可缺省的默认参数：
            kernel_size: 卷积核大小，(width, height)，默认(3, 3)
            strides: 步长，(width, height)，默认(1, 1)
            padding: 填充方式，默认 same
            use_bias: 是否使用偏置b，默认不使用
            depthwise_initializer: 卷积核初始化方式，默认 he_normal
            depthwise_regularizer: 卷积核正则化项，默认 L2正则化，衰减权重系数为L2_CONV_DECAY
        :return: 深度可分离卷积运算的输出
        """
        conv_params.setdefault('kernel_size', (3, 3))
        conv_params.setdefault('strides', (1, 1))
        conv_params.setdefault('padding', 'same')
        conv_params.setdefault('use_bias', False)
        conv_params.setdefault('depthwise_initializer', 'he_normal')
        conv_params.setdefault('depthwise_regularizer', keras.regularizers.l2(cls.L2_CONV_DECAY))
        conv = keras.layers.DepthwiseConv2D(**conv_params)(input_x)
        return conv
    
    @classmethod
    def batch_normalization(cls, input_x):
        """
        对输入执行batch normalization运算
        :param input_x: 输入tensor
        :return: BN运算后的tensor
        """
        bn = keras.layers.BatchNormalization(axis=cls.CHANNEL_AXIS, momentum=cls.BN_MOMENTUM,
                                             gamma_regularizer=keras.regularizers.l2(cls.BN_L2_GAMMA_DECAY),
                                             epsilon=cls.BN_EPSILON)(input_x)
        return bn
    
    @classmethod
    def activation(cls, input_x, activation='relu', **activation_params):
        """
        激活函数运算
        :param input_x: 输入tensor
        :param activation: 激活函数类型
        :param activation_params: 激活函数参数
        :return: 激活运算后的tensor
        """
        output = keras.layers.Activation(activation=activation, **activation_params)(input_x)
        return output
    
    @classmethod
    def _add_hard_swish(cls):
        """ 添加hard swish作为keras的自定义激活函数 """
        def hard_swish(input_x, max_value=6.):
            """ (x * ReLU6(x+3)) / 6 """
            h_swish = input_x * keras.layers.ReLU(max_value=max_value)(input_x + 3.) / max_value
            return h_swish
        # e.g. keras.layers.Activation(activation = 'h_swish')(5.)
        keras.utils.get_custom_objects().update({'h_swish': keras.layers.Activation(hard_swish)})
    
    @classmethod
    def element_wise_add(cls, identity, residual, is_nin=False):
        """
        逐元素加的合并单位分支和残差分支的运算
        :param identity: shortcut的单位量分支
        :param residual: shortcut的残差量分支
        :param is_nin: 是否对单位量实施NIN卷积操作
        :return: 相加合并结果tensor
        """
        identity_shape = keras.backend.int_shape(identity)
        residual_shape = keras.backend.int_shape(residual)
        stride_width = int(round(identity_shape[cls.ROW_AXIS] / residual_shape[cls.ROW_AXIS]))
        stride_height = int(round(identity_shape[cls.COL_AXIS] / residual_shape[cls.COL_AXIS]))
    
        if is_nin:
            identity = cls.convolution(identity,
                                       filters=residual_shape[cls.CHANNEL_AXIS],
                                       kernel_size=(1, 1),
                                       strides=(stride_width, stride_height),
                                       padding='valid')
            identity = cls.batch_normalization(identity)
            
        merge = keras.layers.add(inputs=[identity, residual])
        return merge

    @classmethod
    def conv_bn(cls, input_x, filters, **conv_params):
        """
        卷积 + 批归一化 运算
        :param input_x: 输入tensor
        :param filters: 卷积核数量，channel数
        :param conv_params: 卷积参数，参见 BasicBackbone.convolution
        :return: 运算后的tensor
        """
        conv = cls.convolution(input_x, filters, **conv_params)
        bn = cls.batch_normalization(conv)
        return bn

    @classmethod
    def depthwise_conv_bn(cls, input_x, **conv_params):
        """
        深度可分离卷积 + 批归一化 运算
        :param input_x: 输入tensor
        :param conv_params: 深度可分离卷积参数，参见 BasicBackbone.depthwise_conv
        :return: 运算后的tensor
        """
        conv = cls.depthwise_conv(input_x, **conv_params)
        bn = cls.batch_normalization(conv)
        return bn

    @classmethod
    def bn_activation(cls, input_x, activation='relu', **activation_params):
        """
        批归一化 + 激活 运算
        :param input_x: 输入tensor
        :param activation: 激活函数类型名称
        :param activation_params: 激活函数参数列表
        :return: 运算后的tensor
        """
        bn = cls.batch_normalization(input_x)
        act = cls.activation(bn, activation=activation, **activation_params)
        return act
