# -*- coding: utf-8 -*-
"""
File mobilenet_v2.py
@author:ZhengYuwei
"""
from tensorflow import keras
from backbone.basic_backbone import BasicBackbone


class MobileNetV2(BasicBackbone):
    
    @classmethod
    def _inverted_residual_module(cls, input_x, filters, expand_ratio=6, strides=(2, 2)):
        net = cls._expand_depthwise_linear(input_x, filters, expand_ratio, strides)
        net = cls.element_wise_add(input_x, net, is_nin=False)
        return net
    
    @classmethod
    def _expand_depthwise_linear(cls, input_x, filters, expand_ratio=6, strides=(2, 2)):
        """
        MobileNet v2基本模块：expand、depthwise、linear
        :param input_x: 输入tensor
        :param filters: 模块输出的通道数
        :param expand_ratio: 扩张比例，默认为6
        :param strides: 步长，默认(2, 2)
        :return: 模块运算后的输出tensor
        """
        input_filters = keras.backend.int_shape(input_x)[-1]
        depthwise_filters = expand_ratio * input_filters
        # x6 (1, 1) expand
        net = cls.conv_bn(input_x, filters=depthwise_filters, kernel_size=(1, 1), strides=(1, 1), padding='same')
        net = cls.activation(net)
        # (3, 3) depthwise
        net = cls.depthwise_conv_bn(net, strides=strides)
        net = cls.activation(net)
        # (1, 1) linear bottleneck
        net = cls.conv_bn(net, filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')
        return net
        
    @classmethod
    def build(cls, input_x):
        """
        构建 MobileNet v2网络，整体网络的stride也是32
        :param input_x: 网络输入图形矩阵
        :return: 网络输出tensor
        """
        net = cls.conv_bn(input_x, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')
        net = cls.activation(net)
        
        net = cls._expand_depthwise_linear(net, filters=16, expand_ratio=1, strides=(1, 1))
        
        net = cls._expand_depthwise_linear(net, filters=24, expand_ratio=6, strides=(2, 2))
        net = cls._inverted_residual_module(net, filters=24, expand_ratio=6, strides=(1, 1))

        net = cls._expand_depthwise_linear(net, filters=32, expand_ratio=6, strides=(2, 2))
        net = cls._inverted_residual_module(net, filters=32, expand_ratio=6, strides=(1, 1))
        net = cls._inverted_residual_module(net, filters=32, expand_ratio=6, strides=(1, 1))

        net = cls._expand_depthwise_linear(net, filters=64, expand_ratio=6, strides=(1, 1))
        net = cls._inverted_residual_module(net, filters=64, expand_ratio=6, strides=(1, 1))
        net = cls._inverted_residual_module(net, filters=64, expand_ratio=6, strides=(1, 1))
        net = cls._inverted_residual_module(net, filters=64, expand_ratio=6, strides=(1, 1))

        net = cls._expand_depthwise_linear(net, filters=96, expand_ratio=6, strides=(2, 2))
        net = cls._inverted_residual_module(net, filters=96, expand_ratio=6, strides=(1, 1))
        net = cls._inverted_residual_module(net, filters=96, expand_ratio=6, strides=(1, 1))

        net = cls._expand_depthwise_linear(net, filters=160, expand_ratio=6, strides=(2, 2))
        net = cls._inverted_residual_module(net, filters=160, expand_ratio=6, strides=(1, 1))
        net = cls._inverted_residual_module(net, filters=160, expand_ratio=6, strides=(1, 1))

        net = cls._expand_depthwise_linear(net, filters=320, expand_ratio=6, strides=(1, 1))
        # 原始是1280个channel的输出
        net = cls.conv_bn(net, filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same')
        net = cls.activation(net)
        return net
