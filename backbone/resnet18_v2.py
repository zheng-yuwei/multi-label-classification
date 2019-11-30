# -*- coding: utf-8 -*-
"""
File resnet18_v2.py
@author:ZhengYuwei
"""
from tensorflow import keras
from backbone.basic_backbone import BasicBackbone


class ResNet18_v2(BasicBackbone):
    """ 改动后的ResNet-v2 18，网络前端 7x7卷积->3x3卷积 """
    
    @classmethod
    def _residual_v2_block(cls, input_x, filters, is_nin=True, **conv_params):
        """
        一个残差模块里的 block
        input-> bn+relu-> conv-> bn+relu-> conv-> add->
                       |----> conv(1 X 1)+bn ---->|
        或
        input-> bn+relu-> conv-> bn+relu-> conv-> add->
           |------------------------------------->|
        :param input_x: 残差block的输入
        :param filters: 卷积核数，残差运算后的channel数
        :param is_nin: shortcut是否需要进行NIN运算
        :param conv_params: 卷积参数，参见 BasicBackbone.convolution
        :return: 卷积块运算之后的tensor
        """
        pre_activation = cls.bn_activation(input_x)
        residual = cls.convolution(pre_activation, filters=filters, **conv_params)
        conv_params.update(strides=(1, 1))
        residual = cls.bn_activation(residual)
        residual = cls.convolution(residual, filters=filters, **conv_params)
        if is_nin:
            identity = cls.element_wise_add(pre_activation, residual, is_nin=True)
        else:
            identity = cls.element_wise_add(input_x, residual, is_nin=False)
        return identity

    @classmethod
    def _residual_v2_module(cls, input_x, filters, **conv_params):
        """
        一个resnet v2残差模块块：
        input-> bn+relu-> conv-> bn+relu-> conv-> add-> bn+relu-> conv-> bn+relu-> conv-> add->
                       |----> conv(1 X 1)+bn ---->| |------------------------------------->|
        :param input_x: 该残差块的输入
        :param filters: 卷积核数，残差运算后的channel数
        :param conv_params: 卷积参数，参见 BasicBackbone.convolution
        :return:
        """
        first_block = cls._residual_v2_block(input_x, filters, is_nin=True, **conv_params)
        second_block = cls._residual_v2_block(first_block, filters, is_nin=False)
        return second_block

    @classmethod
    def build(cls, input_x):
        """
        构造resnet18 v2基础网络，接受layers.Input，卷积层+add层+BN层+activation层输出，tf维度为 NHWC
        :param input_x: layers.Input对象
        :return: 卷积层+BN层+activation层输出，tf维度为 NHWC=(N, H/32, W/32, 512)
        """
        net = cls.convolution(input_x, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')
        net = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(net)

        # 4 * 残差模块
        net = cls._residual_v2_module(net, filters=64)
        net = cls._residual_v2_module(net, filters=128, strides=(2, 2))
        net = cls._residual_v2_module(net, filters=256, strides=(2, 2))
        net = cls._residual_v2_module(net, filters=512, strides=(2, 2))
        net = cls.bn_activation(net)
        
        return net
