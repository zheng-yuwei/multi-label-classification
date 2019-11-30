# -*- coding: utf-8 -*-
"""
File resnet18.py
@author:ZhengYuwei
注意：
1. resnet v1中，conv层都是有bias的，resnext则没有，resnet v2部分有部分没有；(但，个人感觉可以全部都不要，因为有BN）
2. resnet v2 使用 pre-activation，resnet和resnext不用；（有没有pre-activation其实差不多，inception加强多一些）
3. 18和34层用2层 3x3 conv层的block，50及以上的用3层(1,3,1)conv层、具有bottleneck（4倍差距）的block
"""
from tensorflow import keras
from backbone.basic_backbone import BasicBackbone


class ResNet18(BasicBackbone):
    """ 改动后的ResNet 18，网络前端 7x7卷积->3x3卷积 """
    
    @classmethod
    def _residual_block(cls, input_x, filters, is_nin=True, **conv_params):
        """
        一个残差模块里的 block
        input-> conv+bn->relu-> conv+bn-> add->relu->
             |-----> conv(1 X 1)+bn ------>|
        :param input_x: 残差block的输入
        :param filters: 卷积核数，残差运算后的channel数
        :param is_nin: shortcut是否需要进行NIN运算
        :param conv_params: 卷积参数，参见 BasicBackbone.convolution
        :return: 卷积块运算之后的tensor
        """
        residual = cls.conv_bn(input_x, filters, **conv_params)
        residual = cls.activation(residual)
        conv_params.update(strides=(1, 1))
        residual = cls.conv_bn(residual, filters, **conv_params)
        identity = cls.element_wise_add(input_x, residual, is_nin=is_nin)
        identity = cls.activation(identity)
        return identity
    
    @classmethod
    def _residual_module(cls, input_x, filters, **conv_params):
        """
        一个残差模块：
        input-> conv+bn->relu-> conv+bn-> add->relu-> conv+bn->relu-> conv+bn-> add -> relu
             |-----> conv(1 X 1)+bn ----->|        |--------------------------->|
        :param input_x: 该残差块的输入
        :param filters: 卷积核数，残差运算后的channel数
        :param conv_params: 卷积参数，参见 BasicBackbone.convolution
        :return:
        """
        first_block = cls._residual_block(input_x, filters, is_nin=True, **conv_params)
        second_block = cls._residual_block(first_block, filters, is_nin=False)
        return second_block
    
    @classmethod
    def build(cls, input_x):
        """
        构造resnet18基础网络，接受layers.Input，卷积层+BN层+add层+activation层输出，tf维度为 NHWC
        :param input_x: layers.Input对象
        :return: 卷积层+BN层+add层+activation层输出，tf维度为 NHWC=(N, H/32, W/32, 512)
        """
        net = cls.conv_bn(input_x, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')
        net = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(net)
        net = cls.activation(net)

        # 4 * 残差模块
        net = cls._residual_module(net, filters=64)
        net = cls._residual_module(net, filters=128, strides=(2, 2))
        net = cls._residual_module(net, filters=256, strides=(2, 2))
        net = cls._residual_module(net, filters=512, strides=(2, 2))
        
        return net
