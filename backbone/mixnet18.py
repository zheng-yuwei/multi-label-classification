# -*- coding: utf-8 -*-
"""
File mixnet.py
@author: ZhengYuwei
"""
import numpy as np
from tensorflow import keras
from backbone.basic_backbone import BasicBackbone


class MixNet18(BasicBackbone):
    """
    MixNet 18:不是论文中的MixNet的结构
    只是借鉴了MixNet的不同kernel size mix到一起的想法，同时也使用depthwise
    不使用depthwise的话，就是resnext 18了
    """
    
    MIX_KERNEL_SIZES = [(3, 3), (5, 5), (7, 7), (9, 9)]
    MIX_KERNEL_RATIO = np.array([0, 8, 4, 2, 2], dtype=np.float)
    MIX_KERNEL_RATIO = MIX_KERNEL_RATIO.cumsum() / MIX_KERNEL_RATIO.sum()
    
    @classmethod
    def _mix_residual_block(cls, input_x, filters, is_nin=True, **conv_params):
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
        
        mix_residuals = list()
        mix_kernel_nums = filters * cls.MIX_KERNEL_RATIO
        mix_kernel_nums = mix_kernel_nums.astype(np.int)
        
        for i, kernel_size in enumerate(cls.MIX_KERNEL_SIZES):
            mix_residual = keras.layers.Lambda(lambda x: x[:, :, :, mix_kernel_nums[i]:mix_kernel_nums[i+1]])(residual)
            mix_conv = cls.depthwise_conv_bn(mix_residual, kernel_size=kernel_size)
            mix_residuals.append(mix_conv)
        mix_residuals = keras.layers.concatenate(inputs=mix_residuals, axis=cls.CHANNEL_AXIS)
        identity = cls.element_wise_add(input_x, mix_residuals, is_nin=is_nin)
        identity = cls.activation(identity)
        return identity
    
    @classmethod
    def _mix_residual_module(cls, input_x, filters, **conv_params):
        """
        一个残差模块：
        input-> conv+bn->relu-> conv+bn-> add->relu-> conv+bn->relu-> conv+bn-> add -> relu
             |-----> conv(1 X 1)+bn ----->|        |--------------------------->|
        :param input_x: 该残差块的输入
        :param filters: 卷积核数，残差运算后的channel数
        :param conv_params: 卷积参数，参见 BasicBackbone.convolution
        :return:
        """
        first_block = cls._mix_residual_block(input_x, filters, is_nin=True, **conv_params)
        second_block = cls._mix_residual_block(first_block, filters, is_nin=False)
        return second_block
    
    @classmethod
    def build(cls, input_x):
        """
        构造mixnet18基础网络，接受layers.Input，卷积层+BN层+add层+activation层输出，tf维度为 NHWC
        :param input_x: layers.Input对象
        :return: 卷积层+BN层+add层+activation层输出，tf维度为 NHWC=(N, H/32, W/32, 512)
        """
        net = cls.conv_bn(input_x, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')
        net = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(net)
        net = cls.activation(net)

        # 4 * 残差模块
        net = cls._mix_residual_module(net, filters=64)
        net = cls._mix_residual_module(net, filters=128, strides=(2, 2))
        net = cls._mix_residual_module(net, filters=256, strides=(2, 2))
        net = cls._mix_residual_module(net, filters=512, strides=(2, 2))
        
        return net
