# -*- coding: utf-8 -*-
"""
File resnext.py
@author: ZhengYuwei
"""
import numpy as np
from tensorflow import keras
from backbone.basic_backbone import BasicBackbone


class ResNeXt18(BasicBackbone):
    """
    ResNeXt 18:不是论文中的ResNeXt的结构
    只是借鉴了ResNeXt中 分组卷积 + 不同组不同卷积核大小 的思想
    大概可以看成不使用depthwise的MixNet 18
    """
    
    MIX_KERNEL_SIZES = [(3, 3), (5, 5), (7, 7), (9, 9)]
    # 分为32组，每组理论至少4个channel，不足的话可以把组数减半
    GROUP_NUMS = np.array([16, 8, 4, 4], dtype=np.int)
    SMALL_GROUP_NUMS = GROUP_NUMS // 2
    TOTAL_GROUP_NUMS = np.sum(GROUP_NUMS)
    SMALL_TOTAL_GROUP_NUMS = np.sum(SMALL_GROUP_NUMS)

    @classmethod
    def _inception_residual_block(cls, input_x, filters, is_nin=True, **conv_params):
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
        
        # 每组至少4个channel
        if filters % cls.SMALL_TOTAL_GROUP_NUMS != 0:
            raise ValueError('卷积核数必须可以被组数整除！')
        if filters / cls.SMALL_TOTAL_GROUP_NUMS < 4:
            raise ValueError('卷积核数分组后，每组至少有4个通道！')
        # 判断分为32组还是16组
        group_nums = cls.GROUP_NUMS
        total_group_num = cls.TOTAL_GROUP_NUMS
        if filters % cls.TOTAL_GROUP_NUMS != 0 or filters / cls.TOTAL_GROUP_NUMS < 4:
            group_nums = cls.SMALL_GROUP_NUMS
            total_group_num = cls.SMALL_TOTAL_GROUP_NUMS
        # 分组卷积
        group_channel = filters // total_group_num
        group_residuals = list()
        start_channel = 0
        end_channel = start_channel
        for i, group in enumerate(group_nums):
            for j in range(group):
                end_channel += group_channel
                group_residual = keras.layers.Lambda(lambda x: x[:, :, :, start_channel:end_channel])(residual)
                group_conv = cls.conv_bn(group_residual, filters=group_channel, kernel_size=cls.MIX_KERNEL_SIZES[i])
                group_residuals.append(group_conv)
        group_residuals = keras.layers.concatenate(inputs=group_residuals, axis=cls.CHANNEL_AXIS)
        identity = cls.element_wise_add(input_x, group_residuals, is_nin=is_nin)
        identity = cls.activation(identity)
        return identity

    @classmethod
    def _inception_residual_module(cls, input_x, filters, **conv_params):
        """
        一个残差模块：
        input-> conv+bn->relu-> conv+bn-> add->relu-> conv+bn->relu-> conv+bn-> add -> relu
             |-----> conv(1 X 1)+bn ----->|        |--------------------------->|
        :param input_x: 该残差块的输入
        :param filters: 卷积核数，残差运算后的channel数
        :param conv_params: 卷积参数，参见 BasicBackbone.convolution
        :return:
        """
        first_block = cls._inception_residual_block(input_x, filters, is_nin=True, **conv_params)
        second_block = cls._inception_residual_block(first_block, filters, is_nin=False)
        return second_block

    @classmethod
    def build(cls, input_x):
        """
        构造resnext18基础网络，接受layers.Input，卷积层+BN层+add层+activation层输出，tf维度为 NHWC
        :param input_x: layers.Input对象
        :return: 卷积层+BN层+add层+activation层输出，tf维度为 NHWC=(N, H/32, W/32, 512)
        """
        net = cls.conv_bn(input_x, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')
        net = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(net)
        net = cls.activation(net)

        # 4 * 残差模块
        net = cls._inception_residual_module(net, filters=64)
        net = cls._inception_residual_module(net, filters=128, strides=(2, 2))
        net = cls._inception_residual_module(net, filters=256, strides=(2, 2))
        net = cls._inception_residual_module(net, filters=512, strides=(2, 2))
        
        return net
