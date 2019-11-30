# -*- coding: utf-8 -*-
"""
File multi_label_model.py
@author:ZhengYuwei
"""
import logging
from tensorflow import keras
from backbone.resnet18 import ResNet18
from backbone.resnet18_v2 import ResNet18_v2
from backbone.resnext import ResNeXt18
from backbone.mixnet18 import MixNet18
from backbone.mobilenet_v2 import MobileNetV2


class Classifier(object):
    """
    分类器，自定义了多标签的head(多输出keras.models.Model对象)
    """
    BACKBONE_RESNET_18 = 'resnet-18'
    BACKBONE_RESNET_18_V2 = 'resnet-18-v2'
    BACKBONE_RESNEXT_18 = 'resnext-18'
    BACKBONE_MIXNET_18 = 'mixnet-18'
    BACKBONE_MOBILENET_V2 = 'mobilenet-v2'
    BACKBONE_TYPE = {
        BACKBONE_RESNET_18: ResNet18,
        BACKBONE_RESNET_18_V2: ResNet18_v2,
        BACKBONE_RESNEXT_18: ResNeXt18,
        BACKBONE_MOBILENET_V2: MobileNetV2,
        BACKBONE_MIXNET_18: MixNet18
    }

    @classmethod
    def _multi_label_head(cls, net, output_shape, output_names):
        """
        多标签分类器的head，上接全连接输入，下输出多个标签的多分类softmax输出
        :param net: 全连接输入
        :param output_shape: 多标签输出的每个分支的类别数列表
        :param output_names: 多标签输出的每个分支的名字
        :return: keras.models.Model对象
        """
        # 全连接层：先做全局平均池化，然后flatten，然后再全连接层
        net = keras.layers.GlobalAveragePooling2D()(net)
        net = keras.layers.Flatten()(net)
        
        # 不同标签分支
        outputs = list()
        for num, name in zip(output_shape, output_names):
            output = keras.layers.Dense(units=num, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                        activation="softmax", name=name)(net)
            """
            output = keras.layers.Dense(units=num, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                        kernel_regularizer=keras.regularizers.l2(ResNet18.L2_WEIGHT),
                                        bias_regularizer=keras.regularizers.l2(ResNet18.L2_WEIGHT),
                                        activation="softmax", name=name)(net)
            """
            outputs.append(output)
        return outputs

    @classmethod
    def build(cls, backbone, input_shape, output_shape, output_names):
        """
        构建backbone基础网络的多标签分类keras.models.Model对象
        :param backbone: 基础网络，枚举变量 Classifier.NetType
        :param input_shape: 输入尺寸
        :param output_shape: 多标签输出的每个分支的类别数列表
        :param output_names: 多标签输出的每个分支的名字
        :return: resnet18基础网络的多标签分类keras.models.Model对象
        """
        if len(input_shape) != 3:
            raise Exception('模型输入形状必须是3维形式')

        if backbone in cls.BACKBONE_TYPE.keys():
            backbone = cls.BACKBONE_TYPE[backbone]
        else:
            raise ValueError("没有该类型的基础网络！")
        
        if len(input_shape) != 3:
            raise Exception('模型输入形状必须是3维形式')

        logging.info('构造多标签分类模型，基础网络：%s', backbone)
        input_x = keras.layers.Input(shape=input_shape)
        backbone_model = backbone.build(input_x)
        outputs = Classifier._multi_label_head(backbone_model, output_shape, output_names)
        model = keras.models.Model(inputs=input_x, outputs=outputs, name=backbone)
        return model


if __name__ == '__main__':
    """
    可视化网络结构，使用plot_model需要先用conda安装GraphViz、pydotplus
    """
    from configs import FLAGS
    model_names = Classifier.BACKBONE_TYPE.keys()
    for model_name in model_names:
        test_model = Classifier.build(model_name, FLAGS.input_shape, FLAGS.output_shapes, FLAGS.output_names)
        keras.utils.plot_model(test_model, to_file='../images/{}.svg'.format(model_name), show_shapes=True)
        test_model.summary()
