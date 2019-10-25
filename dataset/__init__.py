# -*- coding: utf-8 -*-
"""
File __init__.py
@author: ZhengYuwei
功能：
    构建tf.data.Dataset对象，生成训练集/验证集/测试集，以提供模型训练、测试
    构建方式主要包含:
    1. 直接从label文件中读取信息进行构建（file_util)；
    2. 由label文件读取信息生成tfrecord、读取tfrecord方式，构建数据集（tfrecord_util）；
"""