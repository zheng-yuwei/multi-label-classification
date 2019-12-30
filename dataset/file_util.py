# -*- coding: utf-8 -*-
"""
File file_util.py
@author:ZhengYuwei
"""
import os
import logging
import functools
import tensorflow as tf

from dataset.dataset_util import DatasetUtil


class FileUtil(object):
    """
    从标签文件中，构造返回(image, label)的tf.data.Dataset数据集
    标签文件内容如下：
    image_name label0,label1,label2,...
    """

    @staticmethod
    def _parse_string_line(string_line, root_path):
        """
        解析文本中的一行字符串行，得到图片路径（拼接图片根目录）和标签
        :param string_line: 文本中的一行字符串，image_name label0 label1 label2 label3 ...
        :param root_path: 图片根目录
        :return: DatasetV1Adapter<(图片路径Tensor(shape=(), dtype=string)，标签Tensor(shape=(?,), dtype=float32))>
        """
        strings = tf.string_split([string_line], delimiter=' ').values
        image_path = tf.string_join([root_path, strings[0]], separator=os.sep)
        labels = tf.string_to_number(strings[1:])
        return image_path, labels
    
    @staticmethod
    def _parse_image(image_path, _, image_size):
        """
        根据图片路径和标签，读取图片
        :param image_path: 图片路径, Tensor(shape=(), dtype=string)
        :param _: 标签Tensor(shape(?,), dtype=float32))，本函数只产生图像dataset，故不需要
        :param image_size: 图像需要resize到的大小
        :return: 归一化的图片 Tensor(shape=(48, 144, ?), dtype=float32)
        """
        # 图片
        image = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize_images(image, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # 这里使用tf.float32会将照片归一化，也就是 *1/255
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.reverse(image, axis=[2])  # 读取的是rgb，需要转为bgr
        return image

    @staticmethod
    def _parse_labels(_, labels, num_labels):
        """
        根据图片路径和标签，解析标签
        :param _: 图片路径, Tensor(shape=(), dtype=string)，本函数只产生标签dataset，故不需要
        :param labels: 标签，Tensor(shape=(?,), dtype=float32)
        :param num_labels: 每个图像对于输出的标签数（多标签分类模型）
        :return: 标签 DatasetV1Adapter<(多个标签Tensor(shape=(), dtype=float32), ...)>
        """
        label_list = list()
        for label_index in range(num_labels):
            label_list.append(labels[label_index])
        return label_list

    @staticmethod
    def get_dataset(file_path, root_path, image_size, num_labels, batch_size, is_augment=True, is_test=False):
        """
        从标签文件读取数据，并解析为（image_path, labels)形式的列表
        标签文件内容格式为：
        image_name label0,label1,label2,label3,...
        :param file_path: 标签文件路径
        :param root_path: 图片路径的根目录，用于和标签文件中的image_name拼接
        :param image_size: 图像需要resize到的尺寸
        :param num_labels: 每个图像对于输出的标签数（多标签分类模型）
        :param batch_size: 批次大小
        :param is_augment: 是否对图片进行数据增强
        :param is_test: 是否为测试阶段，测试阶段的话，输出的dataset中多包含image_path
        :return: tf.data.Dataset对象
        """
        logging.info('利用标签文件、图片根目录生成tf.data数据集对象：')
        logging.info('1. 解析标签文件；')
        dataset = tf.data.TextLineDataset(file_path)
        dataset = DatasetUtil.shuffle_repeat(dataset, batch_size)
        dataset = dataset.map(functools.partial(FileUtil._parse_string_line, root_path=root_path),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        logging.info('2. 读取图片数据，构造image set和label set；')
        image_set = dataset.map(functools.partial(FileUtil._parse_image, image_size=image_size),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        labels_set = dataset.map(functools.partial(FileUtil._parse_labels, num_labels=num_labels),
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if is_augment:
            logging.info('2.1 image set数据增强；')
            image_set = DatasetUtil.augment_image(image_set)

        logging.info('3. image set数据标准化；')
        image_set = image_set.map(lambda image: tf.image.per_image_standardization(image),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if is_test:
            logging.info('4. 完成tf.data (image, label, path) 测试数据集构造；')
            path_set = dataset.map(lambda image_path, label: image_path,
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = tf.data.Dataset.zip((image_set, labels_set, path_set))
        else:
            logging.info('4. 完成tf.data (image, label) 训练数据集构造；')
            # 合并image、labels：
            # DatasetV1Adapter<shapes:((48,144,?), ((), ..., ())), types:(float32,(float32,...,flout32))>
            dataset = tf.data.Dataset.zip((image_set, labels_set))
        logging.info('5. 构造tf.data多epoch训练模式；')
        dataset = DatasetUtil.batch_prefetch(dataset, batch_size)
        return dataset


if __name__ == '__main__':
    import cv2
    import numpy as np
    import time
    
    # 开启eager模式进行图片读取、增强和展示
    tf.enable_eager_execution()
    train_file_path = './test_sample/label.txt'  # 标签文件
    image_root_path = './test_sample'  # 图片根目录
    
    train_batch = 100
    train_set = FileUtil.get_dataset(train_file_path, image_root_path, image_size=(48, 144), num_labels=10,
                                     batch_size=train_batch, is_augment=True)
    start = time.time()
    for count, data in enumerate(train_set):
        for i in range(data[0].shape[0]):
            cv2.imshow('a', np.array(data[0][i]))
            cv2.waitKey(1)

    for count, data in enumerate(train_set):
        print('一批(%d)图像 shape：' % train_batch, data[0].shape)
        for i in range(data[0].shape[0]):
            cv2.imshow('a', np.array(data[0][i]))
            cv2.waitKey(1)
        print('一批(%d)标签 shape：' % train_batch, len(data[1]))
        for i in range(len(data[1])):
            print(data[1][i])
        if count == 100:
            break
    print('耗时：', time.time() - start)
