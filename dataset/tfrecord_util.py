# -*- coding: utf-8 -*-
"""
File tfrecord_util.py
@author:ZhengYuwei
功能：
1. TFRecordUtil.generate
    用tf.gfile.FastGFile的read方法读取图片（要先确保shape一致），加上label，制作example，写tfrecord文件
    用tf.gfile.FastGFile读取图片而不是用cv2.imread读取图片，
    因为cv2.imread读取图片会使得存储的tfrecord扩大近8~10倍大小（相比原始jpg图片），而gfile只会增大一点
    另一种是不用tfrecord，而是训练时从图片名列表中读取图片，构造训练数据，速度会慢一点（个人感觉可以忽略）
    在进行这一切之前，确保图片不为空（cv2.imread(file) is not None）
2. TFRecordUtil.get_dataset
    从tfrecord数据集中，构造tf.data.Dataset，解析图片、标签，返回未初始化的迭代器
"""
import os
import logging
import time
import functools
import numpy as np
import tensorflow as tf
import cv2

from dataset.dataset_util import DatasetUtil


class TFRecordUtil(object):
    """ 图片-标签数据集 保存tfrecord，读取tfrecord """
    
    @staticmethod
    def generate(image_paths, labels, tfrecord_path):
        """ 用tf.gfile.FastGFile的read方法读取图片（要先确保shape一致），加上label，制作example，写tfrecord文件
        :param image_paths: 图片路径，list
        :param labels: 对应的标签，list
        :param tfrecord_path: tfrecord文件路径
        """
        if tf.gfile.Exists(tfrecord_path):
            logging.warning('TFRecord数据集(%s)已经存在，不再生成...', tfrecord_path)
            return

        total = len(image_paths)
        if len(labels) != total:
            logging.error('图片路径数量(%d)不等于标签数量(%d)', total, len(labels))
            return
        
        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            for index, [image_path, label] in enumerate(zip(image_paths, labels)):
                if (index + 1) % 1000 == 0:
                    logging.info('\r>> %d/%d done...', index + 1, total)
                if not os.path.exists(image_path):
                    logging.warning('图片不存在：%s', image_path)
                    continue
                # 读取图片，open(image_path, 'rb').read()和tf.read_file(image_path)也同样效果
                image_data = tf.gfile.GFile(image_path, 'rb').read()  # type(image_data)为bytes
                # 多label转化为string
                label = np.asanyarray(label, dtype=np.int).tostring()
                # 制作example，并序列化
                tf_serialized = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
                        })).SerializeToString()
                
                writer.write(tf_serialized)
        return
    
    @staticmethod
    def _parse_tfrecord(serialized_example):
        """ 从序列化的tf.train.example解析出image（归一化）和label
        :param serialized_example: tfrecord读取来的序列化的tf.train.example数据
        :return: 归一化的图片，标签
        """
        example = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            }
        )
        
        image = tf.image.decode_jpeg(example['image'])
        label = tf.decode_raw(example['label'], tf.int32)
        label = tf.cast(label, tf.float32)
        return image, label

    @staticmethod
    def _parse_image(image, _, image_size):
        """
        根据图片路径和标签，读取图片
        :param image: 原始图片rgb数据, Tensor(shape=(原始尺寸), dtype=int)
        :param _: 标签Tensor(shape(?,), dtype=float32))，本函数只产生图像dataset，故不需要
        :param image_size: 图像需要resize到的大小
        :return: 归一化的图片 Tensor(shape=(48, 144, ?), dtype=float32)
        """
        # 图片
        image = tf.image.resize_images(image, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # 这里使用tf.float32会将照片归一化，也就是 *1/255
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.reverse(image, axis=[2])  # 读取的是rgb，需要转为bgr
        return image

    @staticmethod
    def _parse_labels(_, labels, num_labels):
        """
        根据图片路径和标签，解析标签
        :param _: 图片路径, Tensor(shape=(), dtype=int)，本函数只产生标签dataset，故不需要
        :param labels: 标签，Tensor(shape=(?,), dtype=float32)
        :param num_labels: 每个图像对于输出的标签数（多标签分类模型）
        :return: 标签 DatasetV1Adapter<(多个标签Tensor(shape=(), dtype=float32), ...)>
        """
        label_list = list()
        for label_index in range(num_labels):
            label_list.append(labels[label_index])
        return label_list

    @staticmethod
    def get_dataset(tfrecord_path_mode, image_size, num_labels, batch_size, is_augment=True):
        """ 从tfrecord数据集中，构造tf.data，解析图片、标签，返回未初始化的迭代器
        :param tfrecord_path_mode: tfrecord数据集名的模式，使用glob进行匹配
        :param image_size: 图像需要resize到的尺寸
        :param num_labels: 每个图像对于输出的标签数（多标签分类模型）
        :param batch_size: 训练的batch大小
        :param is_augment: 是否进行数据增强
        :return: tf.data.Dataset对象
        """
        logging.info('1. 读取tfrecord文件，生成可初始化迭代器')
        # 获取tfrecord，并进行解析
        tfrecord_path_list = tf.data.Dataset.list_files(tfrecord_path_mode)
        dataset = tf.data.TFRecordDataset(tfrecord_path_list)
        dataset = DatasetUtil.shuffle_repeat(dataset, batch_size)

        logging.info('2. 读取图片数据，构造image set和label set；')
        dataset = dataset.map(TFRecordUtil._parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        image_set = dataset.map(functools.partial(TFRecordUtil._parse_image, image_size=image_size),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        labels_set = dataset.map(functools.partial(TFRecordUtil._parse_labels, num_labels=num_labels),
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if is_augment:
            logging.info('2.1 image set数据增强；')
            image_set = DatasetUtil.augment_image(image_set)

        logging.info('3. image set数据白化；')
        image_set = image_set.map(lambda image: tf.image.per_image_standardization(image),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        logging.info('4. 完成tf.data (image, label) 整体数据集构造，多epoch训练模式；')
        dataset = tf.data.Dataset.zip((image_set, labels_set))

        logging.info('5. 构造tf.data多epoch训练模式；')
        dataset = DatasetUtil.batch_prefetch(dataset, batch_size)
        return dataset


if __name__ == '__main__':
    label_txt_path = './test_sample/label.txt'  # 标签文件
    image_root_dir = './test_sample'  # 图片根目录
    record_filename = './test_sample/{}.record'  # tfrecord存储目录
    # 开启eager模式进行图片读取、增强和展示
    tf.enable_eager_execution()

    # 1. 得到图片路径列表、标签数据列表
    train_image_paths = list()
    train_labels = list()
    with open(label_txt_path, 'r', encoding='UTF-8') as label_file:
        for line in label_file:
            line = line.split(" ")
            train_image_paths.append(os.path.join(image_root_dir, line[0]))
            train_labels.append(line[1:])
            
    # 2. 制作record
    file_names = record_filename.format('test_sample')
    TFRecordUtil.generate(train_image_paths, train_labels, file_names)
    
    # 3. 读取record
    train_batch = 100
    train_set = TFRecordUtil.get_dataset(file_names, image_size=(48, 144), num_labels=10,
                                         batch_size=train_batch, is_augment=True)
    start = time.time()
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
