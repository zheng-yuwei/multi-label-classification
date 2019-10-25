# -*- coding: utf-8 -*-
"""
Created on 2019/7/17
File run.py
@author:ZhengYuwei
"""
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from logging.handlers import RotatingFileHandler

from multi_label.trainer import MultiLabelClassifier
from configs import FLAGS

if FLAGS.mode == 'test':
    tf.enable_eager_execution()
if FLAGS.mode in ('train', 'debug'):
    keras.backend.set_learning_phase(True)
else:
    keras.backend.set_learning_phase(False)
np.random.seed(6)
tf.set_random_seed(800)


def generate_logger(filename, **log_params):
    """
    生成日志记录对象记录日志
    :param filename: 日志文件名称
    :param log_params: 日志参数
    :return:
    """
    level = log_params.setdefault('level', logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(level=level)
    formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s')
    # 定义一个RotatingFileHandler，最多备份3个日志文件，每个日志文件最大1M
    file_handler = RotatingFileHandler(filename, maxBytes=1 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(formatter)
    # 控制台输出
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console)
 
 
def run():
    # gpu模式
    if FLAGS.gpu_mode != MultiLabelClassifier.CPU_MODE:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_gpu
        # tf.device('/gpu:{}'.format(FLAGS.gpu_device))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 按需
        sess = tf.Session(config=config)

        """
        # 添加debug：nan或inf过滤器
        from tensorflow.python import debug as tf_debug
        from tensorflow.python.debug.lib.debug_data import InconvertibleTensorProto
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        
        # nan过滤器
        def has_nan(datum, tensor):
            _ = datum  # Datum metadata is unused in this predicate.
            if isinstance(tensor, InconvertibleTensorProto):
                # Uninitialized tensor doesn't have bad numerical values.
                # Also return False for data types that cannot be represented as numpy
                # arrays.
                return False
            elif (np.issubdtype(tensor.dtype, np.floating) or
                  np.issubdtype(tensor.dtype, np.complex) or
                  np.issubdtype(tensor.dtype, np.integer)):
                return np.any(np.isnan(tensor))
            else:
                return False

        # inf过滤器
        def has_inf(datum, tensor):
            _ = datum  # Datum metadata is unused in this predicate.
            if isinstance(tensor, InconvertibleTensorProto):
                # Uninitialized tensor doesn't have bad numerical values.
                # Also return False for data types that cannot be represented as numpy
                # arrays.
                return False
            elif (np.issubdtype(tensor.dtype, np.floating) or
                  np.issubdtype(tensor.dtype, np.complex) or
                  np.issubdtype(tensor.dtype, np.integer)):
                return np.any(np.isinf(tensor))
            else:
                return False
        
        # 添加过滤器
        sess.add_tensor_filter("has_nan", has_nan)
        sess.add_tensor_filter("has_inf", has_inf)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        """
        keras.backend.set_session(sess)

    generate_logger(filename=FLAGS.log_path)
    logging.info('TensorFlow version: %s', tf.__version__)  # 1.13.1
    logging.info('Keras version: %s', keras.__version__)  # 2.2.4-tf

    classifier = MultiLabelClassifier()

    # 模型训练
    if FLAGS.mode == 'train':
        train_dataset = classifier.prepare_data(FLAGS.train_label_path, FLAGS.train_set_dir, FLAGS.is_augment)
        classifier.train(train_dataset, None)
        logging.info('训练完毕！')

    # 用于测试，
    elif FLAGS.mode == 'test':
        # 测试用单GPU测试，若是多GPU模型，需要先转为单GPU模型，然后再执行测试
        if FLAGS.gpu_num > 1:
            classifier.convert_multi2single()
            logging.info('多GPU训练模型转换单GPU运行模型成功，请使用单GPU测试！')
            return

        total_test, wrong_count, great_total_count, great_wrong_count, great_wrong_records = test_model(classifier)
        logging.info('预测总数：%d\t 错误数：%d', total_test, wrong_count)
        logging.info('大于置信度总数：%d\t 错误数：%d\t 准确率：%f', great_total_count, great_wrong_count,
                     1 - great_wrong_count/(great_total_count + 1e-7))
        # logging.info('错误路径是：\n%s', great_wrong_records)
        logging.info('测试完毕！')

    # 用于调试，查看训练的模型中每一层的输出/梯度
    elif FLAGS.mode == 'debug':
        import cv2
        train_dataset = classifier.prepare_data(FLAGS.train_label_path, FLAGS.train_set_dir, FLAGS.is_augment)
        get_trainable_layers = classifier.get_trainable_layers_func()
        for images, labels in train_dataset:
            cv2.imshow('a', np.array(images[0]))
            cv2.waitKey(1)
            outputs = get_trainable_layers(images)  # 每一个可训练层的输出
            gradients = classifier.get_gradients(images, labels)  # 每一个可训练层的参数梯度
            assert outputs is not None
            assert gradients is not None
            logging.info("=============== debug ================")

    # 将模型保存为pb模型
    elif FLAGS.mode == 'save_pb':
        # 保存模型记得注释eager execution
        classifier.save_mobile()

    # 将模型保存为服务器pb模型
    elif FLAGS.mode == 'save_serving':
        # 保存模型记得注释eager execution
        classifier.save_serving()
    else:
        raise ValueError('Mode Error!')


def test_model(classifier):
    """ 模型测试
    :param classifier: 训练完毕的多标签分类模型
    :return: 总测试样本数, 总错误样本数，大于置信度的总样本数, 大于置信度的错误样本数, 错误样本路径记录
    """
    # import cv2
    # 测试集包含(image, labels, image_path)
    test_set = classifier.prepare_data(FLAGS.test_label_path, FLAGS.test_set_dir, is_augment=False, is_test=True)
    base_conf = FLAGS.base_confidence  # 置信度基线

    # 实际标签，预测标签，预测概率（label数，验证样本数）
    total_test = int(np.ceil(FLAGS.val_set_size / FLAGS.batch_size) * FLAGS.batch_size)
    truth = np.zeros(shape=(len(FLAGS.output_shapes), total_test))
    pred = np.zeros(shape=(len(FLAGS.output_shapes), total_test))
    prob = np.zeros(shape=(len(FLAGS.output_shapes), total_test))
    start_index, end_index = 0, FLAGS.batch_size
    great_wrong_records = list()  # 大于置信度的错误路径集合
    for images, labels, paths in test_set:
        great_wrong_records = np.concatenate((great_wrong_records, np.array(paths)), axis=0)
        truth[:, start_index:end_index] = np.array(labels)
        results = classifier.predict(np.array(images))
        pred[:, start_index:end_index] = np.array([np.argmax(result, axis=-1) for result in results])
        prob[:, start_index:end_index] = np.array([np.max(result, axis=-1) for result in results])
        start_index, end_index = end_index, end_index + FLAGS.batch_size
        logging.info('finish: %d/%d', start_index, total_test)
        if start_index >= total_test:
            break

    # 比较truth和pred，prob和base conf，以统计评价指标
    valid_mask = (truth != -1)  # 有效的待预测位置标记（无效标签/未知类别的在label里真实标签为-1）
    wrong_mask = abs(pred - truth) > 0.5  # 预测错误的位置标记
    great_conf_mask = (prob >= base_conf)  # 预测置信度大于基线的位置标记
    wrong_result = np.any(valid_mask & wrong_mask, axis=0)
    great_conf_result = np.all(~valid_mask | great_conf_mask, axis=0)

    # 总错误数，大于置信度错误数，总大于置信度样本数
    wrong_count = np.sum(wrong_result)
    great_total_count = np.sum(great_conf_result)
    great_wrong_count = np.sum(wrong_result & great_conf_result)
    # 记录大于置信度的预测错误标签
    if np.any(wrong_result & great_conf_result):
        great_wrong_records = [u.decode() for u in great_wrong_records[wrong_result & great_conf_result]]

    # plot_confusion_matrix(truth, pred)
    return total_test, wrong_count, great_total_count, great_wrong_count, great_wrong_records


def plot_confusion_matrix(y_trues, y_preds):
    from utils import draw_tools
    for i in range(y_trues.shape[0]):
        valid_mask = (y_trues[i] != -1)
        draw_tools.plot_confusion_matrix(y_trues[i][valid_mask], y_preds[i][valid_mask],
                                         ['cls_{}'.format(i) for i in range(FLAGS.output_shapes[i])],
                                         FLAGS.output_names[i], is_save=True)
    return


if __name__ == '__main__':
    run()
