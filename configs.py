# -*- coding: utf-8 -*-
"""
File configs.py
@author:ZhengYuwei
"""
import datetime
import numpy as np
from easydict import EasyDict
from multi_label.multi_label_model import Classifier


def lr_func(epoch):
    # step_epoch = [10, 20, 30, 40, 50, 60, 70, 80]
    # step_lr = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0] # 0.0001
    step_epoch = [10, 140, 200, 260, 300]
    step_lr = [0.00001, 0.001, 0.0001, 0.00001, 0.000001]
    i = 0
    while i < len(step_epoch) and epoch > step_epoch[i]:
        i += 1
    return step_lr[i]


FLAGS = EasyDict()

# 数据集
FLAGS.train_set_dir = 'dataset/test_sample'
FLAGS.train_label_path = 'dataset/test_sample/label.txt'
FLAGS.test_set_dir = 'dataset/test_sample'
FLAGS.test_label_path = 'dataset/test_sample/label.txt'
# 模型权重的L2正则化权重直接写在对应模型的骨干网络定义文件中
FLAGS.input_shape = (48, 144, 3)  # (H, W, C)
FLAGS.output_shapes = (34, 64, 34, 34, 34, 34, 42, 12, 2, 6)  # 多标签输出，每个标签预测的类别数
FLAGS.output_names = ['class_{}'.format(i+1) for i in range(10)]
FLAGS.loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5]
FLAGS.mode = 'train'  # train, test, debug, save_pb, save_serving
FLAGS.model_backbone = Classifier.BACKBONE_RESNET_18
FLAGS.optimizer = 'radam'  # sgdm, adam, adabound, radam
FLAGS.is_augment = True
FLAGS.is_label_smoothing = False
FLAGS.is_focal_loss = False
FLAGS.is_gradient_harmonized = True
FLAGS.type = FLAGS.model_backbone + '-' + FLAGS.optimizer
FLAGS.type += ('-aug' if FLAGS.is_augment else '')
FLAGS.type += ('-smooth' if FLAGS.is_label_smoothing else '')
FLAGS.type += ('-focal' if FLAGS.is_focal_loss else '')
FLAGS.type += ('-ghm' if FLAGS.is_gradient_harmonized else '')
FLAGS.log_path = 'logs/log-{}.txt'.format(FLAGS.type)
# 训练参数
FLAGS.train_set_size = 14  # 160108
FLAGS.val_set_size = 14  # 35935
FLAGS.batch_size = 5  # 3079
FLAGS.steps_per_epoch = int(np.ceil(FLAGS.train_set_size / FLAGS.batch_size))
FLAGS.validation_steps = int(np.ceil(FLAGS.val_set_size / FLAGS.batch_size))

FLAGS.epoch = 300
FLAGS.init_lr = 0.0002  # nadam推荐使用值
# callback的参数
FLAGS.ckpt_period = 20  # 模型保存
FLAGS.stop_patience = 500  # early stop
FLAGS.stop_min_delta = 0.0001
FLAGS.lr_func = lr_func  # 学习率更新函数
# FLAGS.logger_batch = 20  # 打印训练学习的batch间隔
# tensorboard日志保存目录
FLAGS.tensorboard_dir = 'logs/' + 'lpr-{}-{}'.format(FLAGS.type, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
# 模型保存
FLAGS.checkpoint_path = 'models/{}/'.format(FLAGS.type)
FLAGS.checkpoint_name = 'lp-recognition-{}'.format(FLAGS.type) + '-{epoch: 3d}-{loss: .5f}.ckpt'
FLAGS.serving_model_dir = 'models/serving'
FLAGS.pb_model_dir = 'models/pb'
# 测试参数
FLAGS.base_confidence = 0.83  # 基础置信度
# 训练gpu
FLAGS.gpu_mode = 'cpu'
FLAGS.gpu_num = 1
FLAGS.visible_gpu = '0'  # ','.join([str(_) for _ in range(FLAGS.gpu_num)])
FLAGS.gpu_device = '0'
