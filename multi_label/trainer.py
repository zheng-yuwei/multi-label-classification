# -*- coding: utf-8 -*-
"""
File trainer.py
@author:ZhengYuwei
"""
import os
import logging
import tensorflow as tf
from tensorflow import saved_model
from tensorflow import keras

from configs import FLAGS
from dataset.file_util import FileUtil
from multi_label.multi_label_model import Classifier
from multi_label.multi_label_loss import MyLoss


class MultiLabelClassifier(object):
    """
    训练分类器：
    1. 初始化分类器模型、训练参数等；
    2. 调用prepare_data函数准备训练、验证数据集；
    3. 调用train函数训练。
    """

    GPU_MODE = 'gpu'
    CPU_MODE = 'cpu'

    def __init__(self):
        """ 训练初始化 """
        # 构建模型网络
        self.backbone = FLAGS.model_backbone  # 网络类型
        self.input_shape = FLAGS.input_shape
        self.output_shapes = FLAGS.output_shapes

        model = Classifier.build(self.backbone, self.input_shape, self.output_shapes, FLAGS.output_names)
        # 训练模型: cpu，gpu 或 多gpu
        if FLAGS.gpu_mode == MultiLabelClassifier.GPU_MODE and FLAGS.gpu_num > 1:
            self.model = keras.utils.multi_gpu_model(model, gpus=FLAGS.gpu_num)
        else:
            self.model = model
        self.model.summary()
        self.history = None

        # 加载预训练模型（若有）
        self.checkpoint_path = FLAGS.checkpoint_path
        if self.checkpoint_path is None:
            self.checkpoint_path = 'models/'
        if os.path.isfile(self.checkpoint_path):
            if os.path.exists(self.checkpoint_path):
                self.model.load_weights(self.checkpoint_path)
                logging.info('加载模型成功！')
            else:
                self.checkpoint_path = os.path.dirname(self.checkpoint_path)
        if os.path.isdir(self.checkpoint_path):
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            latest = tf.train.latest_checkpoint(self.checkpoint_path)
            if latest is not None:
                self.model.load_weights(latest)
                logging.info('加载模型成功！')
                logging.info(latest)
        else:
            self.checkpoint_path = os.path.dirname(self.checkpoint_path)
        self.checkpoint_path = os.path.join(self.checkpoint_path, FLAGS.checkpoint_name)

        # 设置训练过程中的回调函数
        tensorboard = keras.callbacks.TensorBoard(log_dir=FLAGS.tensorboard_dir)
        cp_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_path, save_weights_only=True,
                                                      verbose=1, period=FLAGS.ckpt_period)
        es_callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=FLAGS.stop_min_delta,
                                                    patience=FLAGS.stop_patience, verbose=0, mode='min')
        lr_callback = keras.callbacks.LearningRateScheduler(FLAGS.lr_func)
        # from utils.logger_callback import NBatchProgbarLogger
        # log_callback = NBatchProgbarLogger(display=FLAGS.logger_batch)
        self.callbacks = [tensorboard, cp_callback, es_callback, lr_callback, ]

        # 设置模型优化方法
        self.loss_function = list()
        for _ in self.output_shapes:
            loss_function = MyLoss(self.model,
                                   is_label_smoothing=FLAGS.is_label_smoothing,
                                   is_focal_loss=FLAGS.is_focal_loss,
                                   is_gradient_harmonized=FLAGS.is_gradient_harmonized).categorical_crossentropy
            self.loss_function.append(loss_function)

        optimizer = keras.optimizers.SGD(lr=FLAGS.init_lr, momentum=0.95, nesterov=True)
        if FLAGS.optimizer == 'adam':
            optimizer = keras.optimizers.Adam(lr=FLAGS.init_lr, amsgrad=True)  # 用AMSGrad
        elif FLAGS.optimizer == 'adabound':
            from keras_adabound import AdaBound
            optimizer = AdaBound(lr=1e-3, final_lr=0.1)
        elif FLAGS.optimizer == 'radam':
            from utils.radam import RAdam
            optimizer = RAdam(lr=1e-3)

        # 由于是多标签分类损失，最终答应的损失信息为：
        # loss: 44.6420 - class_1_loss: 7.8428 - class_2_loss: 5.8357 - class_3_loss: 4.5361 - class_4_loss: 4.7954
        # - class_5_loss: 4.1554 - class_6_loss: 4.6104 - class_7_loss: 5.5645 - class_8_loss: 0.6412
        # - class_9_loss: 0.7639 - class_10_loss: 4.1163
        # class_1_loss等为对应标签的平均样本损失，loss=所有标签平均样本损失 + 权重 * 罚项（正则项）误差(model.losses)
        self.model.compile(optimizer=optimizer, loss=self.loss_function, loss_weights=FLAGS.loss_weights)

        # 设置模型训练参数
        self.mini_batch = FLAGS.batch_size
        self.epoch = FLAGS.epoch

    def prepare_data(self, label_file_path, image_root_dir, is_augment=False, is_test=False):
        """
        数据集准备，返回可初始化迭代器，使用前需要先sess.run(iterator.initializer)进行初始化
        :param label_file_path: 标签文件路径，格式参考 代码具体接口解释
        :param image_root_dir: 图片文件根目录
        :param is_augment: 是否进行数据增强
        :param is_test: 是否为测试阶段
        :return: tf.data.Dataset对象
        """
        logging.info('加载数据集：%s', label_file_path)
        dataset = FileUtil.get_dataset(label_file_path, image_root_dir, image_size=self.input_shape[0:2],
                                       num_labels=len(self.output_shapes), batch_size=self.mini_batch,
                                       is_augment=is_augment, is_test=is_test)
        return dataset

    def train(self, train_set, val_set, train_steps=FLAGS.steps_per_epoch, val_steps=FLAGS.validation_steps):
        """
        使用训练集和验证集进行模型训练
        :param train_set: 训练数据集的tf.data.Dataset对象
        :param val_set: 验证数据集的tf.data.Dataset对象
        :param train_steps: 每个训练epoch的迭代次数
        :param val_steps: 每个验证epoch的迭代次数
        :return:
        """
        if val_set:
            self.history = self.model.fit(train_set, epochs=self.epoch, validation_data=val_set,
                                          steps_per_epoch=train_steps, validation_steps=val_steps,
                                          callbacks=self.callbacks, verbose=2)
        else:
            self.history = self.model.fit(train_set, epochs=self.epoch, steps_per_epoch=train_steps,
                                          callbacks=self.callbacks, verbose=2)
        logging.info('模型训练完毕！')

    def save_serving(self):
        """ 使用TensorFlow Serving时的保存方式：
            serving-save-dir/
                saved_model.pb
                variables/
                    .data & .index
        """
        outputs = dict()
        for index, name in enumerate(FLAGS.output_names):
            outputs[name] = self.model.outputs[index]

        builder = saved_model.builder.SavedModelBuilder(FLAGS.serving_model_dir)
        signature = saved_model.signature_def_utils.predict_signature_def(inputs={'images': self.model.input},
                                                                          outputs=outputs)
        with keras.backend.get_session() as sess:
            builder.add_meta_graph_and_variables(sess=sess,
                                                 tags=[saved_model.tag_constants.SERVING],
                                                 signature_def_map={'predict': signature})
            builder.save()
        logging.info('serving模型保存成功!')

    def save_mobile(self):
        """
        保存模型为pb模型：先转为h5，再保存为pb（没法直接转pb）
        """
        # 获取待保存ckpt文件的文件名
        latest = tf.train.latest_checkpoint(os.path.dirname(self.checkpoint_path))
        model_name = os.path.splitext(os.path.basename(latest))[0]
        if not os.path.exists(FLAGS.pb_model_dir):
            os.makedirs(FLAGS.pb_model_dir)
        # 将整个模型保存为h5（包含图结构和参数），然后再重新加载
        h5_path = os.path.join(FLAGS.pb_model_dir, '{}.h5'.format(model_name))
        self.model.save(h5_path, overwrite=True, include_optimizer=False)
        model = keras.models.load_model(h5_path)
        model.summary()
        # 保存pb
        with keras.backend.get_session() as sess:
            output_names = [out.op.name for out in model.outputs]
            input_graph_def = sess.graph.as_graph_def()
            for node in input_graph_def.node:
                node.device = ""
            graph = tf.graph_util.remove_training_nodes(input_graph_def)
            graph_frozen = tf.graph_util.convert_variables_to_constants(sess, graph, output_names)
            tf.train.write_graph(graph_frozen, FLAGS.pb_model_dir, '{}.pb'.format(model_name), as_text=False)
        logging.info("pb模型保存成功！")

    def evaluate(self, test_set, steps):
        """
        使用测试集进行模型评估
        :param test_set: 测试集的tf.data.Dataset对象
        :param steps: 每一个epoch评估次数
        :return:
        """
        test_loss, test_acc = self.model.evaluate(test_set)
        logging.info('Test accuracy:', test_acc, steps)

    def predict(self, test_images):
        """
        使用测试图片进行模型测试
        :param test_images: 测试图片
        :return:
        """
        predictions = self.model.predict(test_images)
        return predictions

    def get_gradients(self, images, labels, persistent=False):
        """
        在给定输入，获取所有可训练权重向量的梯度向量
        :param images: 输入图像
        :param labels: 标签ground truth
        :param persistent: 是否用持久化的tape，一般不用，除非开启debug模式在该函数内debug
        :return: 获取所有可训练参数的梯度
        """
        with tf.GradientTape(persistent=persistent) as tape:
            y_preds = self.model(images)
            y_truths = labels
            loss = 0
            for y_truth, y_pred in zip(y_truths, y_preds):
                loss += self.loss_function(y_truth, y_pred)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        gradients = [{weight.name: gradient} for gradient, weight in zip(gradients, self.model.trainable_weights)]
        return gradients

    def get_trainable_layers_func(self):
        """
        构造keras函数：在给定输入，获取所有layer的预测结果
        usage:
            classifier = TrainClassifier(backbone=Classifier.BACKBONE_RESNET_18)
            get_trainable_layers = classifier.get_trainable_layers_func()
            outputs = get_trainable_layers(test_images)  # test_images [None, 48, 144, 3]
        :return: 获取所有layers的预测结果的keras函数
        """
        trainable_names = [weight.name for weight in self.model.trainable_weights]
        trainable_names = set([name.split('/')[0] for name in trainable_names])
        trainable_outputs = [{layer.name: layer.output} for layer in self.model.layers
                             if layer.name in trainable_names]
        get_trainable_layers = keras.backend.function(inputs=[self.model.input], outputs=trainable_outputs)
        return get_trainable_layers

    def get_layers_func(self):
        """
        构造keras函数：在给定输入，获取所有layer的预测结果
        usage:
            classifier = TrainClassifier(backbone=Classifier.BACKBONE_RESNET_18)
            get_layers = classifier.get_layer_func()
            outputs = get_layers(test_images)  # test_images [None, 48, 144, 3]
        :return: 获取所有layers的预测结果的keras函数
        """
        layers_output = [layer.output for layer in self.model.layers]
        get_layers = keras.backend.function(inputs=[self.model.input], outputs=layers_output)
        return get_layers

    def convert_multi2single(self):
        """
        将多GPU训练的模型转为单GPU模型，从而可以在单GPU上运行测试
        :return:
        """
        # it's necessary to save the model before use this single GPU model
        multi_model = self.model.layers[FLAGS.gpu_num + 1]  # get single GPU model weights
        dir_name = self.checkpoint_path
        if not os.path.isdir(self.checkpoint_path):
            dir_name = os.path.dirname(self.checkpoint_path)
        latest = tf.train.latest_checkpoint(dir_name)
        save_path = os.path.join(dir_name, 'single_' + os.path.basename(latest))
        multi_model.save_weights(save_path)
