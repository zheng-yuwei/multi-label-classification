# -*- coding: utf-8 -*-
"""
File multi_label_loss.py
@author:ZhengYuwei
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


class MyLoss(object):
    """ 损失函数 """
    def __init__(self, model, **options):
        self.model = model
        self.is_label_smoothing = options.setdefault('is_label_smoothing', False)
        self.is_focal_loss = options.setdefault('is_focal_loss', False)
        self.is_gradient_harmonizing = options.setdefault('is_gradient_harmonized', False)

        self.loss_func = self._normal_categorical_crossentropy()
        # 标签平滑
        if self.is_label_smoothing:
            self.smoothing_epsilon = options.setdefault('smoothing_epsilon', 0.005)
        # focal loss损失函数
        if self.is_focal_loss:
            gamma = options.setdefault('focal_loss_gamma', 2.0)
            alpha = options.setdefault('focal_loss_alpha', 1.0)
            self.loss_func = self._categorical_focal_loss(gamma, alpha)
        # gradient harmonized mechanism
        if self.is_gradient_harmonizing:
            bins = options.setdefault('ghm_loss_bins', 30)
            momentum = options.setdefault('ghm_loss_momentum', 0.75)
            self.loss_func = self._categorical_ghm_loss(bins, momentum)

    @staticmethod
    def _normal_categorical_crossentropy():
        """ 自带的多标签分类损失函数 categorical_crossentropy """
        def categorical_crossentropy(y_truth, y_pred, _):
            return keras.backend.categorical_crossentropy(y_truth, y_pred)
        return categorical_crossentropy

    @staticmethod
    def _categorical_focal_loss(gamma=2.0, alpha=1.0):
        """ 返回多分类 focal loss 函数
        Formula: loss = -alpha*((1-p_t)^gamma)*log(p_t)
        Parameters:
            alpha -- the same as wighting factor in balanced cross entropy, default 0.25
            gamma -- focusing parameter for modulating factor (1-p), default 2.0
        """
        def focal_loss(y_truth, y_pred, _):
            epsilon = keras.backend.epsilon()
            y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
            cross_entropy = -y_truth * keras.backend.log(y_pred)
            weight = alpha * keras.backend.pow(keras.backend.abs(y_truth - y_pred), gamma)
            loss = weight * cross_entropy
            loss = keras.backend.sum(loss, axis=1)
            return loss
        return focal_loss

    @staticmethod
    def _categorical_ghm_loss(bins=30, momentum=0.75):
        """ 返回多分类 GHM 损失函数：
                把每个区间上的梯度做平均，也就是说把梯度拉平，回推到公式上等价于把loss做平均
        Formula:
            loss = sum(crossentropy_loss(p_i,p*_i) / GD(g_i))
            GD(g) = S_ind(g) / delta = S_ind(g) * M
            S_ind(g) = momentum * S_ind(g) + (1 - momentum) * R_ind(g)
            R_ind(g)是 g=|p-p*| 所在梯度区间[(i-1)delta, i*delta]的样本数
            M = 1/delta，这个是个常数，理论上去掉只有步长影响
        Parameters: （论文默认）
            bins -- 区间个数，default 30
            momentum -- 使用移动平均来求区间内样本数，动量部分系数，论文说不敏感
        """
        # 区间边界
        edges = np.array([i/bins for i in range(bins + 1)])
        edges = np.expand_dims(np.expand_dims(edges, axis=-1), axis=-1)
        acc_sum = 0
        if momentum > 0:
            acc_sum = tf.zeros(shape=(bins,), dtype=tf.float32)

        def ghm_class_loss(y_truth, y_pred, valid_mask):
            epsilon = keras.backend.epsilon()
            y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
            # 0. 计算本次mini-batch的梯度分布：R_ind(g)
            gradient = keras.backend.abs(y_truth - y_pred)
            # 获取概率最大的类别下标，将该类别的梯度做为该标签的梯度代表
            # 没有这部分就是每个类别的梯度都参与到GHM，实验表明没有这部分会更好些
            # truth_indices_1 = keras.backend.expand_dims(keras.backend.argmax(y_truth, axis=1))
            # truth_indices_0 = keras.backend.expand_dims(keras.backend.arange(start=0,
            #                                                                  stop=tf.shape(y_pred)[0],
            #                                                                  step=1, dtype='int64'))
            # truth_indices = keras.backend.concatenate([truth_indices_0, truth_indices_1])
            # main_gradient = tf.gather_nd(gradient, truth_indices)
            # gradient = tf.tile(tf.expand_dims(main_gradient, axis=-1), [1, y_pred.shape[1]])
            
            # 求解各个梯度所在的区间，并落到对应区间内进行密度计数
            grads_bin = tf.logical_and(tf.greater_equal(gradient, edges[:-1, :, :]), tf.less(gradient, edges[1:, :, :]))
            valid_bin = tf.boolean_mask(grads_bin, valid_mask, name='valid_gradient', axis=1)
            valid_bin = tf.reduce_sum(tf.cast(valid_bin, dtype=tf.float32), axis=(1, 2))
            # 2. 更新指数移动平均后的梯度分布：S_ind(g)
            nonlocal acc_sum
            acc_sum = tf.add(momentum * acc_sum, (1 - momentum) * valid_bin, name='update_bin_number')
            # sample_num = tf.reduce_sum(acc_sum)  # 是否乘以总数，乘上效果反而变差了
            # 3. 计算本次mini-batch不同loss对应的梯度密度：GD(g)
            position = tf.slice(tf.where(grads_bin), [0, 1], [-1, 2])
            value = tf.gather_nd(acc_sum, tf.slice(tf.where(grads_bin), [0, 0], [-1, 1]))  # * bins
            grad_density = tf.sparse.SparseTensor(indices=position, values=value,
                                                  dense_shape=tf.shape(gradient, out_type=tf.int64))
            grad_density = tf.sparse.to_dense(grad_density, validate_indices=False)
            grad_density = grad_density * tf.expand_dims(valid_mask, -1) + (1 - tf.expand_dims(valid_mask, -1))

            # 4. 计算本次mini-batch不同样本的损失：loss
            cross_entropy = -y_truth * keras.backend.log(y_pred)
            # loss = cross_entropy / grad_density * sample_num
            loss = cross_entropy / grad_density
            loss = keras.backend.sum(loss, axis=1)
            """
            # 调试用，打印tensor
            print_op = tf.print('acc_sum: ', acc_sum, '\n',
                                'grad_density: ', grad_density, '\n',
                                'cross_entropy: ', cross_entropy, '\n',
                                'loss:', loss, '\n',
                                '\n',
                                '=================================================\n',
                                summarize=100)
            with tf.control_dependencies([print_op]):
                return tf.identity(loss)
            """
            return loss
        return ghm_class_loss

    def categorical_crossentropy(self, y_truth, y_pred):
        """ 单标签多分类损失函数
        :param y_truth: 真实类别值, (?, ?)
        :param y_pred: 预测类别值, (?, num_classes)
        :return: loss
        """
        num_classes = keras.backend.cast(keras.backend.int_shape(y_pred)[-1], dtype=tf.int32)  # 类别数
        # 将sparse的truth输出flatten, 记录无效标签(-1)和有效标签(>=0)位置，后续用于乘以loss
        y_truth = keras.backend.flatten(y_truth)
        valid_mask = 1.0 - tf.cast(tf.less(y_truth, 0), dtype=tf.float32)
        # 转为one_hot
        y_truth = keras.backend.cast(y_truth, dtype=tf.uint8)
        y_truth = keras.backend.one_hot(indices=y_truth, num_classes=num_classes)

        # 标签平滑
        if self.is_label_smoothing:
            num_classes = keras.backend.cast(num_classes, dtype=y_pred.dtype)
            y_truth = (1.0 - self.smoothing_epsilon) * y_truth + self.smoothing_epsilon / num_classes

        loss = self.loss_func(y_truth, y_pred, valid_mask)
        loss = loss * valid_mask
        """
        # 调试用，打印tensor
        print_op = tf.print(
            # 'y_pred: ', y_pred, '\n',
            # 'y_truth: ', y_truth, '\n',
            # 'valid_mask: ', valid_mask, '\n',
            # 'loss:', loss, '\n',
            # 'normal_loss:', self._normal_categorical_crossentropy()(y_truth, y_pred, valid_mask), '\n',
            'layer losses (regularization)', tf.transpose(self.model.losses), '\n',
            'mean loss:', tf.reduce_mean(loss), '\t',
            'sum layer losses:', tf.reduce_sum(tf.transpose(self.model.losses)), '\n',
            '=================================================\n',
            summarize=100
        )
        with tf.control_dependencies([print_op]):
            return tf.identity(loss)
        """
        return loss
