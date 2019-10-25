# -*- coding: utf-8 -*-
"""
File logger_callback.py
@author:ZhengYuwei
"""
from tensorflow import keras


class NBatchProgbarLogger(keras.callbacks.ProgbarLogger):
    """ 训练过程中，每N个batch打印log到stdout的回调函数 """

    def __init__(self, count_mode='samples', stateful_metrics=None, display=1000, verbose=1):
        """
        :param count_mode:
        :param stateful_metrics: 打印的metrics
        :param display: batch数打印一次记录
        :param verbose: 是否打印训练logger
        """
        super(NBatchProgbarLogger, self).__init__(count_mode, stateful_metrics)
        self.display = display
        self.display_step = 1
        self.verbose = verbose
        self.epochs = 0

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        """
        # 分布式计算时需要注意
        num_steps = logs.get('num_steps', 1)
        if self.use_steps:
            self.seen += num_steps
        else:
            self.seen += batch_size * num_steps
        """
        self.seen += 1
        self.display_step += 1
        # Skip progbar update for the last batch, will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target and self.display_step % self.display == 0:
            # 打印的metrics
            for k in self.params['metrics']:
                if k in logs:
                    self.log_values.append((k, logs[k]))
            self.progbar.update(self.seen, self.log_values)
