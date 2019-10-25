# -*- coding: utf-8 -*-
"""
File check_label_file.py
@author:ZhengYuwei
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion matrix', is_save=False):
    """ 绘制混淆矩阵
    :param y_true: 正确类别标签
    :param y_pred: 预测类别标签
    :param labels: 类别标签列表
    :param title: 图名
    :param is_save: 是否保存图片
    :return:
    """
    if labels:
        y_true = [labels[int(i)] for i in y_true]
        y_pred = [labels[int(i)] for i in y_pred]
    # 计算混淆矩阵，y轴是true，x轴是predicted
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    conf_matrix_pred_sum = np.sum(conf_matrix, axis=0, keepdims=True).astype(float) + 1e-7
    conf_matrix_percent = conf_matrix / conf_matrix_pred_sum * 100  # 沿y轴的百分比

    annot = np.empty_like(conf_matrix).astype(str)
    nrows, ncols = conf_matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            c = conf_matrix[i, j]
            p = conf_matrix_percent[i, j]
            if i == j:
                s = conf_matrix_pred_sum[0][i]
                # annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
                annot[i, j] = '%.2f%%\n%d' % (p, c)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)

    # 绘制混淆矩阵图
    conf_matrix = pd.DataFrame(conf_matrix, index=labels, columns=labels, dtype='float')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    # Oranges,Oranges_r,YlGnBu,Blues,RdBu, PuRd ...
    sns.heatmap(conf_matrix, annot=annot, fmt='', ax=ax, cmap='YlGnBu',
                annot_kws={"size": 11}, linewidths=0.5)
    # 设置坐标轴
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, fontsize=10)
    ax.xaxis.set_ticks_position('none')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)
    ax.yaxis.set_ticks_position('none')

    plt.title(title, size=18)
    plt.xlabel('Predicted', size=16)
    plt.ylabel('Actual', size=16)
    plt.tight_layout()
    if is_save:
        plt.savefig(os.path.join('.', title+'.jpg'))
    else:
        plt.show()


if __name__ == '__main__':
    y_predict = np.random.randint(low=0, high=10, size=(100,))
    y_truth = np.random.randint(low=0, high=10, size=(100,))
    y_labels = [str(i)+'s' for i in range(10)]
    plot_confusion_matrix(y_truth, y_predict, y_labels)
