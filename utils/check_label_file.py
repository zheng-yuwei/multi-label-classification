# -*- coding: utf-8 -*-
"""
File check_label_file.py
@author:ZhengYuwei
"""
import os
import cv2

# 检查标签文件中的图片是否存在且可以打开
train_set_dir = '/home/train_set/images'
train_label_path = '/home/train_set/train.txt'
lines = list()
with open(train_label_path, 'r') as file:
    for line in file:
        img_name = line.strip().split(' ')[0]
        img_path = os.path.join(train_set_dir, img_name)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                lines.append(line)

lines[-1] = lines[-1].strip()
new_train_label_path = os.path.join(os.path.dirname(train_label_path), 'new_train.txt')
with open(new_train_label_path, 'w') as file:
    file.writelines(lines)
