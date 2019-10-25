# -*- coding: utf-8 -*-
"""
File dataset_util.py
@author:ZhengYuwei
功能：
功能：
1. DatasetUtil.augment_image
    对传入的image构成的tf.data.Dataset数据集，进行图像数据增强，包含：
    - 等概率加噪：高斯噪声、椒盐噪声、不加噪声；
    - 对 对比度、亮度、饱和度 进行一定范围的随机扰动
    - mixup（待添加及实验）
    - 图像平移（当前场景不适用）
    - 图像旋转和翻转（当前场景不适用）
    - 随机crop（当前场景不适用）
2. DatasetUtil.shuffle_repeat：随机扰乱数据，重复（整个数据集层面）生成数据；
3. DatasetUtil.batch_prefetch：预生成批次数据。
"""
import tensorflow as tf


class DatasetUtil(object):
    """ 对传入的生成（image, label)形式的tf.data.Dataset数据集，加工为可供训练使用的数据集 """
    # 数据增强的超参，这个可能需要先不使用数据增强训练，调整超参，然后再用数据增强训练对比，然后调节这些超参
    _random_brightness = 30. / 255.  # 随机亮度
    _random_low_contrast = 0.9  # 对比度最低值
    _random_up_contrast = 1.1  # 对比度最大值
    _random_low_saturation = 0.9  # 饱和度最小值
    _random_up_saturation = 1.1  # 饱和度最大值
    _random_normal = 0.01  # 随机噪声

    @staticmethod
    def _add_gauss_noise(image):
        """ 加入高斯噪声 """
        image = image + tf.cast(tf.random_normal(tf.shape(image), mean=0, stddev=DatasetUtil._random_normal),
                                tf.float32)
        return image

    @staticmethod
    def _add_salt_pepper_noise(image):
        """ 加入椒盐噪声 """
        shp = tf.shape(image)[:-1]
        mask_select = tf.keras.backend.random_binomial(shape=shp, p=DatasetUtil._random_normal)
        mask_noise = tf.keras.backend.random_binomial(shape=shp, p=0.5)  # 同样概率的椒盐
        image = image * tf.expand_dims(1 - mask_select, -1) + tf.expand_dims(mask_noise * mask_select, -1)
        return image

    @staticmethod
    def _add_noise(image):
        """ 对图片进行数据增强：高斯噪声或椒盐噪声 """
        # 噪声类型
        noise_type = tf.random_uniform([], minval=0, maxval=3, dtype=tf.int32)
        image = tf.case(pred_fn_pairs=[(tf.equal(noise_type, 0),
                                        lambda: DatasetUtil._add_salt_pepper_noise(image)),
                                       (tf.equal(noise_type, 1),
                                        lambda: DatasetUtil._add_gauss_noise(image))],
                        default=lambda: image)
        return image

    @staticmethod
    def _augment_cond_0(image):
        """ 对图片进行数据增强：亮度，饱和度，对比度 """
        image = tf.image.random_brightness(image, max_delta=DatasetUtil._random_brightness)
        image = tf.image.random_saturation(image, lower=DatasetUtil._random_low_saturation,
                                           upper=DatasetUtil._random_up_saturation)
        image = tf.image.random_contrast(image, lower=DatasetUtil._random_low_contrast,
                                         upper=DatasetUtil._random_up_contrast)
        return image
    
    @staticmethod
    def _augment_cond_1(image):
        """ 对图片进行数据增强：饱和度，亮度，对比度 """
        image = tf.image.random_saturation(image, lower=DatasetUtil._random_low_saturation,
                                           upper=DatasetUtil._random_up_saturation)
        image = tf.image.random_brightness(image, max_delta=DatasetUtil._random_brightness)
        image = tf.image.random_contrast(image, lower=DatasetUtil._random_low_contrast,
                                         upper=DatasetUtil._random_up_contrast)
        return image
    
    @staticmethod
    def _augment_cond_2(image):
        """ 对图片进行数据增强：饱和度，对比度, 亮度 """
        image = tf.image.random_saturation(image, lower=DatasetUtil._random_low_saturation,
                                           upper=DatasetUtil._random_up_saturation)
        image = tf.image.random_contrast(image, lower=DatasetUtil._random_low_contrast,
                                         upper=DatasetUtil._random_up_contrast)
        image = tf.image.random_brightness(image, max_delta=DatasetUtil._random_brightness)
        return image
    
    @staticmethod
    def _augment(image):
        """ 对图片进行数据增强：饱和度，对比度, 亮度，加噪
        :param image: 待增强图片 (H, W, ?)
        :return:
        """
        image = DatasetUtil._add_noise(image)
        # 数据增强顺序
        color_ordering = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)
        image = tf.case(pred_fn_pairs=[(tf.equal(color_ordering, 0),
                                        lambda: DatasetUtil._augment_cond_0(image)),
                                       (tf.equal(color_ordering, 1),
                                        lambda: DatasetUtil._augment_cond_1(image)),
                                       (tf.equal(color_ordering, 2),
                                        lambda: DatasetUtil._augment_cond_2(image))],
                        default=lambda: image)
        image = tf.clip_by_value(image, 0.0, 1.0)  # 防止数据增强越界
        return image
    
    @staticmethod
    def augment_image(image_set):
        """ 对传入的tf.data.Dataset数据集进行图片数据增强，构造批次
        :param image_set: tf.data.Dataset数据集，产生(image, label)形式的数据
        :return: 增强后的tf.data.Dataset对象
        """
        # 进行数据增强（这个map需要在repeat之后，才能每次repeat都进行不一样的增强效果）
        image_set = image_set.map(lambda image: DatasetUtil._augment(image),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return image_set

    @staticmethod
    def shuffle_repeat(dataset, batch_size):
        """ 对传入的tf.data.Dataset数据集进行 shuffle 和 repeat
        :param dataset: tf.data.Dataset数据集，产生(image, label)形式的数据
        :param batch_size: 训练的batch大小
        :return: shuffle 和 repeat后的tf.data.Dataset对象
        """
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=5 * batch_size))
        return dataset

    @staticmethod
    def batch_prefetch(dataset, batch_size):
        """ 生成批次并预加载
        :param dataset: tf.data.Dataset数据集
        :param batch_size: 训练的batch大小
        :return: 输出批次并预加载的tf.data.Dataset数据集
        """
        # 缓存数据到内存
        # dataset = dataset.cache()
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
