"""
VGG16 特征提取相关的工具函数。

提供从单张图片路径提取 relu6 特征的简化接口，供预测和检索使用。
"""

import numpy as np
import tensorflow as tf
from tensorflow_vgg import vgg16, utils


def extract_features_for_image(image_path: str) -> np.ndarray:
    """
    从给定的图片路径加载图像，使用预训练 VGG16 提取 relu6 特征。

    返回值:
        shape 为 (1, feature_dim) 的 numpy 数组。
    """
    img_ready = utils.load_image(image_path)
    img_batch = img_ready.reshape((1, 224, 224, 3))

    with tf.Session() as sess:
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)
        feed_dict = {input_: img_batch}
        codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

    return codes_batch

