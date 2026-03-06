# coding=utf-8
"""
预测新的图片

Author: yunhu
Date: 2019/5/19
"""

# 引入顺序是标准库，第三方库，本地模块
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow_vgg import vgg16, utils
from PyQt5 import QtWidgets, QtCore

import ftrain

pre_value = ""
labels_vecs = ['flowerBird', 'human', 'landscape']
labels_vecs = np.array(labels_vecs)
# 用于保存和还原训练的模型参数
saver = tf.train.Saver()


def per_picture():
    """
    从用户输入的图像文件中加载和准备图像数据
    """
    # 图像路径
    img_path = input('Input the path and image name:')
    img_ready = utils.load_image(img_path)
    test_pic_arr = [img_ready.reshape((1, 224, 224, 3))]
    # 将多个数组连接起来
    images = np.concatenate(test_pic_arr)
    return images


def get_image_retrieval_result():
    """
    获取图像预测结果
    """
    global pre_value
    images = per_picture()
    with tf.Session() as sess:
        # 图片预处理，输入到 vgg16 中计算特征值
        vgg = vgg16.Vgg16()
        # 定义神经网络的输入层
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            # 载入 VGG16 模型
            vgg.build(input_)

        feed_dict = {input_: images}
        # 计算特征值
        codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
        # 返回 y 矩阵中最大值的下标，如果是二维的加 1
        pre_value = tf.argmax(ftrain.predicted, 1)
        # 加载训练好的新模型，使用 TensorFlow 的 tf.train.Saver() 对象 saver 从最新的检查点文件中还原（恢复）模型参数
        saver.restore(sess, tf.train.latest_checkpoint(ftrain.MODEL_SAVE_PATH))
        # 计算预测值
        pre_value = sess.run(pre_value, feed_dict={ftrain.inputs_: codes_batch})
        print(pre_value)
        print("The prediction paint is:", labels_vecs[pre_value])
