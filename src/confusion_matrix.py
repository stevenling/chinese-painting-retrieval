"""
使用已训练好的分类模型，对 test_photos/ 目录下的测试图片批量进行预测，
并将预测结果填入 3x3 的矩阵 res，用于观察各类别之间的混淆情况。

Author: yunhu
Date: 2019/5/19
"""
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import tensorflow as tf
from PyQt5 import QtWidgets, QtCore
from tensorflow_vgg import utils, vgg16

import ftrain
import config

# 数据来源文件夹
from config import TEST_DATA_DIR as test_data_dir
# 返回指定的文件夹包含的文件或文件夹的名字的列表
contents = os.listdir(test_data_dir)
# classes 最终是一个列表，比如 ['flowerBird', 'human', 'landscape']
# 表示测试集下有这几个类别文件夹。
classes = [each for each in contents if os.path.isdir(test_data_dir + each)]

pre_value = ""
res = np.zeros((3,3))
labels_vecs = ['flowerBird', 'human', 'landscape']
labels_vecs = np.array(labels_vecs)
real_img_url_list = []


def get_img_url_list():
    """
    获取所有图片的完整路径
    """
    test_pic_arr = []
    print(classes)  # ['flowerBird', 'human', 'landscape']
    for each in classes:
        print("Starting {} images".format(each))
        class_path = os.path.join(test_data_dir, each)
        # paint_photos/human
        # 具体的文件名
        files = os.listdir(class_path)
        for i, file in enumerate(files, 1):  # file 人物验证1
            #print(i)
            #print(files)
            # 完整图像路径：使用 config.TEST_IMAGE_PREFIX 拼接
            real_img_url = os.path.join(config.TEST_IMAGE_PREFIX, "test_photos", each, file)
            #print(real_img_url)
            real_img_url_list.append(real_img_url)  # 获取所有图像文件路径 存到 list 中
    print(real_img_url_list)


def per_picture(count):
    """
    根据索引从 real_img_url_list 中取出第 count 张图片，
    加载并预处理后，返回形状为 (1, 224, 224, 3) 的图像数组。
    """
    test_pic_arr = []
    img_ready = utils.load_image(real_img_url_list[count])
    test_pic_arr.append(img_ready.reshape((1, 224, 224, 3)))
    images = np.concatenate(test_pic_arr)
    return images

saver = tf.train.Saver()
i = 0
j = 0


def get_image_retrieval_result():
    """
    使用训练好的模型对测试集中的图片逐张进行预测，
    并将预测结果填入混淆矩阵 res 中，用于统计各类别的分类情况。
    """
    global pre_value
    global i, j
    count = 0
    while count < 9:
        images = per_picture(count)
        count = count + 1
        with tf.Session() as sess:
            vgg = vgg16.Vgg16()
            input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
            with tf.name_scope("content_vgg"):
                vgg.build(input_)
            feed_dict = {input_: images}
            codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
            pre_value = tf.argmax(ftrain.predicted, 1)
            saver.restore(sess, tf.train.latest_checkpoint(ftrain.MODEL_SAVE_PATH))
            pre_value = sess.run(pre_value, feed_dict={ftrain.inputs_: codes_batch})
            if j == 3:
                j = 0
                i = i + 1
            res[i][j] = pre_value + 1
            print(res[i][j])
            # print(labels_vecs[pre_value])
            # print("The prediction paint is:", labels_vecs[pre_value])


def show():
    """
    显示混淆矩阵的情况
    """
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            print(res[i][j])


def main():
    # 获取测试图片集合
    get_img_url_list()  
    get_image_retrieval_result()  
    show()

if __name__ == '__main__':
    main()