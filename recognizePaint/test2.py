"""
使用已训练好的分类模型，对 test_photos/ 目录下的测试图片批量进行预测，
并将预测结果填入 3x3 的矩阵 res，用于观察各类别之间的混淆情况。
"""

# coding=utf-8
import numpy as np
import os
import sys
import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

from PyQt5 import QtWidgets, QtCore
import ftrain

# 测试数据所在的根目录，每个子文件夹对应一个类别
test_data_dir = 'test_photos/'  # 数据来源文件夹
contents = os.listdir(test_data_dir)  # 返回指定的文件夹包含的文件或文件夹的名字的列表
classes = [each for each in contents if os.path.isdir(test_data_dir + each)]

# 当前预测类别索引
pre_value = ""
# 用于保存混淆矩阵，3x3 对应三类国画
res = np.zeros((3, 3))
labels_vecs = ['flowerBird', 'human', 'landscape']
labels_vecs = np.array(labels_vecs)
# 存放所有测试图片的完整路径列表
realImgUrlList = []
# 复用一个 Session 和一个 VGG16 模型实例
sess = tf.Session()
vgg = vgg16.Vgg16()


def get_img_url_list():
    """
    获取 test_photos/ 下所有测试图片的完整路径，
    并保存到全局列表 realImgUrlList 中。
    """
    # print(classes) # ['flowerBird', 'human', 'landscape']
    for each in classes:
        print("Starting {} images".format(each))
        class_path = test_data_dir + each
        # paint_photos/human
        files = os.listdir(class_path)  # 具体的文件名
        preImgUrl = "C://Users/Administrator/PycharmProjects/recognizePaint/"  # 前缀
        for i, file in enumerate(files, 1):  # file 人物验证1
            # test_photos/flowerBird/花鸟验证1.jpg
            imgUrl = class_path + "/" + file
            # 完整的图像路径（绝对路径前缀 + 相对路径）
            realImgUrl = preImgUrl + imgUrl
            realImgUrlList.append(realImgUrl)
    print(realImgUrlList)


def per_picture(count):
    """
    根据下标 count 从 realImgUrlList 中取出一张测试图片，
    加载并预处理为 (1, 224, 224, 3) 的输入张量。
    """
    test_pic_arr = []
    img_ready = utils.load_image(realImgUrlList[count])
    test_pic_arr.append(img_ready.reshape((1, 224, 224, 3)))
    images = np.concatenate(test_pic_arr)
    return images


saver = tf.train.Saver()
i = 0
j = 0


def get_image_retrieval_result():
    """
    使用训练好的分类网络对多张测试图片依次进行预测，
    并将预测结果写入混淆矩阵 res 中。
    """
    global pre_value
    global sess
    global vgg
    global i, j
    count = 0
    # 这里只是示例代码，只对前 9 张测试图片进行预测
    while count < 9:
        images = per_picture(count)
        count = count + 1
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)
        feed_dict = {input_: images}
        codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
        pre_value = tf.argmax(ftrain.predicted, 1)

        # 恢复训练好的分类模型参数
        saver.restore(sess, tf.train.latest_checkpoint(ftrain.MODEL_SAVE_PATH))

        # 计算预测类别索引
        pre_value = sess.run(pre_value, feed_dict={ftrain.inputs_: codes_batch})
        # 按 3x3 矩阵行列填充预测结果
        if j == 3:
            j = 0
            i = i + 1
        res[i][j] = pre_value + 1
        print(res[i][j])


def show():
    """
    打印当前混淆矩阵 res 的每个元素值。
    """
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            print(res[i][j])


def main():
    """
    脚本入口：初始化 Session 和 VGG16 模型，
    获取测试图片列表，执行批量预测并显示混淆矩阵。
    """
    sess = tf.Session()
    vgg = vgg16.Vgg16()
    get_img_url_list()  # 获取测试图片集合
    get_image_retrieval_result()
    show()


if __name__ == '__main__':
    main()