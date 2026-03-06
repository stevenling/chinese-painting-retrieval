"""
加载已经训练好的分类模型，使用 ftrain.py 中划分好的测试集 test_x / test_y
评估模型在测试集上的准确率，并打印结果。
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

import ftrain

# 创建 Saver 用于从 checkpoint 中恢复训练好的模型参数
saver = tf.train.Saver()

with tf.Session() as sess:
    # 从 ftrain.MODEL_SAVE_PATH 恢复最近一次保存的模型
    saver.restore(sess, tf.train.latest_checkpoint(ftrain.MODEL_SAVE_PATH))

    # 使用 ftrain 中预先划分好的测试集 test_x / test_y 计算测试集准确率
    feed = {
        ftrain.inputs_: ftrain.test_x,
        ftrain.labels_: ftrain.test_y,
    }
    test_acc = sess.run(ftrain.accuracy, feed_dict=feed)
    print("Test accuracy: {:.4f}".format(test_acc))
