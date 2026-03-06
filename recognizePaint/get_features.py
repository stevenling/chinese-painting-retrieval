# coding=utf-8
"""
获取数据集的特征值，存放在 codes.npy 中

Author: yunhu
Date: 2019/5/19
"""

import os
import numpy as np
import tensorflow as tf

from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

# 接下来我们将 flower_photos 文件夹中的国画图片都载入到进来，并且用图片所在的子文件夹作为标签值。
data_dir = 'paint_photos/' #数据来源文件夹
contents = os.listdir(data_dir)# 返回指定的文件夹包含的文件或文件夹的名字的列表
classes = [each for each in contents if os.path.isdir(data_dir + each)]
#classes ['flowerBird', 'human', 'landscape']

# 利用 vgg16 计算得到特征值
# 首先设置计算 batch 的值，如果运算平台的内存越大，这个值可以设置得越高
batch_size = 5
# 用 codes_list 来存储特征值
codes_list = []
# 用 labels 来存储国画的类别
labels = []
# batch 数组用来临时存储图片数据
batch = []
codes = None


def build_vgg_model():
    """
    构建 VGG16 模型和输入占位符。
    返回值为 (vgg, input_)。
    """
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(input_)
    return vgg, input_


def process_batch(batch_images, input_, vgg, sess, codes_array):
    """
    将当前 batch 的图像送入 VGG16 计算特征，并累积到 codes_array 中。
    返回更新后的 codes_array。
    """
    images = np.concatenate(batch_images)
    feed_dict = {input_: images}
    codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

    if codes_array is None:
        return codes_batch

    return np.concatenate((codes_array, codes_batch))


def extract_features_for_class(class_name, data_dir, batch_size, input_, vgg, sess, codes_array, labels_list):
    """
    对某一类别（class_name）下的所有图片提取特征，
    并将特征累积到 codes_array 中，同时把标签写入 labels_list。
    """
    batch_images = []
    class_path = os.path.join(data_dir, class_name)
    files = os.listdir(class_path)

    for ii, file in enumerate(files, 1):
        img_path = os.path.join(class_path, file)
        img = utils.load_image(img_path)
        batch_images.append(img.reshape((1, 224, 224, 3)))
        labels_list.append(class_name)

        # 如果图片数量到了 batch_size 或者已经是最后一张，则开始计算该 batch 的特征
        if ii % batch_size == 0 or ii == len(files):
            codes_array = process_batch(batch_images, input_, vgg, sess, codes_array)
            batch_images = []
            print('{} images processed for class {}'.format(ii, class_name))

    return codes_array


with tf.Session() as sess:
    vgg, input_ = build_vgg_model()
    # 对每个不同种类的国画分别用 VGG16 计算特征值
    for class_name in classes:
        print("Starting {} images".format(class_name))
        codes = extract_features_for_class(
            class_name,
            data_dir,
            batch_size,
            input_,
            vgg,
            sess,
            codes,
            labels
        )


# 这样我们就可以得到一个 codes 数组，和一个 labels 数组，分别存储了所有国画的特征值和类别。
# 可以用如下的代码将这两个数组保存到硬盘上：
np.save("codes.npy", codes)

import csv

with open('labels', 'w') as f:
    """
    标签写入到 labels 文件中
    """
    # delimiter是分隔符
    writer = csv.writer(f, delimiter='\n') 
    writer.writerow(labels)
    # pickle.dump(labels,f)