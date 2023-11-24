# coding=utf-8
"""
获取数据集的特征值，存放在 codes.npy 中
author yunhu
date 2019/5/19
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
with tf.Session() as sess:
    # 构建 VGG16 模型对象
    vgg = vgg16.Vgg16()
    # 开辟一个空间 神经网络构建 graph 的时候在模型中的占位
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3]) 
    # 图像是一个维度是 [None,224,224,3] 的张量
    with tf.name_scope("content_vgg"):
        # 载入 VGG16 模型
        vgg.build(input_)
    # 对每个不同种类的国画分别用 VGG16 计算特征值
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        # paint_photos/human
        # 具体的文件名
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            # 载入图像并放入 batch 数组中
            # paint_photos/human/人物1.jpg
            img = utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            # 标签名
            labels.append(each)

            # 如果图片数量到了 batch_size 则开始具体的运算，或者结束
            if ii % batch_size == 0 or ii == len(files):
                # 拼接
                images = np.concatenate(batch)
                feed_dict = {input_: images}
                # 计算特征值
                codes_batch = sess.run(vgg.relu6, feed_dict = feed_dict)
                # 将结果放入到 codes 数组中， codes 存放所有图片的特征值
                if codes is None:
                    codes = codes_batch
                else:
                    # 拼接
                    codes = np.concatenate((codes, codes_batch))

                # 清空数组准备下一个batch的计算
                batch = []
                print('{} images processed'.format(ii))


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