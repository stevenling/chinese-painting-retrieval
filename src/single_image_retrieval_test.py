"""
简单的图像检索测试脚本：
- 对一张固定路径的图片提取 VGG16 特征；
- 使用训练好的分类模型得到特征表示；
- 从 SQLite 数据库 paint.db 中取出 label 为 "flowerBird" 的所有图片特征，
  计算与查询图片的欧式距离，找到距离最近的一张并输出其文件名。
"""

import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils
import numpy as np
import io
import os
import sqlite3
import json
import pickle
import ftrain


def convert_array(text):
    """
    将数据库中以二进制形式存储的 numpy 数组文本转换回 numpy 数组。
    当前函数未在本脚本中实际使用，保留作为示例。
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def test():
    """
    对一张固定路径的图片进行特征提取，
    然后在数据库中查找标签为 "flowerBird" 的图片特征，
    计算欧式距离并输出距离最近图片的文件名。
    """
    saver = tf.train.Saver()
    image_url = "D://Desktop/花鸟1.jpg"  # 待检索图片的本地路径

    # 预处理输入图像，调整为 (1, 224, 224, 3)
    test_pic_arr = []
    img_ready = utils.load_image(image_url)
    test_pic_arr.append(img_ready.reshape((1, 224, 224, 3)))
    images = np.concatenate(test_pic_arr)  # 预处理好图像

    with tf.Session() as sess:
        # 使用预训练 VGG16 提取图像特征
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)
        feed_dict = {input_: images}
        codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

        # 使用训练好的分类网络，将 VGG 特征送入全连接网络得到最终特征
        pre_value = tf.argmax(ftrain.predicted, 1)
        saver.restore(sess, tf.train.latest_checkpoint(ftrain.MODEL_SAVE_PATH))
        pre_value = sess.run(pre_value, feed_dict={ftrain.inputs_: codes_batch})

        # 连接数据库，读取标签为 flowerBird 的所有记录
        conn = sqlite3.connect("paint.db", detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        cursor.execute('select * from image where label = "flowerBird"')
        values = cursor.fetchall()

        # 计算与每条记录特征的欧式距离
        rst = np.zeros(len(values))
        [codes_batch] = codes_batch  # 去掉 batch 这一维
        for i, temp_values in enumerate(values):
            temp_feature = pickle.loads(temp_values[3])  # 反序列化特征
            rst[i] = distance(temp_feature, codes_batch)

        # 找到距离最小的一条记录，输出对应的文件名
        rst_index = np.argsort(rst)
        res_record = values[rst_index[0]]
        retrieval_result = res_record[2]
        print(retrieval_result)


def distance(x1, x2):
    """
    计算两个特征向量之间的欧式距离的平方和。
    """
    return np.sum((x1 - x2) ** 2)


def main():
    """
    脚本入口，执行一次检索测试。
    """
    test()


if __name__ == '__main__':
    main()