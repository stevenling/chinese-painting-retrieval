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
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def test():
    saver = tf.train.Saver()
    imageUrl = "D://Desktop/花鸟1.jpg"  # 获取编辑框的本地图标路径
    testPicArr = []
    img_ready = utils.load_image(imageUrl)
    testPicArr.append(img_ready.reshape((1, 224, 224, 3)))
    images = np.concatenate(testPicArr)  # 预处理好图像
    with tf.Session() as sess:
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)
        feed_dict = {input_: images}
        # 计算特征值
        codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
        #print(codes_batch)
        #返回y矩阵中最大值的下标，如果是二维的加1
        preValue = tf.argmax(ftrain.predicted, 1)
        # 加载训练好的新模型
        saver.restore(sess, tf.train.latest_checkpoint(ftrain.MODEL_SAVE_PATH))
        # 计算预测值
        preValue = sess.run(preValue, feed_dict={ftrain.inputs_: codes_batch})
        conn = sqlite3.connect("paint.db", detect_types=sqlite3.PARSE_DECLTYPES)

        #conn = sqlite3.connect('paint.db')
        cursor = conn.cursor()
        cursor.execute('select * from image where label = "flowerBird"')  # 获取标签是花鸟的数据
        values = cursor.fetchall()  # 使用featchall获得结果集（list）
        # 从结果集中依次取出特征值
        # print(values)
        rst = np.zeros(len(values))
        #codes_batch = np.array(codes_batch)
        [codes_batch] = codes_batch  #去掉一个维度
        #print(codes_batch)
        #print(codes_batch.dtype)
        #codes_batch = codes_batch.astype('<U78')
        #print(codes_batch.dtype)
        i = 0
        #sqlite3.register_converter("array", convert_array)

        for i, tempValues in enumerate(values):
            tempFeature = pickle.loads(tempValues[3])
            #print(tempFeature.dtype)
            rst[i] = distance(tempFeature, codes_batch)
            #aaaaprint(rst[i])
        rst_index = np.argsort(rst)
        #print(rst_index)
        #print(rst_index[0])
        resRecord = values[rst_index[0]]#查到的记录是
        retrievalResult = resRecord[2]
        print(retrievalResult)

def distance(x1, x2):
    return (np.sum((x1 - x2) ** 2))

def main():
    test()
    
if __name__ == '__main__':
    main()