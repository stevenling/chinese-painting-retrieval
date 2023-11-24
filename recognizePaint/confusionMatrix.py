#coding=utf-8
import numpy as np
import os
import sys
import tensorflow as tf
import ftrain

from tensorflow_vgg import vgg16
from tensorflow_vgg import utils
from PyQt5 import QtWidgets, QtCore

# 数据来源文件夹
test_data_dir = 'test_photos/' 
# 返回指定的文件夹包含的文件或文件夹的名字的列表
contents = os.listdir(test_data_dir)
classes = [each for each in contents if os.path.isdir(test_data_dir + each)]

preValue = ""
res = np.zeros((3,3))
labels_vecs = ['flowerBird','human','landscape']
labels_vecs = np.array(labels_vecs)
realImgUrlList = []

def get_img_url_list():
    """
    获取所有图片的完整路径
    """
    testPicArr = []
    print(classes) # ['flowerBird', 'human', 'landscape']
    for each in classes:
        print("Starting {} images".format(each))
        class_path = test_data_dir + each
        # paint_photos/human
        # 具体的文件名
        files = os.listdir(class_path)
        preImgUrl = "C://Users/Administrator/PycharmProjects/recognizePaint/"  # 前缀
        for i, file in enumerate(files, 1):  # file 人物验证1
            #print(i)
            #print(files)
            imgUrl = class_path + "/" + file  ## test_photos/flowerBird/花鸟验证1.jpg
            realImgUrl = preImgUrl + imgUrl # 完整的图像路径
            #print(realImgUrl)
            realImgUrlList.append(realImgUrl)  # 获取所有图像文件路径 存到 list 中
    print(realImgUrlList)


def per_picture(count):
    testPicArr = []
    img_ready = utils.load_image(realImgUrlList[count])
    testPicArr.append(img_ready.reshape((1,224,224,3)))
    images = np.concatenate(testPicArr)
    return images

saver = tf.train.Saver()
i = 0
j = 0

def get_image_retrieval_result():
    global preValue
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
            preValue = tf.argmax(ftrain.predicted, 1)
            saver.restore(sess, tf.train.latest_checkpoint(ftrain.MODEL_SAVE_PATH))
            preValue = sess.run(preValue, feed_dict={ftrain.inputs_: codes_batch})
            if(j == 3):
                j = 0
                i = i + 1
            res[i][j] = preValue + 1
            print(res[i][j])
            #print(labels_vecs[preValue])
            #print ("The prediction paint is:", labels_vecs[preValue])

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