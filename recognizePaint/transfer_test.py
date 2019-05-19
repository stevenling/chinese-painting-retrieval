#coding=utf-8
import os
import numpy as np
import tensorflow as tf
import ftrain
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils


#用测试集来测试模型效果
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint(ftrain.MODEL_SAVE_PATH))
    #训练好的参数提取出来
    feed = {ftrain.inputs_: ftrain.test_x,
            ftrain.labels_: ftrain.test_y}
    test_acc = sess.run(ftrain.accuracy,feed_dict=feed)
    print ("Test accuracy: {:.4f}".format(test_acc))