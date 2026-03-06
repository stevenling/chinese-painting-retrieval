import os
import numpy as np
import tensorflow as tf
import ftrain

from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

# 运行多少轮次
epochs = 20
# 统计训练效果的频率
iteration = 0
# 保存模型的保存器
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()  # 协调器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 入队线程启动器

    for e in range(epochs):
        for x, y in ftrain.get_batches(ftrain.train_x, ftrain.train_y):
            feed = {ftrain.inputs_: x,
                    ftrain.labels_: y}
            # 训练模型
            loss, _ = sess.run([ftrain.cost, ftrain.optimizer], feed_dict=feed)
            print("Epoch: {}/{}".format(e + 1, epochs),
                  "Iteration: {}".format(iteration),
                  "Training loss: {:.5f}".format(loss))
            iteration += 1
            # 1 个 iteration 等于使用 batch size 个样本训练一次
            # 1 个 epoch 等于使用训练集中的全部样本训练一次
            if iteration % 5 == 0:
                feed = {ftrain.inputs_: ftrain.val_x,
                        ftrain.labels_: ftrain.val_y}

                val_acc = sess.run(ftrain.accuracy, feed_dict=feed)
                # 输出用验证集验证训练进度
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Validation Acc: {:.4f}".format(val_acc))
                # 保存模型
    saver.save(sess, os.path.join(ftrain.MODEL_SAVE_PATH, ftrain.MODEL_NAME))