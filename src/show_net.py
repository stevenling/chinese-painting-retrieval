"""
加载已训练好的 TensorFlow 模型（paint.ckpt.meta），
并将其计算图导出到 ./log 目录，便于使用 TensorBoard 可视化网络结构。

Author: yunhu
Date: 2019/5/19
"""

import tensorflow as tf

g = tf.Graph()
with g.as_default() as g:
    tf.train.import_meta_graph('./checkpoints/paint.ckpt.meta')

with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir='./log', graph=g)