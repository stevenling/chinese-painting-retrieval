"""
分类模型加载与预测相关的工具函数。

基于 ftrain.py 中定义的计算图和 checkpoint，提供统一的预测接口。
"""

from typing import Sequence

import numpy as np
import tensorflow as tf

import config
import ftrain
from core.features import extract_features_for_image


def load_trained_session() -> tf.Session:
    """
    从配置的模型目录中加载最近一次保存的分类模型，返回一个已恢复权重的 Session。
    """
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(config.MODEL_DIR))
    return sess


def predict_classes_from_features(
    sess: tf.Session,
    features: np.ndarray,
) -> np.ndarray:
    """
    使用已加载好的 Session 和特征数组进行前向预测，返回类别索引数组。

    参数:
        sess: 已恢复权重的 TensorFlow Session。
        features: shape 为 (N, feature_dim) 的特征数组。
    """
    pre_value = tf.argmax(ftrain.predicted, 1)
    preds = sess.run(pre_value, feed_dict={ftrain.inputs_: features})
    return preds


def predict_class_from_image_path(image_path: str) -> int:
    """
    直接从图片路径完成一整套预测流程：
    - 使用 VGG16 提取特征
    - 加载分类模型
    - 输出预测类别索引（int）
    """
    features = extract_features_for_image(image_path)
    sess = load_trained_session()
    try:
        preds = predict_classes_from_features(sess, features)
    finally:
        sess.close()
    return int(preds[0])

