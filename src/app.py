# coding=utf-8
"""
预测新的图片

Author: yunhu
Date: 2019/5/19
"""

import numpy as np

import config
from core import model_v2 as model

labels_vecs = np.array(config.CLASS_NAMES)


def get_image_retrieval_result():
    """
    从用户输入的图片路径读取图像，使用 VGG16 + 训练好的分类模型进行预测，
    并在控制台输出预测的类别索引和对应的类别名称。
    """
    img_path = input('Input the path and image name:')
    pre_value = model.predict_class_from_image_path(img_path)
    print(pre_value)
    print("The prediction paint is:", labels_vecs[pre_value])
