"""
使用端到端微调好的 VGG16 模型进行预测的工具函数。

- 从 checkpoints/vgg16_finetune.keras 加载模型（只加载一次并缓存）；
- 提供从图片路径直接得到类别索引和中文类别名的接口。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config


MODEL_FILENAME = "vgg16_finetune.keras"


@lru_cache(maxsize=1)
def load_finetuned_model() -> keras.Model:
    """
    从磁盘加载已经微调好的 VGG16 端到端模型，并缓存到内存中。
    """
    model_path = tf.io.gfile.join(config.MODEL_DIR, MODEL_FILENAME)
    model = keras.models.load_model(model_path)
    return model


def _load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    从磁盘加载一张图片，并做与训练阶段相同的预处理：
    - resize 到 224x224
    - 转为 float32 数组
    - 使用 keras.applications.vgg16.preprocess_input 标准化

    返回:
        shape = (1, 224, 224, 3) 的 numpy 数组。
    """
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x)
    return x


def predict_class_from_image_path(image_path: str) -> int:
    """
    给定一张图片路径，返回预测的类别索引（int）。
    """
    model = load_finetuned_model()
    x = _load_and_preprocess_image(image_path)
    preds = model.predict(x, verbose=0)
    class_idx = int(np.argmax(preds, axis=1)[0])
    return class_idx


def predict_class_and_label_zh(image_path: str) -> Tuple[int, str]:
    """
    给定图片路径，返回 (类别索引, 中文类别名)。
    """
    class_idx = predict_class_from_image_path(image_path)
    if 0 <= class_idx < len(config.CLASS_NAMES_ZH):
        label_zh = config.CLASS_NAMES_ZH[class_idx]
    else:
        label_zh = "未知类别"
    return class_idx, label_zh

