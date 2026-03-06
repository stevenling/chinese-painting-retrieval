# coding=utf-8
"""
使用 TensorFlow 2.x / Keras 提取数据集的特征值，存放在 codes.npy 中。

流程：
- 遍历 data/paint_photos/ 下的子目录（类别），加载所有图片
- 使用预训练 VGG16 的 'fc2' 层输出作为图像特征
- 将所有图像特征堆叠成 codes.npy，并将对应标签写入 labels 文件
"""

import os
import csv
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config


def load_and_preprocess(img_path: str) -> np.ndarray:
    """
    加载单张图片并预处理成 (1, 224, 224, 3) 的输入张量。
    """
    # 读取并缩放图片到 VGG16 所需的 224x224 尺寸
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    # 转换为 float32 的 HWC 格式数组
    x = keras.preprocessing.image.img_to_array(img)
    # 扩展 batch 维度，形状从 (224, 224, 3) 变为 (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    # 按 VGG16 官方规则做预处理（减均值、通道顺序等）
    x = keras.applications.vgg16.preprocess_input(x)
    return x


def main() -> None:
    # 数据根目录，每个子文件夹对应一个类别，如 flowerBird / human / landscape
    data_dir = config.DATA_DIR  # e.g. data/paint_photos
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR 不存在，请检查 config.DATA_DIR 配置: {data_dir}")

    # 找出 data_dir 下的所有“类别文件夹”
    contents = os.listdir(data_dir)
    classes: List[str] = [d for d in contents if os.path.isdir(os.path.join(data_dir, d))]

    # 加载预训练 VGG16 模型，使用全连接层 fc2 的输出作为图像特征
    base_model = keras.applications.VGG16(weights="imagenet", include_top=True)
    feature_layer = base_model.get_layer("fc2").output
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=feature_layer)

    # 提取整个数据集的特征和标签
    codes, labels = extract_dataset_features(data_dir, classes, feature_extractor, batch_size=5)

    # 合并所有 batch，得到 (N, feat_dim) 的特征矩阵，其中 N 为所有图片总数
    print("Final features shape:", codes.shape)

    # 保存特征和标签到磁盘，供后续训练和检索使用
    np.save(config.CODES_PATH, codes)
    print(f"Saved features to: {config.CODES_PATH}")

    with open(config.LABELS_PATH, "w") as f:
        writer = csv.writer(f, delimiter="\n")
        writer.writerow(labels)
    print(f"Saved labels to: {config.LABELS_PATH}")


def extract_dataset_features(
    data_dir: str,
    classes: List[str],
    feature_extractor: tf.keras.Model,
    batch_size: int = 5,
) -> tuple[np.ndarray, List[str]]:
    """
    遍历 data_dir 下的所有类别文件夹，使用给定的 feature_extractor 提取特征。

    返回:
        codes: shape=(N, feat_dim) 的特征矩阵
        labels: 长度为 N 的类别名称列表
    """
    codes_batches: List[np.ndarray] = []  # 存放每个 batch 的特征
    labels: List[str] = []  # 存放与特征对应的类别名称

    for cls in classes:
        class_path = os.path.join(data_dir, cls)
        files = os.listdir(class_path)  # 该类别下的所有图片文件名
        print(f"Starting {cls} images")

        batch_imgs: List[np.ndarray] = []  # 当前 batch 的图片张量列表
        for i, fname in enumerate(files, 1):
            img_path = os.path.join(class_path, fname)
            # 加载并预处理当前图片
            x = load_and_preprocess(img_path)
            batch_imgs.append(x)
            # 记录该图片所属的类别
            labels.append(cls)

            # 当累积到一个 batch 或者已经到达最后一张图片时，进行一次前向计算
            if i % batch_size == 0 or i == len(files):
                # 将当前 batch 中所有图片堆叠成一个大数组 (batch, 224, 224, 3)
                batch_arr = np.vstack(batch_imgs)
                # 通过 VGG16 的 fc2 层前向计算得到特征向量 (batch, feat_dim)
                feats = feature_extractor(batch_arr).numpy()
                # 保存该 batch 的特征
                codes_batches.append(feats)
                # 清空 batch，准备下一个 batch
                batch_imgs = []
                print(f"{i} images processed for class {cls}")

    # 合并所有 batch，得到 (N, feat_dim) 的特征矩阵，其中 N 为所有图片总数
    codes = np.vstack(codes_batches)
    return codes, labels


if __name__ == "__main__":
    main()
