"""
使用从 VGG16 提取的图像特征训练国画分类器（tf.keras 版本），并划分训练集、验证集和测试集。

主要功能：
- 从本地特征文件和标签文件中加载数据
- 使用 StratifiedShuffleSplit 按 8:1:1 划分训练/验证/测试集
- 构建两层全连接网络（256 维 + 3 类 softmax），使用 model.fit 进行训练
- 保存训练好的 Keras 模型到 checkpoints 目录

Author: yunhu
Date: 2019/5/19
"""

import os
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit

import config

# 模型保存路径和名称（使用新的 .keras 格式）
MODEL_SAVE_DIR = config.MODEL_DIR
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "classifier_keras.keras")
LABELS_PATH = config.LABELS_PATH
CODES_FILE_PATH = config.CODES_PATH

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def load_codes_and_labels() -> tuple[np.ndarray, List[str]]:
    """
    加载特征矩阵 codes.npy 和对应的标签列表 labels。
    """
    print("Current working directory:", os.getcwd())

    # 加载 codes.npy
    try:
        codes = np.load(CODES_FILE_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File '{CODES_FILE_PATH}' not found. 请先运行 get_features.py 生成特征。"
        )

    # 加载 labels 文件
    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(
            f"File '{LABELS_PATH}' not found. 请先运行 get_features.py 生成标签。"
        )

    labels_str: List[str] = []
    with open(LABELS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels_str.append(line)

    if codes.shape[0] != len(labels_str):
        raise ValueError(
            f"特征和标签数量不一致: codes={codes.shape[0]}, labels={len(labels_str)}。"
            "请确认 get_features.py 生成的文件匹配。"
        )

    return codes, labels_str


def encode_labels(labels_str: List[str]) -> tuple[np.ndarray, LabelBinarizer]:
    """
    将字符串标签列表编码为 one-hot 矩阵，并返回编码后的标签和编码器。
    """
    lb = LabelBinarizer()
    lb.fit(labels_str)
    labels_vecs = lb.transform(labels_str)  # shape: (N, num_classes)
    print("Labels classes:", lb.classes_)
    return labels_vecs, lb


def split_train_val_test(
    codes: np.ndarray,
    labels_str: List[str],
    labels_vecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 StratifiedShuffleSplit 将数据按 8:1:1 划分为训练集、验证集和测试集。
    """
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valtest_idx = next(ss.split(codes, labels_str))

    half_val_len = int(len(valtest_idx) / 2)
    val_idx, test_idx = valtest_idx[:half_val_len], valtest_idx[half_val_len:]

    train_x, train_y = codes[train_idx], labels_vecs[train_idx]
    val_x, val_y = codes[val_idx], labels_vecs[val_idx]
    test_x, test_y = codes[test_idx], labels_vecs[test_idx]

    print("Train shapes (x, y):", train_x.shape, train_y.shape)
    print("Validation shapes (x, y):", val_x.shape, val_y.shape)
    print("Test shapes (x, y):", test_x.shape, test_y.shape)

    return train_x, train_y, val_x, val_y, test_x, test_y


def build_classifier(input_dim: int, num_classes: int) -> keras.Model:
    """
    构建两层全连接的分类模型：
    - 256 隐层 + Dropout 防止过拟合
    - 输出层为 num_classes 维 softmax
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(1e-4),
            ),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def build_and_train_model() -> None:
    """
    使用 Keras 训练分类器并保存模型。
    """
    codes, labels_str = load_codes_and_labels()

    # 1. 标签 one-hot 编码
    labels_vecs, _ = encode_labels(labels_str)

    # 2. 按 8:1:1 划分 train / val / test（分层抽样）
    train_x, train_y, val_x, val_y, test_x, test_y = split_train_val_test(
        codes, labels_str, labels_vecs
    )

    # 3. 构建 Keras 模型（两层全连接：256 + num_classes）
    num_classes = labels_vecs.shape[1]
    input_dim = codes.shape[1]
    model = build_classifier(input_dim, num_classes)

    # 4. 训练模型（可以根据需要调整 epoch / batch_size）
    # 拿训练集反复喂给模型多轮，每次一小批 32 张图，同时用验证集监控效果。
    EPOCHS = 50
    BATCH_SIZE = 32

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    _ = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
    )

    # 5. 在测试集上评估
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

    # 6. 保存模型
    model.save(MODEL_SAVE_PATH)
    print(f"Saved Keras classifier model to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    build_and_train_model()