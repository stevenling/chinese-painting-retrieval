"""
使用 tf.keras.applications.VGG16 做端到端微调训练：

- 从 data/paint_photos/ 目录按子文件夹标签直接读图片；
- 使用 ImageNet 预训练的 VGG16 作为特征提取 backbone（不包含顶层）；
- 接 GlobalAveragePooling + 自定义全连接分类头（256 + Dropout + 3 类 softmax）；
- 先冻结 VGG16，只训练分类头；
- 再解冻 VGG16 的 block5 做小学习率微调（fine-tuning）。

Author: yunhu
Date: 2019/5/19
"""

from __future__ import annotations

import os

import tensorflow as tf
from tensorflow import keras

import config


AUTOTUNE = tf.data.AUTOTUNE

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10


def build_datasets():
    """
    从文件夹构建训练 / 验证 / 测试数据集。

    训练 + 验证：来自 DATA_DIR，按 0.2 做 validation_split；
    测试：来自 TEST_DATA_DIR，单独目录。
    """
    train_ds = keras.preprocessing.image_dataset_from_directory(
        config.DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=42,
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        config.DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=42,
    )

    test_ds = keras.preprocessing.image_dataset_from_directory(
        config.TEST_DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 性能优化：预取、缓存
    def configure(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.prefetch(AUTOTUNE)

    return configure(train_ds), configure(val_ds), configure(test_ds)


def build_model(num_classes: int) -> keras.Model:
    """
    构建 VGG16 + 全连接分类头的端到端模型。
    """
    base_model = keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False  # 初始只训练分类头

    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.1),
        ]
    )

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    x = keras.applications.vgg16.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


def compile_for_feature_extraction(model: keras.Model) -> None:
    """
    编译模型：只训练自定义分类头（VGG16 冻结）。
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def compile_for_fine_tuning(model: keras.Model, base_model: keras.Model) -> None:
    """
    编译模型：解冻 VGG16 的 block5 做小学习率微调。
    """
    base_model.trainable = True

    for layer in base_model.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_and_finetune() -> None:
    """
    端到端训练流程：
    1. 冻结 VGG16，只训练分类头。
    2. 解冻 block5，做小学习率微调。
    3. 在测试集上评估，并保存模型。
    """
    print("Current working directory:", os.getcwd())

    train_ds, val_ds, test_ds = build_datasets()
    num_classes = len(config.CLASS_NAMES)

    model = build_model(num_classes)

    # 找到底层 VGG16 模型，方便之后微调
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name.startswith("vgg16"):
            base_model = layer
            break
    if base_model is None:
        # 退一步：按名字查找
        base_model = model.get_layer("vgg16")

    # 1. 先训练分类头
    compile_for_feature_extraction(model)

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

    print("Start feature extraction training...")
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=[early_stopping, reduce_lr],
    )

    # 2. 微调 VGG16 的 block5
    print("Start fine-tuning on VGG16 block5...")
    compile_for_fine_tuning(model, base_model)

    fine_tune_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
    ]

    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=fine_tune_callbacks,
    )

    # 3. 在测试集上评估
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[Fine-tuned VGG16] Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

    # 保存模型
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    save_path = os.path.join(config.MODEL_DIR, "vgg16_finetune.keras")
    model.save(save_path)
    print(f"Saved fine-tuned VGG16 model to: {save_path}")


if __name__ == "__main__":
    train_and_finetune()

