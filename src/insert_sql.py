"""
读取预先提取好的图像特征和类别标签，将它们插入到 SQLite 数据库 paint.db 的 image 表中。

数据来源：
- labels 文件：按顺序存放每张图片对应的类别名称
- imageData.pkl：使用 pickle 序列化后的图像特征列表（或数组）

主要流程：
- 从 labels 文件中加载标签到内存
- 从 imageData.pkl（或 codes.npy）中加载特征
- 遍历 paint_photos/ 目录下的所有图片，按顺序为每张图片插入一条数据库记录

Author: yunhu
Date: 2019/5/19
"""

import pickle
import os
import sqlite3
import io
import numpy as np
import config

LABELS = config.LABELS_PATH
labels = []
CODES = config.CODES_PATH


def get_labels():
    """
    从 LABELS 文件中读取每一行标签，去掉首尾空白后追加到全局列表 labels 中。
    如果 LABELS 为空，则提示先运行特征提取脚本。
    """
    # 存储标签的文件存在
    if LABELS:
        with open(LABELS, "r") as f:
            label = f.readlines()
            for line in label:
                # 去掉头尾的空格
                line = line.strip()
                labels.append(line)
    else:
        print("No such file,please run get_feature.py first")


def insert_db():
    """
    将标签和对应的图像特征插入到 SQLite 数据库 paint.db 的 image 表中。

    当前实现：
    - 从 imageData.pkl 中加载特征（按顺序对应每张图片）
    - 遍历 paint_photos/ 下的所有类别和图片文件
    - 为前 5000 张图片构造 (id, label, imgPath, feature) 并插入 image 表
    """
    codes = None
    if CODES:
        # 存在有图片特征值的文件 就加载（当前未使用 codes，而是使用 imageData.pkl）
        codes = np.load(CODES)
    else:
        print("No such file,please run get_feature.py first")

    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    pkl_path = os.path.join(config.BASE_DIR, 'imageData.pkl')
    with open(pkl_path, 'rb') as pkl_file:
        imgFeature = pickle.load(pkl_file)  # 特征值
    data_dir = config.DATA_DIR
    contents = os.listdir(data_dir)
    classes = [each for each in contents if os.path.isdir(os.path.join(data_dir, each))]
    i = 0
    for each in classes:
        class_path = os.path.join(data_dir, each)
        files = os.listdir(class_path)  # 具体的文件名 所有的
        for file in files:
            if i < 5000:
                tempLabel = labels[i]
                tempFeature = imgFeature[i]
                # tempFeature = codes[i]
                # tempFeature = tempFeature.tostring()
                # tempFeature = tempFeature.astype(np.float32)  # b不可直接转
                # print(tempFeature.dtype) #float32
                tempId = i + 1
                tempFeatBin = pickle.dumps(tempFeature)
                print(tempId)
                cursor.execute('insert into image (id, label, imgPath, feature) VALUES (?, ?, ?, ?)',
                               (tempId, tempLabel, file, sqlite3.Binary(tempFeatBin)))
                # cursor.execute('insert into image VALUES (%d,%s,%s,%s)',([tempId,  tempLabel, file, tempFeature]))
                i = i + 1
    print(cursor.rowcount)
    cursor.close()
    conn.commit()
    conn.close()


def main():
    """
    脚本入口：先加载标签，再将特征和标签写入数据库。
    """
    get_labels()
    insert_db()


if __name__ == '__main__':
    main()