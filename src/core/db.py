"""
SQLite 数据库相关的工具函数：
- 创建连接
- 按标签查询图像特征
"""

from typing import List, Tuple

import sqlite3
import pickle
import numpy as np

import config


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """
    创建并返回一个 SQLite 连接。
    """
    path = db_path or config.DB_PATH
    return sqlite3.connect(path)


def fetch_features_by_label(
    conn: sqlite3.Connection,
    label: str,
) -> List[Tuple[str, np.ndarray]]:
    """
    根据标签从 image 表中查询所有记录，返回 (img_path, feature_array) 列表。
    """
    cursor = conn.cursor()
    cursor.execute('select imgPath, feature from image where label = ?', [label])
    rows = cursor.fetchall()
    results: List[Tuple[str, np.ndarray]] = []
    for img_path, feat_blob in rows:
        feature = pickle.loads(feat_blob)
        results.append((img_path, feature))
    cursor.close()
    return results

