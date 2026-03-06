"""
基于特征向量的图像检索工具函数：
- 计算欧式距离
- 按距离对图库排序，返回最相似的结果
"""

from typing import Sequence, Tuple, List

import numpy as np


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    计算两个特征向量之间的欧式距离平方和。
    """
    return float(np.sum((x1 - x2) ** 2))


def rank_by_distance(
    query_feat: np.ndarray,
    gallery_feats: Sequence[np.ndarray],
    gallery_meta: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str, float]]:
    """
    根据与查询特征的欧式距离，对图库中的特征进行排序。

    参数:
        query_feat: 查询图片的特征向量，shape=(feat_dim,)。
        gallery_feats: 库中图片特征列表。
        gallery_meta: 与 gallery_feats 对应的元信息列表，每项为 (img_path, label)。

    返回:
        排序后的列表 [(img_path, label, distance), ...]，距离从小到大。
    """
    distances: List[Tuple[str, str, float]] = []
    for (img_path, label), feat in zip(gallery_meta, gallery_feats):
        d = euclidean_distance(query_feat, feat)
        distances.append((img_path, label, d))
    distances.sort(key=lambda x: x[2])
    return distances


def top_k(
    ranked_results: List[Tuple[str, str, float]],
    k: int = 5
) -> List[Tuple[str, str, float]]:
    """
    从排序结果中取前 k 个。
    """
    return ranked_results[:k]

