"""
项目统一配置文件：集中管理路径、文件名等常量，避免在各个脚本中硬编码。
"""

import os

# 代码所在目录（src）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录（仓库根目录：chinese-painting-retrieval）
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# 数据目录（顶层 data 目录下）
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "paint_photos")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test_photos")

# 特征与标签文件（仍然放在 recognizePaint 目录下）
CODES_PATH = os.path.join(BASE_DIR, "codes.npy")
LABELS_PATH = os.path.join(BASE_DIR, "labels")

# 模型保存目录
MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_NAME = "paint.ckpt"

# SQLite 数据库路径
DB_PATH = os.path.join(BASE_DIR, "paint.db")

# 爬虫相关：雅昌艺术网国画频道根地址
BASE_URL = "https://gallery.artron.net"

# 检索结果图像所在根目录（用于 GUI 拼接本地路径）
# 如需在本机上显示数据库中存储的图像，请将此路径改为实际的本地图像根目录。
RETRIEVAL_IMAGE_ROOT = DATA_DIR

# 类别名称（英文，用于模型 / 数据库）
CLASS_NAMES = ["flowerBird", "human", "landscape"]
# 类别名称（中文，用于界面展示）
CLASS_NAMES_ZH = ["花鸟", "人物", "山水"]

