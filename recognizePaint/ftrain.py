# coding=utf-8
import os
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils
from sklearn.model_selection import StratifiedShuffleSplit

# 模型保存的路径和名称
MODEL_SAVE_PATH = "./checkpoints/"
MODEL_NAME = "paint.ckpt"
LABELS = "/recognizePaint/labels"
CODES_FILE_PATH = "/recognizePaint/codes.npy"

codes = None
label = []
# 存储标签
labels = []

# 获取当前工作目录
current_directory = os.getcwd()
print("Current working directory:", current_directory)

if CODES_FILE_PATH:
    # 存在有图片特征值的文件就加载
    try:
        codes = np.load(CODES_FILE_PATH)
    # 在这里可以对 loaded_array 进行进一步的操作
    except FileNotFoundError:
        print(f"File '{CODES_FILE_PATH}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("No such file, please run get_feature.py first")

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

# 准备训练集，验证集和测试集
# 首先我把 labels 数组中的分类标签用 One Hot Encode 的方式替换
labels.pop()  # 多读了一个空格 删除
lb = LabelBinarizer()  # 标签二值化
lb.fit(labels)  # 将标签放入lb中
labels_vecs = lb.transform(labels)  # 进行transform得到索引值。inverse_transform(y)：根据索引值y获得原始数据。
# print(labels_vecs)
# return codes,labels,labels_vecs

'''
交叉验证是指在给定的建模样本中，拿出其中的大部分样本进行模型训练，
生成模型，留小部分样本用刚建立的模型进行预测，并求这小部分样本的预测误差，
记录它们的平方加和。这个过程一直进行，直到所有的样本都被预测了一次而且仅被预测一次，
比较每组的预测误差，选取误差最小的那一组作为训练模型。
接下来就是抽取数据，
而且 labels 数组中的数据也还没有被打乱，
所以最合适的方法是使用 StratifiedShuffleSplit 方法来进行分层随机划分。
假设我们使用训练集：验证集：测试集 = 8:1:1，那么代码如下：
'''
# 10 个数据 8 个是训练数据，2 个是测试数据
# 分割数据集
ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
train_idx, val_idx = next(ss.split(codes, labels))
# 训练集索引 以及标签索引
# print(len(train_idx))
# 总共 2400 条数据，切成 1920 240 240
# train_idx 有 1920
# val_idx 有 480 条
# half_val_len 有 240
half_val_len = int(len(val_idx) / 2)

# 验证集与测试集对半分
val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

train_x, train_y = codes[train_idx], labels_vecs[train_idx]

val_x, val_y = codes[val_idx], labels_vecs[val_idx]

test_x, test_y = codes[test_idx], labels_vecs[test_idx]

# 训练集数量 维度 标签数量 维度
print("Train shapes (x, y):", train_x.shape, train_y.shape)
print("Validation shapes (x, y):", val_x.shape, val_y.shape)
print("Test shapes (x, y):", test_x.shape, test_y.shape)

# 训练网络
'''
分好了数据集之后，就可以开始对数据集进行训练了，
假设我们使用一个 256 维的全连接层，
一个 3 维的全连接层（因为我们要分类 3 种不同类的国画），
和一个 softmax 层。当然，这里的网络结构可以任意修改，尝试其他的结构以找到合适的结构。
'''
# 输入数据的维度

# 标签数据的维度
labels_ = tf.placeholder(tf.int64, shape=[None, labels_vecs.shape[1]])  # 3 维
inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])

# 加入一个 256 维的全连接的层
fc = tf.contrib.layers.fully_connected(inputs_, 256)
# 加入一个 3 维的全连接层
logits = tf.contrib.layers.fully_connected(fc, labels_vecs.shape[1], activation_fn=None)
# 得到最后的预测分布，将输出层的原始分数（logits）转换为概率分布
predicted = tf.nn.softmax(logits)

# 计算 cross entropy 值 softmax 交叉熵
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)

# 计算损失函数
# 损失函数其实表示的是真实值与网络的估计值的误差
cost = tf.reduce_mean(cross_entropy)

# 采用用得最广泛的 AdamOptimizer 优化器
optimizer = tf.train.AdamOptimizer().minimize(cost)
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_batches(x, y, n_batches=5):
    """
    为了方便把数据分成一个个 batch 以降低内存的使用，还可以再用一个函数专门用来生成 batch。
    这是一个生成器函数, 按照 n_batches 的大小将数据划分了小块 
    """
    # batch 数量
    batch_size = len(x) // n_batches

    for ii in range(0, n_batches * batch_size, batch_size):
        # 如果不是最后一个 batch，那么这个 batch 中应该有 batch_size 个数据
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
            # 否则的话，那剩余的不够 batch_size 的数据都凑入到一个 batch 中
        else:
            X, Y = x[ii:], y[ii:]
        # 生成器语法，返回 X 和 Y 
        yield X, Y
