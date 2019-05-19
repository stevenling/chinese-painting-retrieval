# coding=utf-8
import os
import numpy as np
import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

# 接下来我们将 flower_photos 文件夹中的国画图片都载入到进来，并且用图片所在的子文件夹作为标签值。
data_dir = 'paint_photos/' #数据来源文件夹
contents = os.listdir(data_dir)#返回指定的文件夹包含的文件或文件夹的名字的列表
classes = [each for each in contents if os.path.isdir(data_dir + each)]
#classes ['flowerBird', 'human', 'landscape']

# 利用vgg16计算得到特征值
# 首先设置计算batch的值，如果运算平台的内存越大，这个值可以设置得越高
batch_size = 5

# 用codes_list来存储特征值
codes_list = []

# 用labels来存储国画的类别
labels = []

# batch数组用来临时存储图片数据
batch = []

codes = None

with tf.Session() as sess:
    # 构建VGG16模型对象
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3]) #开辟一个空间 神经网络构建graph的时候在模型中的占位
    #图像是一个维度是[None,224,224,3]的张量
    with tf.name_scope("content_vgg"): #
        # 载入VGG16模型
        vgg.build(input_)

    # 对每个不同种类的国画分别用VGG16计算特征值
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        #paint_photos/human
        files = os.listdir(class_path)#具体的文件名
        for ii, file in enumerate(files, 1):
            # 载入图像并放入batch数组中
            img = utils.load_image(os.path.join(class_path, file))#paint_photos/human/人物1.jpg
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)#标签名

            # 如果图片数量到了batch_size则开始具体的运算
            if ii % batch_size == 0 or ii == len(files):
                images = np.concatenate(batch) #拼接

                feed_dict = {input_: images}
                # 计算特征值
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                # 将结果放入到codes数组中
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))#拼接

                # 清空数组准备下一个batch的计算
                batch = []
                print('{} images processed'.format(ii))
# code is a two-dimensional array including features of all pictures
# 这样我们就可以得到一个 codes 数组，和一个 labels 数组，分别存储了所有国画的特征值和类别。
# 可以用如下的代码将这两个数组保存到硬盘上：
np.save("codes.npy", codes)

import csv

#labels文件名
with open('labels', 'w') as f:
    writer = csv.writer(f, delimiter='\n') #delimiter是分隔符
    writer.writerow(labels)
    # pickle.dump(labels,f)