import numpy as np
import os
import sys
import pickle
import sqlite3
import tensorflow as tf
import ftrain
import app

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

codes_batch = []
imageUrl = ""
# 图像的类别
preValue = ""  
labels_vecs = ['flowerBird','human','landscape']
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("图像预测")
        MainWindow.resize(1500, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget) #第一个按钮点击显示图片
        self.pushButton.setGeometry(QtCore.QRect(600, 90, 200, 50))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setFont(QtGui.QFont("Roman times", 15, QtGui.QFont.Bold))  # 第一个按钮设置字体大小

        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget) #第二个按钮点击进行图像预测
        self.pushButton2.setGeometry(QtCore.QRect(900, 90, 200, 50))
        self.pushButton2.setObjectName("pushButton2")
        self.pushButton2.setFont(QtGui.QFont("Roman times", 15, QtGui.QFont.Bold))  # 第二个按钮设置字体大小

        self.pushButton3 = QtWidgets.QPushButton(self.centralwidget) #第三个按钮点击进行图像检索
        self.pushButton3.setGeometry(QtCore.QRect(1200, 90, 200, 50))
        self.pushButton3.setObjectName("pushButton3")
        self.pushButton3.setFont(QtGui.QFont("Roman times", 15, QtGui.QFont.Bold))  # 设置字体大小

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 90, 200, 50))
        self.label.setObjectName("label")
        self.label.setFont(QtGui.QFont("Roman times", 15, QtGui.QFont.Bold))  # 设置字体大小


        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(250, 90, 300, 50))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setFont(QtGui.QFont("Roman times", 15, QtGui.QFont.Bold))  # 设置字体大小

        # 显示第一张图像
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(100, 200, 800, 500))#左上宽高
        self.label_2.setObjectName("label_2")

        # 显示分类结果
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(700, 200, 200, 200))
        self.label_3.setObjectName("label_3")
        # 设置字体
        self.label_3.setFont(QtGui.QFont("Roman times", 40, QtGui.QFont.Bold))

        # 显示检索出来的图像
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1000, 200, 800, 500))
        self.label_4.setObjectName("label_4")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # 点击预测按钮
        self.pushButton.clicked.connect(self.showImage)  
        # 点击显示图像类别
        self.pushButton2.clicked.connect(self.showImageCategory) 
        # 点击显示检索到的图像
        self.pushButton3.clicked.connect(self.showRetrievalResult) 
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):  
        """
        显示前端的
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像检索"))
        self.pushButton.setText(_translate("MainWindow", "显示图像"))
        self.pushButton2.setText(_translate("MainWindow", "点击预测"))
        self.pushButton3.setText(_translate("MainWindow", "检索图像"))
        self.label.setText(_translate("MainWindow", "请输入图像路径："))
        self.label_2.setText(_translate("MainWindow", "")) # 输入图像的控件
        self.label_3.setText(_translate("MainWindow", "")) # 分类结果
        self.label_4.setText(_translate("MainWindow", "")) # 检索出来的图像的控件
    
    def showImageCategory(self):
        """
        显示图像的类别
        """
        # 输入的图像的特征值
        global codes_batch 
        global preValue
        # 获取编辑框的本地图标路径
        imageUrl = self.lineEdit.text()  
        # app.get_image_retrieval_result(imageUrl)
        testPicArr = []
        img_ready = utils.load_image(imageUrl)
        testPicArr.append(img_ready.reshape((1, 224, 224, 3)))
        images = np.concatenate(testPicArr)  # 预处理好图像
        saver = tf.train.Saver()
        with tf.Session() as sess:
            vgg = vgg16.Vgg16()
            input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
            with tf.name_scope("content_vgg"):
                # 载入VGG16模型
                vgg.build(input_)
            feed_dict = {input_: images}
            # 计算特征值
            codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
            # 返回 y 矩阵中最大值的下标，如果是二维的加1
            preValue = tf.argmax(ftrain.predicted, 1)
            # 加载训练好的新模型
            saver.restore(sess, tf.train.latest_checkpoint(ftrain.MODEL_SAVE_PATH))
            # 计算预测值
            preValue = sess.run(preValue, feed_dict={ftrain.inputs_: codes_batch})
            if (preValue == 0):
                self.label_3.setText("花鸟")
            elif (preValue == 1):
                self.label_3.setText("人物")
            elif (preValue == 2):
                self.label_3.setText("山水")

    # 显示输入图像
    def showImage(self):
        print("start show image")
        self.label_3.setText("")
        self.label_4.setText("")
        imageUrl = self.lineEdit.text()  
        jpg = QtGui.QPixmap(imageUrl).scaled(400, 400)
        self.label_2.setPixmap(jpg)

    # 显示图像检索结果，从数据库中取出对应类别的所有特征值 每一个与输入的图像进行计算欧式距离 选择最小的
    def showRetrievalResult(self):
        print("start Retrieval image")
        global preValue
        global codes_batch
        global labels_vecs
        print(preValue)
        codes_batch = np.array(codes_batch)
        [codes_batch] = codes_batch  #去掉一个维度

        #nowLabel = labels_vecs[preValue] #当前的类别
        if preValue == 0:
            nowLabel = "flowerBird"
        elif (preValue == 1):
            nowLabel = "human"
        elif (preValue == 2):
            nowLabel = "landscape"
        print(nowLabel)
        conn = sqlite3.connect('paint.db')
        cursor = conn.cursor()
        #cursor.execute('select * from image where label = "flowerBird"')# 获取标签是花鸟的数据
        cursor.execute('select * from image where label = ?', [nowLabel])

        values = cursor.fetchall()  # 使用featchall获得结果集（list）
        # 从结果集中依次取出特征值
        #print(values)
        rst = np.zeros(len(values))
        for i, tempValues in enumerate(values):
            tempFeature = pickle.loads(tempValues[3])
            #print(tempFeature.dtype)
            rst[i] = self.distance(tempFeature, codes_batch)
            #print(rst[i])
        rst_index = np.argsort(rst)
        #print(rst_index)
        #print(rst_index[0])
        resRecord = values[rst_index[0]]#查到的记录是
        retrievalResult = resRecord[2]
        print(retrievalResult)

        #显示检索出来的图像
        prefixImageURl = "C://Users/Administrator/PycharmProjects/recognizePaint/paint_photos/"
        self.label_4.setText("")
        imageUrl = prefixImageURl + nowLabel + "/"+retrievalResult #拼接图像路径 前缀 + 文件名
        print(imageUrl)
        jpg = QtGui.QPixmap(imageUrl).scaled(400, 400)
        self.label_4.setPixmap(jpg)

    def distance(self,x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
