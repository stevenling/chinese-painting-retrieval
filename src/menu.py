
import numpy as np
import os
import sys
import pickle
import sqlite3
import tensorflow as tf
import ftrain

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import config
from core import model_v2 as model
from core.features import extract_features_for_image
from core import db as db_utils
from core import retrieval as retrieval_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pre_value = ""  # 图像的类别


class Ui_MainWindow(object):
    def setup_ui(self, MainWindow):
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
        #
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

        #显示第一张图像
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(100, 200, 800, 500))
        self.label_2.setObjectName("label_2")

        #显示分类结果
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(300, 80, 200, 200))#左上宽高
        self.label_3.setObjectName("label_3")
        self.label_3.setFont(QtGui.QFont("Roman times", 40, QtGui.QFont.Bold))#设置字体
        #
        # #显示检索出来的图像
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(800, 200, 800, 500))
        self.label_4.setObjectName("label_4")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton.clicked.connect(self.show_image)  #点击预测按钮
        self.pushButton2.clicked.connect(self.show_image_category) #点击显示图像类别
        self.pushButton3.clicked.connect(self.show_retrieval_result) #点击显示检索到的图像
        self.retranslate_ui(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslate_ui(self, MainWindow):  #显示前端的
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像检索"))

        self.pushButton.setText(_translate("MainWindow", "显示图像"))

        self.pushButton2.setText(_translate("MainWindow", "点击预测"))
        #
        # self.pushButton3.setText(_translate("MainWindow", "检索图像"))

        self.label.setText(_translate("MainWindow", "请输入图像路径："))
        self.label_2.setText(_translate("MainWindow", "")) #输入图像的控件
        self.label_3.setText(_translate("MainWindow", "")) #分类结果
        # self.label_4.setText(_translate("MainWindow", "")) #检索出来的图像的控件

    # 显示图像的类别
    def show_image_category(self):
        print("start image category")
        global pre_value
        image_url = self.lineEdit.text()  # 获取编辑框的本地图片路径

        if not image_url:
            QMessageBox.warning(None, "输入错误", "请先在输入框中填写图像路径。")
            return
        if not os.path.exists(image_url):
            QMessageBox.warning(None, "路径不存在", f"找不到图像文件：\n{image_url}")
            return

        # 使用统一的模型接口进行预测
        try:
            pre_value = model.predict_class_from_image_path(image_url)
        except Exception as e:
            print("Predict error:", e)
            QMessageBox.critical(None, "预测失败", "分类模型预测出错，请检查日志输出。")
            return
        print(pre_value)
        if 0 <= pre_value < len(config.CLASS_NAMES_ZH):
            self.label_3.setText(config.CLASS_NAMES_ZH[pre_value])
        else:
            self.label_3.setText("未知类别")

    #显示输入图像
    def show_image(self):
        image_url = self.lineEdit.text()  # 获取编辑框的本地图标路径
        if not image_url:
            QMessageBox.warning(None, "输入错误", "请先在输入框中填写图像路径。")
            return
        if not os.path.exists(image_url):
            QMessageBox.warning(None, "路径不存在", f"找不到图像文件：\n{image_url}")
            return
        jpg = QtGui.QPixmap(image_url).scaled(400, 400)
        self.label_2.setPixmap(jpg)

    # #显示图像检索结果  从数据库中取出对应类别的所有特征值 每一个与输入的图像进行计算欧式距离 选择最小的
    def show_retrieval_result(self):
        print("start Retrieval image")
        global pre_value

        image_url = self.lineEdit.text()
        if not image_url:
            QMessageBox.warning(None, "输入错误", "请先在输入框中填写图像路径。")
            return

        # 重新预测一次类别，确保 pre_value 有效
        try:
            pre_value = model.predict_class_from_image_path(image_url)
        except Exception as e:
            print("Predict error:", e)
            QMessageBox.critical(None, "预测失败", "分类模型预测出错，请检查日志输出。")
            return
        print("predicted class index:", pre_value)

        if 0 <= pre_value < len(config.CLASS_NAMES):
            now_label = config.CLASS_NAMES[pre_value]
        else:
            print("Unknown class index:", pre_value)
            QMessageBox.warning(None, "预测结果异常", f"未知类别索引：{pre_value}")
            return

        # 获取查询图片的特征向量 (feat_dim,)
        query_feat_batch = extract_features_for_image(image_url)
        query_feat = query_feat_batch[0]

        # 从数据库中取出同类图片的特征
        try:
            conn = db_utils.connect()
            records = db_utils.fetch_features_by_label(conn, now_label)
        except Exception as e:
            print("DB error:", e)
            QMessageBox.critical(None, "数据库错误", "读取数据库时发生错误，请检查 paint.db。")
            return
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if not records:
            print("No records found in DB for label:", now_label)
            QMessageBox.information(None, "无检索结果", f"数据库中没有找到类别为 {now_label} 的记录。")
            return

        gallery_paths = [img_path for img_path, _ in records]
        gallery_feats = [feat for _, feat in records]
        gallery_meta = [(p, now_label) for p in gallery_paths]

        ranked = retrieval_utils.rank_by_distance(query_feat, gallery_feats, gallery_meta)
        top_k = retrieval_utils.top_k(ranked, k=1)
        if not top_k:
            QMessageBox.information(None, "无检索结果", "未找到相似图像。")
            return

        top_img_path, _, dist = top_k[0]
        print("Top retrieval:", top_img_path, "distance:", dist)

        # 显示检索出来的图像：使用 RETRIEVAL_IMAGE_ROOT 拼接完整路径
        self.label_4.setText("")
        full_path = os.path.join(config.RETRIEVAL_IMAGE_ROOT, now_label, top_img_path)
        print("Retrieval image full path:", full_path)
        if not os.path.exists(full_path):
            QMessageBox.warning(None, "图像缺失", f"检索结果图像文件不存在：\n{full_path}")
            return
        jpg = QtGui.QPixmap(full_path).scaled(400, 400)
        self.label_4.setPixmap(jpg)

from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setup_ui(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())