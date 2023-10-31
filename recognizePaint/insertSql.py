import pickle
import os
import sqlite3
import io
import numpy as np

LABELS = "labels"
labels = []
CODES = "codes.npy"

# 获取存储的标签
def getLabels():
    # 存储标签的文件存在
    if LABELS: 
        with open(LABELS, "r") as f:
            label = f.readlines()
            for line in label:
                line = line.strip() #去掉头尾的空格
                labels.append(line)
    else:
        print("No such file,please run get_feature.py first")

def insertDb():
    codes = None
    if CODES:
        codes = np.load(CODES)  # 存在有图片特征值的文件 就加载
    else:
        print("No such file,please run get_feature.py first")


    #sqlite3.register_adapter(np.ndarray, adapt_array)
    conn = sqlite3.connect('paint.db')
    cursor = conn.cursor()
    pkl_file = open('imageData.pkl', 'rb')
    imgFeature = pickle.load(pkl_file) #特征值
    data_dir = 'paint_photos/' #数据来源文件夹
    contents = os.listdir(data_dir)#返回指定的文件夹包含的文件或文件夹的名字的列表 contents ['flowerBird', 'human', 'landscape']
    classes = [each for each in contents if os.path.isdir(data_dir + each)] #classes ['flowerBird', 'human', 'landscape']
    conn = sqlite3.connect('paint.db')
    cursor = conn.cursor()
    i = 0
    for each in classes:
        class_path = data_dir + each  # paint_photos/human
        files = os.listdir(class_path)  # 具体的文件名 所有的
        for file in files:
            if i < 5000:
                tempLabel = labels[i]
                tempFeature = imgFeature[i]
                #tempFeature = codes[i]
                #tempFeature = tempFeature.tostring()
                #tempFeature = tempFeature.astype(np.float32)  # b不可直接转
                #print(tempFeature.dtype) #float32
                tempId = i + 1
                tempFeatBin = pickle.dumps(tempFeature)
                print(tempId)
                cursor.execute('insert into image (id, label, imgPath, feature) VALUES (?, ?, ?, ?)',(tempId,  tempLabel, file, sqlite3.Binary(tempFeatBin)))
                #cursor.execute('insert into image VALUES (%d,%s,%s,%s)',([tempId,  tempLabel, file, tempFeature]))
                i = i + 1
    print(cursor.rowcount) #reusult 1
    cursor.close()
    conn.commit()
    conn.close()

def main():
    getLabels()
    insertDb()
if __name__ == '__main__':
    main()