import pickle
import numpy as np
# 将之前保存的npy文件转换成pkl
def createPkl():
    CODES = "codes.npy"
    codes = None
    if CODES:
        codes = np.load(CODES) #存在有图片特征值的文件 就加载
    else:
        print("No such file,please run get_feature.py first")
    output = open('imageData.pkl', 'wb')
    pickle.dump(codes, output)
    output.close()

def main():
    createPkl()
if __name__ == '__main__':
    main()

