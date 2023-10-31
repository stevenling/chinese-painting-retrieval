import pickle
import numpy as np

def createPkl():
    """
    将之前保存的 npy 文件转换成 pkl
    """
    CODES = "codes.npy"
    codes = None
    if CODES:
        # 存在图片特征值的文件, 那么就加载
        codes = np.load(CODES) 
    else:
        print("No such file,please run get_feature.py first")
    output = open('imageData.pkl', 'wb')
    pickle.dump(codes, output)
    output.close()

def main():
    createPkl()

if __name__ == '__main__':
    main()