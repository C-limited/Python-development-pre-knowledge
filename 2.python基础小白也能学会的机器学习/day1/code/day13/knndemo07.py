import numpy as np

def knn(k,predictPoint,feature,label):
    matrixtemp = (feature - predictPoint)
    matrixtemp2 = np.square(matrixtemp)
    # axis=1 代表逐行相加
    sortindex = np.argsort(np.sqrt(np.sum(matrixtemp2, axis=1)))
    sortlabel = label[sortindex]
    predictprice = np.sum(sortlabel[0:k]) / k
    return predictprice

if __name__ == '__main__':
    feature = np.loadtxt("kc_house_data.csv",delimiter=",",skiprows=1,usecols=(17,18,6))
    label = np.loadtxt("kc_house_data.csv",delimiter=",",skiprows=1,usecols=(2))
    predictPoint = np.array([47.5112,-122.257,5650])
    print(knn(450, predictPoint, feature, label))