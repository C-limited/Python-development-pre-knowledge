import numpy as np

def filter(str):
    if int(str.decode()) == 5:
        return 1
    else :
        return 0

feature = np.loadtxt("train_image.csv",delimiter=",",max_rows=3000) / 255
print(feature)
featureMatrix = np.append(feature,np.ones(shape=(len(feature), 1)),axis=1)
weight = np.ones(785)
learingrate = 0.0001
#获取label , 假设识别数字4
label = np.loadtxt("train_labels.csv",delimiter=",",max_rows=3000,converters={0:filter})


def grandentDecent():
    predict = 1 / (1 + np.exp(-np.dot(featureMatrix, weight)))
    slop = np.dot(featureMatrix.T, (predict - label))
    return slop

def train():
    for i in range(1,5000):
        slop = grandentDecent()
        global weight
        weight = weight - learingrate*slop

    return weight


if __name__ == '__main__':
    #训练
    weight = train()
    testfeature = np.loadtxt("test_image.csv", delimiter=",", max_rows=100) / 255
    for index, item in enumerate(testfeature):
        print(1/(1+np.exp(-(np.dot(np.append(item,1), weight)))))
        print(index)