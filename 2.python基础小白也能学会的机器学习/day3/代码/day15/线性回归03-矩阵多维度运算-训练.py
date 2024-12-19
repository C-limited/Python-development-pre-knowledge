import numpy as np
import time
data = np.loadtxt("train-data.csv",delimiter=",")
m1 = 1
m2 = 1
b = 1
weight = np.array([
    [m1],
    [m2],
    [b]
])
feature = (data[:, 0:2]) #保留维度信息的feature
featureMatrix = (np.append(feature, (np.ones(shape=(len(feature), 1))),axis=1))
label = (np.expand_dims(data[:, -1],axis=1)) #保留了维度信息的label 真实值
learningrate = 0.00001

def grandentdecent():
    result = np.dot(featureMatrix.T,np.dot(featureMatrix,weight)-label)/len(feature)*2
    return result #结果矩阵 矩阵里面的第0行的第0个位置 是mse对b的偏导  0行第1个位置 是mse对m的偏导

def train():
    starttime = time.time()
    for i in range(1,10000000):
        result = grandentdecent()
        global weight
        weight = weight - result*learningrate
        #print(result)
        if (abs(result[0][0])<0.5 and abs(result[1][0])<0.5 and abs(result[2][0])<0.5):
            break
    print("weight={}".format(weight))
    endtime = time.time()
    print("消耗的时间{}".format(endtime-starttime))

if __name__ == '__main__':
    train()

