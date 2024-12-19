#抽取一个函数
import numpy as np
import collections as c

#考虑的是1个维度的数据，（position）  feature [130,200,35,....]
def knn(k,predictPoint,feature,label):
    # 计算每个投掷点距离predictPoint的距离
    distance = list(map(lambda x: abs(predictPoint - x), feature))
    # 对distance的集合 元素从小到大排序（返回的是排序的下标位置)
    sortindex = (np.argsort(distance))
    # 用排序的sortindex来操作 label集合
    sortedlabel = (label[sortindex])
    return (c.Counter(sortedlabel[0:k]).most_common(1)[0][0])

# feature [ [130,0.55], [100,0.51]..... ]
def knn2(k,predictPoint,ballcolor,feature,label):
    # 计算每个投掷点距离(predictPoint,ballcolor)的距离
    distance = list(map(lambda item:((item[0]-predictPoint)**2+(item[1]-ballcolor)**2)**0.5,feature))
    # 对distance的集合 元素从小到大排序（返回的是排序的下标位置)
    sortindex = (np.argsort(distance))
    # 用排序的sortindex来操作 label集合
    sortedlabel = (label[sortindex])
    return (c.Counter(sortedlabel[0:k]).most_common(1)[0][0])

# 数据归一化 normalzation 的knn
def knn3(k,predictPoint,ballcolor,feature,label):
    # 计算每个投掷点距离(predictPoint,ballcolor)的距离
    distance = list(map(lambda item:((item[0]/475-predictPoint/475)**2+((item[1]-0.50)/0.05-(ballcolor-0.50)/0.05)**2)**0.5,feature))
    # 对distance的集合 元素从小到大排序（返回的是排序的下标位置)
    sortindex = (np.argsort(distance))
    # 用排序的sortindex来操作 label集合
    sortedlabel = (label[sortindex])
    return (c.Counter(sortedlabel[0:k]).most_common(1)[0][0])

if __name__ == '__main__':
    traindata = np.loadtxt("data2-train.csv",delimiter=",")
    # 输入值
    feature = (traindata[:, 0:2])
    # 结果label
    label = traindata[:, -1]
    # 预测点，来自测试数据集的每一条记录
    testdata = np.loadtxt("data2-test.csv",delimiter=",")
    k = 36
    count = 0
    for item in testdata:
        predict = knn3(k,item[0],item[1],feature,label)
        real = item[-1]
        if predict == real:
            count = count+1
    print("准确率：{}%".format(count*100.0/len(testdata)))