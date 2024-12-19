#抽取一个函数
import numpy as np
import collections as c

def knn(k,predictPoint,feature,label):
    # 计算每个投掷点距离predictPoint的距离
    distance = list(map(lambda x: abs(predictPoint - x), feature))
    # 对distance的集合 元素从小到大排序（返回的是排序的下标位置)
    sortindex = (np.argsort(distance))
    # 用排序的sortindex来操作 label集合
    sortedlabel = (label[sortindex])
    return (c.Counter(sortedlabel[0:k]).most_common(1)[0][0])

if __name__ == '__main__':
    data = np.array([
        [154, 1],
        [126, 2],
        [70, 2],
        [196, 2],
        [161, 2],
        [371, 4]
    ])
    # 输入值
    feature = (data[:, 0])
    # 结果label
    label = data[:, -1]
    # 预测点
    predictPoint = 200
    k = 3
    print(knn(k, predictPoint, feature, label))