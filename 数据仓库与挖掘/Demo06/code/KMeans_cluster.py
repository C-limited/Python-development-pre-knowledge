#-*- coding: utf-8 -*-
#K-Means聚类算法
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans #导入K均值聚类算法

inputfile = 'tmp/zscoreddata.xlsx' #待聚类的数据文件
k = 5                       #需要进行的聚类类别数

#读取数据并进行聚类分析
data = pd.read_excel(inputfile) #读取数据

#调用k-means算法，进行聚类分析
kmodel = KMeans(n_clusters = k)
kmodel.fit(data) #训练模型

print(kmodel.cluster_centers_) #查看聚类中心
print(kmodel.labels_) #查看各样本对应的类别
# 绘制聚类结果的散点图
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=kmodel.labels_, cmap='viridis')
# 绘制聚类中心点
plt.scatter(kmodel.cluster_centers_[:, 0], kmodel.cluster_centers_[:, 1], marker='x', color='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Results')
plt.show()


