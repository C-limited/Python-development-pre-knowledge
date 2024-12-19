import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 从Excel文件中读取数据
datafile = 'discretization_data.xls'  # 假设这是一个Excel文件
data = pd.read_excel(datafile)
data = data[u'肝气郁结证型系数'].copy()
k = 4

# 等频率离散化
d1 = pd.cut(data, k, labels=range(k))

# 等宽离散化
w = [1.0 * i / k for i in range(k + 1)]
w = data.describe(percentiles=w)[4:4 + k + 1]
w[0] = w[0] * (1 - 1e-10)
d2 = pd.cut(data, w, labels=range(k))

# K均值聚类
kmodel = KMeans(n_clusters=k)
kmodel.fit(data.values.reshape(-1, 1))
c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
w = c.rolling(2).mean().iloc[1:]  # 计算相邻聚类中心之间的中点
w = [0] + list(w[0]) + [data.max()]  # 添加第一个和最后一个边界
d3 = pd.cut(data, w, labels=range(k))

def cluster_plot(d, k):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文字符
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 3))
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')
    plt.ylim(-0.5, k - 0.5)
    plt.show()  # 使用plt.show()来显示图表

cluster_plot(d1, k)
cluster_plot(d2, k)
cluster_plot(d3, k)
