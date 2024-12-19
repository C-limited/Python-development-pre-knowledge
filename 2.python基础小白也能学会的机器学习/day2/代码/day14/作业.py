#预测房价， 经度，维度，房屋使用面积，房价
import  numpy as np

data = np.array([
    [47.5112, -122.257, 1180, 221900],
    [48.5112, -123.257, 1280, 231900],
    [46.5112, -124.257, 1380, 241900],
    [45.5112, -121.257, 1480, 251900],
    [43.5112, -127.257, 1580, 261900],
])

for i in range(0,len(data.T)-1):
    feature = (data.T[i])
    mean = (np.mean(feature))
    std = np.std(feature)
    data.T[i] = (((feature - mean) / std))

print(data)