import numpy as np


def color2num(str):
    dict = {"红":0.50, "黄":0.51,"蓝":0.52,"绿":0.53,"紫":0.54,"粉":0.55}
    return dict[str]
data = np.loadtxt("data1.csv",delimiter=",",converters={1: color2num},encoding="gbk")
print(data)