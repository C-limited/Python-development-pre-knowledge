# -*- coding: utf-8 -*-
# 数据规范化
import pandas as pd
import numpy as np

datafile = 'normalization_data.xls'
data = pd.read_excel(datafile, header=None)
print(data)
data1 = (data - data.min()) / (data.max() - data.min())
data2 = (data - data.mean()) / data.std()
data3 = data / 10 ** np.ceil(np.log10(data.abs().max()))  # 小数定标规范化
print (data1)
print(data2)
print(data3)
