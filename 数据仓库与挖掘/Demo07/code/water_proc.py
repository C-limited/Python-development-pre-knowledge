import pandas as pd
import numpy as np
data = pd.read_excel('../data/original_data.xls')
data[u'发生时间'] = pd.to_datetime(data[u'发生时间'], format='%Y%m%d%H%M%S')  # 将该特征转成日期时间格式（***）
data = data[data[u'水流量'] > 0]  # 只要流量大于0的记录
# print len(data) #7679

data[u'用水停顿时间间隔'] = data[u'发生时间'].diff() / np.timedelta64(1, 'm')  # 将datetime64[ns]转成 以分钟为单位（*****）
data = data.fillna(0)  # 替换掉data[u'用水停顿时间间隔']的第一个空值
print(data.head())
data_explore = data.describe().T
data_explore['null'] = len(data)-data_explore['count']
explore = data_explore[['min','max','null']]
explore.columns = [u'最小值',u'最大值',u'空值数']
print(explore)
