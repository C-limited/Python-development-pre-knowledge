#-*- coding: utf-8 -*-
#数据清洗，过滤掉不符合规则的数据
import numpy as np
import pandas as pd

datafile= 'data/air_data.xlsx' #航空原始数据,第一行为属性标签
cleanedfile = '../tmp/data_cleaned.xlsx' #数据清洗后保存的文件

data = pd.read_excel(datafile) #读取原始数据，指定UTF-8编码（需要用文本编辑器将数据装换为UTF-8编码）

data = data[data['SUM_YR_1'].notnull()*data['SUM_YR_2'].notnull()] #票价非空值才保留

#只保留票价非零的，或者平均折扣率与总飞行公里数同时为0的记录。
index1 = data['SUM_YR_1'] != 0
index2 = data['SUM_YR_2'] != 0
index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0) #该规则是“与”
data = data[index1 | index2 | index3] #该规则是“或”

data.to_excel(cleanedfile) #导出结果

## 选取需求特征
airline_selection = data[["FFP_DATE", "LOAD_TIME", "FLIGHT_COUNT", "LAST_TO_END", "avg_discount", "SEG_KM_SUM"]]

## 构建L特征
L = pd.to_datetime(airline_selection["LOAD_TIME"]) - pd.to_datetime(airline_selection["FFP_DATE"])

##提取数字，由于模型中L单位为：月，所以需要除以30
# L = L.astype("str").str.split(' ').str[0]
# L = L.astype("int")/30

# 对L这一列应用lambda函数，对L中的每一个x都执行函数操作
L = L.apply(lambda x: round(int(str(x).split(' ')[0]) / 30, 2))

### 合并特征
airline_features = pd.concat([L, airline_selection.iloc[:, 2:]], axis=1).rename(columns = {0:'L'})
# airline_features = airline_features.rename(columns={0: 'L'})
print('构建的LRFMC特征前5行为：\n', airline_features.head())

##标准差标准化: 使用sklearn 中preprocessing 模块的StandardScaler 函数;
# 也可以使用自定义的方法（数据分析中标准化方法,因为此处不需要对训练集与测试集用同一套规则）
from sklearn.preprocessing import StandardScaler  ##标准差标准化

data = StandardScaler().fit_transform(airline_features)
np.savez('airline_scale.npz', data)
print('标准化后LRFMC五个特征为：\n', data[:5, :])
