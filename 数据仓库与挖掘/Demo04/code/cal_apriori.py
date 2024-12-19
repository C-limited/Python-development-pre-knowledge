#-*- coding: utf-8 -*-
#使用Apriori算法挖掘菜品订单关联规则
from apriori import * #导入apriori函数

inputfile = 'data/menu_orders_1.csv'
outputfile = 'result/apriori_rules.xls' #结果文件
data = pd.read_csv(inputfile, header = None)
#使用矩阵链接的方式
print(u'\n转换原始数据至0-1矩阵...')
ct = lambda x : pd.Series(1, index = x[pd.notnull(x)]) #转换0-1矩阵的过渡函数
b = map(ct, data.to_numpy()) #用map方式执行
data = pd.DataFrame(list(b)).fillna(0) #实现矩阵转换，空值用0填充
print(u'\n转换完毕。')
del b #删除中间变量b，节省内存

support = 0.2 #最小支持度
confidence = 0.2 #最小置信度
ms = '->' #连接符，默认'->'

find_rule(data, support, confidence, ms).to_excel(outputfile,engine='openpyxl') #保存结果


