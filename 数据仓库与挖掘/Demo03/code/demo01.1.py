import pandas as pd
from scipy.interpolate import lagrange

# 读取 Excel 文件
inputfile = 'data/catering_sale.xls'
outputfile = '../tmp/sales.xls'

data = pd.read_excel(inputfile)

# 对销量小于 400 或大于 5000 的值设置为空值（None）
data.loc[(data['销量'] < 400) | (data['销量'] > 5000), '销量'] = None

def ployinterp_column(s, n, k=5):
    y = s.iloc[max(0, n - k):n].append(s.iloc[n+1:min(n+1+k, len(s))])
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)

# 遍历数据集中的每一列和每一行，如果某个单元格为空值，则调用 ployinterp_column 函数来填充该单元格
for i in data.columns:
    for j in range(len(data)):
        if pd.isnull(data[i][j]):
            data.loc[j, i] = ployinterp_column(data[i], j)

# 将处理后的数据保存到新的 Excel 文件
data.to_excel(outputfile, index=False)
