import pandas as pd
from scipy.interpolate import lagrange

inputfile = 'data/catering_sale.xls'
outputfile = '../tmp/sales.xls'

data = pd.read_excel(inputfile)

data['销量'][(data['销量'] < 400) | (data['销量'] > 5000)] = None

def ployinterp_column(s, n, k=5):
    y = s.iloc[max(0, n - k):n] + s.iloc[n+1:min(n+1+k, len(s))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)

for i in data.columns:
    for j in range(len(data)):
        if pd.isnull(data[i][j]):
            data[i][j] = ployinterp_column(data[i], j)

data.to_excel(outputfile)