import pandas as pd

infile = 'electricity_data.xls'  # 供入供出电量数据
outfile = 'electricity_data1.xls'

data = pd.read_excel(infile)
data[u'线损率'] = (data[u'供出电量']) / data[u'供入电量']

data.to_excel(outfile, index=False)
