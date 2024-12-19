import pandas as pd
import numpy as np

# 读取catering_sale.xlsx文件
df = pd.read_excel('data/catering_sale.xls')

# 将异常值置为空
df['销量'] = df['销量'].apply(lambda x: x if x is None or (400 <= x <= 5000) else None)

# 执行牛顿插值
def newton_interpolation(x, y, x_interp):
    n = len(x)
    coefficients = y.copy()

    for i in range(1, n):
        coefficients[i:] = (coefficients[i:] - coefficients[i - 1]) / (x[i:] - x[i - 1])

    result = coefficients[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x_interp - x[i]) + coefficients[i]

    return result

# 设置前后数据点的数量
window_size = 5

# 填充缺失值
for index, row in df.iterrows():
    if pd.isna(row['销量']):
        start_idx = max(0, index - window_size)
        end_idx = min(len(df) - 1, index + window_size)

        # 获取前后 5 个未缺失的数据
        valid_data = df['销量'].iloc[start_idx:end_idx + 1].dropna()

        if len(valid_data) > 0:
            # 执行牛顿插值
            interpolated_value = newton_interpolation(
                valid_data.index, valid_data.values, index
            )
            df.at[index, '销量'] = interpolated_value

# 导出处理后的数据到sales.xlsx
df.to_excel('sales1.xlsx', index=False)
