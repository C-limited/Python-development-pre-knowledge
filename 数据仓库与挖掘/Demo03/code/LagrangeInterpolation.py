import pandas as pd
from scipy.interpolate import lagrange

# 读取catering_sale.xlsx文件
df = pd.read_excel('catering_sale.xlsx')

# 将异常值置为空
df['销量'] = df['销量'].apply(lambda x: x if x is None or (400 <= x <= 5000) else None)

# 执行拉格朗日插值
def lagrange_interpolation(x, y, x_interp):
    interp_func = lagrange(x, y)
    return interp_func(x_interp)

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
            # 执行拉格朗日插值
            interpolated_value = lagrange_interpolation(
                valid_data.index, valid_data.values, index
            )
            df.at[index, '销量'] = interpolated_value

# 导出处理后的数据到sales.xlsx
df.to_excel('sales.xlsx', index=False)
