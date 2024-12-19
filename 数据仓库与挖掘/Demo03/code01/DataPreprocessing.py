import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 生成一些随机数据
np.random.seed(42)  # 用于可重复性
data = pd.DataFrame({
    'A': np.random.rand(10),
    'B': np.random.rand(10),
    'C': np.random.rand(10),
    'D': [1, 2, 2, 3, 3, 4, 5, 5, 5, 6]  # 一些重复值用于测试
})

print("原始数据:")
print(data)

# 插值处理缺失值
data.interpolate(inplace=True)
print("\n插值后:")
print(data)

# 去除重复值
data_unique = data.drop_duplicates()
print("\n去除重复值后:")
print(data_unique)

# 检查空值
print("\n空值检查:")
print(data.isnull())

# 检查非空值
print("\n非空值检查:")
print(data.notnull())

# 主成分分析（PCA）
pca = PCA()
data_pca = pca.fit_transform(data[['A', 'B', 'C']])
data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3'])
print("\nPCA 结果:")
print(data_pca)

# 生成随机矩阵
random_matrix = pd.DataFrame(np.random.rand(5, 5), columns=['X', 'Y', 'Z', 'W', 'K'])
print("\n随机矩阵:")
print(random_matrix)
