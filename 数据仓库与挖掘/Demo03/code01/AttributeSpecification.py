import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE

# 读取数据
infile = 'principal_component.xls'
data = pd.read_excel(infile,  header=None)

# 分离特征和目标变量
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 目标变量

# 方法1:合并属性
# 为简化,我们直接取每行的均值作为合并后的属性
X_merged = X.mean(axis=1)

# 方法2:前向选择
# 使用决策树作为估计器的递归特征消除法(RFE)进行前向选择
estimator = DecisionTreeRegressor()
selector_forward = RFE(estimator, n_features_to_select=1, step=1)
X_forward = selector_forward.fit_transform(X, y)

# 方法3:后向消除
# 使用决策树作为估计器的递归特征消除法(RFE)进行后向消除
estimator = DecisionTreeRegressor()
selector_backward = RFE(estimator, n_features_to_select=1, step=1)
X_backward = selector_backward.fit_transform(X, y)

# 方法4:决策树归纳
# 使用决策树作为模型
tree_model = DecisionTreeRegressor()
tree_model.fit(X, y)
feature_importances = tree_model.feature_importances_

# 方法5:主成分分析(PCA)
from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# 输出或使用结果
print("合并属性:")
print(X_merged)
print("\n前向选择结果:")
print(X_forward)
print("\n后向消除结果:")
print(X_backward)
print("\n决策树特征重要性:")
print(feature_importances)
print("\nPCA的主成分:")
print(X_pca)