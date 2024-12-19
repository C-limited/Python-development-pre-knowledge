import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 从CSV文件加载数据
file_path = 'D:/University/DataWarehouse/assignment4/menu_orders_1.csv'
df = pd.read_csv(file_path)

# 对数据进行 one-hot 编码
df_encoded = pd.get_dummies(df, columns=df.columns)

# 使用Apriori算法查找频繁项集
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

# 根据频繁项集生成关联规则
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)

# 打印频繁项集和关联规则
print("\n频繁项集:")
print(frequent_itemsets)

print("\n关联规则:")
print(rules)
