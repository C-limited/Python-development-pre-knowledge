import pandas as pd
import numpy as np

# 自定义连接函数，用于实现L_{k-1}到C_k的连接
def connect_string(x, ms):
    x = list(map(lambda i: sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][:l-1] == x[j][:l-1] and x[i][l-1] != x[j][l-1]:
                r.append(x[i][:l-1] + sorted([x[j][l-1], x[i][l-1]]))
    return r

# 寻找关联规则的函数
def find_rule(d, support, confidence, ms='--'):
    result = pd.DataFrame(index=['support', 'confidence'])  # 定义输出结果

    # One-hot编码
    d_encoded = pd.get_dummies(d.apply(lambda x: ' '.join(x.astype(str)), axis=1).str.get_dummies(' '))

    support_series = 1.0 * d_encoded.sum() / len(d_encoded)  # 支持度序列
    column = list(support_series[support_series > support].index)  # 初步根据支持度筛选
    print('\n1-候选集：')
    print(support_series)
    print('\n1-频繁集：')
    print(support_series[support_series > support])
    k = 1
    while len(column) > 1:
        k = k + 1
        print('\n正在进行第%s次搜索...' % k)
        column = connect_string(column, ms)
        column = [sorted(c) for c in column if len(c) == k]  # 确保新生成的候选集元素是有序的
        print('数目：%s' % len(column))
        if len(column) == 0:
            break
        sf = lambda i: d_encoded[i].prod(axis=1, numeric_only=True)  # 新一批支持度的计算函数

        # 创建连接数据
        d_2 = pd.DataFrame(list(map(sf, column)), index=[ms.join(i) for i in column]).T

        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d_encoded)  # 计算连接后的支持度
        print('\n%s-候选集' % k)
        print(support_series_2)
        column = list(support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选

        support_series = pd.concat([support_series, support_series_2])
        print('\n%s-频繁集' % k)
        print(support_series_2[support_series_2 > support])

        column2 = []

        for i in column:
            column2.append(i.split(ms))

        # 在计算置信度序列时，不再进行额外的字符替换
        confidence_series = pd.Series(index=[ms.join(i) for i in column2], dtype=np.float64)

        for i in column2:  # 计算置信度序列
            key = ms.join(i)
            confidence_series[key] = support_series[key] / support_series[ms.join(i[:len(i)-1])]

        for i in confidence_series[confidence_series > confidence].index:  # 置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = confidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(by=['confidence', 'support'], ascending=False)  # 结果整理，输出
    print('\n结果为：')
    print(result)

    return result

# 假设 df 是数据框，support_threshold 和 confidence_threshold 是设定的阈值
df = pd.read_csv('D:/University/DataWarehouse/assignment4/menu_orders_1.csv')
support_threshold = 0.1  # 请根据需求设置支持度阈值
confidence_threshold = 0.5  # 请根据需求设置置信度阈值
result = find_rule(df, support_threshold, confidence_threshold)
