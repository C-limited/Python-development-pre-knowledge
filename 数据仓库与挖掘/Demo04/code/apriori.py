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

    support_series = 1.0 * d.sum() / len(d)  # 支持度序列
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
        print('数目：%s' % len(column))
        if len(column) == 0:
            break
        sf = lambda i: d[i].prod(axis=1, numeric_only=True)  # 新一批支持度的计算函数

        # 创建连接数据
        d_2 = pd.DataFrame(list(map(sf, column)), index=[ms.join(i) for i in column]).T

        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d)  # 计算连接后的支持度
        print('\n%s-候选集' % k)
        print(support_series_2)
        column = list(support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选

        support_series = pd.concat([support_series,support_series_2])
        print('\n%s-频繁集' % k)
        print(support_series_2[support_series_2 > support])

        column2 = []

        for i in column:
            column2.append(i.split(ms))

        confidence_series = pd.Series(index=[ms.join(i) for i in column2],dtype=np.float64)  # 定义置信度序列

        for i in column2:  # 计算置信度序列
            confidence_series[ms.join(i)] = support_series[ms.join(sorted(i))] / support_series[ms.join(i[:len(i)-1])]

        for i in confidence_series[confidence_series > confidence].index:  # 置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = confidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(by=['confidence', 'support'], ascending=False)  # 结果整理，输出
    print('\n结果为：')
    print(result)

    return result