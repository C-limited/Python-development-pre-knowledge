Pandas高级处理
    缺失值处理
    数据离散化
    合并
    交叉表与透视表
    分组与聚合
    综合案例

4.6 高级处理-缺失值处理
    1）如何进行缺失值处理
        两种思路：
            1）删除含有缺失值的样本
            2）替换/插补
        4.6.1 如何处理nan
            1）判断数据中是否存在NaN
                pd.isnull(df)
                pd.notnull(df)
            2）删除含有缺失值的样本
                df.dropna(inplace=False)
               替换/插补
                df.fillna(value, inplace=False)
         4.6.2 不是缺失值nan，有默认标记的
            1）替换 ？-> np.nan
                df.replace(to_replace="?", value=np.nan)
            2）处理np.nan缺失值的步骤
    2）缺失值处理实例
4.7 高级处理-数据离散化
    性别 年龄
A    1   23
B    2   30
C    1   18
    物种 毛发
A    1
B    2
C    3
    男 女 年龄
A   1  0  23
B   0  1  30
C   1  0  18

    狗  猪  老鼠 毛发
A   1   0   0   2
B   0   1   0   1
C   0   0   1   1
one-hot编码&哑变量
4.7.1 什么是数据的离散化
    原始的身高数据：165，174，160，180，159，163，192，184
4.7.2 为什么要离散化
4.7.3 如何实现数据的离散化
    1）分组
        自动分组sr=pd.qcut(data, bins)
        自定义分组sr=pd.cut(data, [])
    2）将分组好的结果转换成one-hot编码
        pd.get_dummies(sr, prefix=)
4.8 高级处理-合并
    numpy
        np.concatnate((a, b), axis=)
        水平拼接
            np.hstack()
        竖直拼接
            np.vstack()
    1）按方向拼接
        pd.concat([data1, data2], axis=1)
    2）按索引拼接
        pd.merge实现合并
        pd.merge(left, right, how="inner", on=[索引])
4.9 高级处理-交叉表与透视表
    找到、探索两个变量之间的关系
    4.9.1 交叉表与透视表什么作用
    4.9.2 使用crosstab(交叉表)实现
        pd.crosstab(value1, value2)
    4.9.3 pivot_table
4.10 高级处理-分组与聚合
    4.10.1 什么是分组与聚合
    4.10.2 分组与聚合API
        dataframe
        sr
