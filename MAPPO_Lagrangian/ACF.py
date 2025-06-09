import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def acf(ts, k):
    # 预计算-均先减去均值（原始序列的均值）
    x = np.array(ts) - np.mean(ts)

    # 存储自相关系数（[c0,c1,...,ck]）
    coef = np.zeros(k + 1)

    # 计算r0原始序列的标准差（有偏:T,无偏:T-k（分子分母可以消掉，这里就省略了除以分母））
    coef[0] = x.dot(x)

    # 循环计算第i阶自协方差
    for i in range(1, k + 1):
        coef[i] = x[:-i].dot(x[i:])

    # 返回自相关系数
    return coef / coef[0]


df_1 = pd.read_excel("C:\\Users\\admin\\Desktop\\results.xlsx", sheet_name='Sheet4')
df_2 = pd.read_excel("C:\\Users\\admin\\Desktop\\results.xlsx", sheet_name='Sheet5')
for i in ['h1', 'h2', 'h3', 'h4', 'h5', 't12', 't23', 't34', 't45']:
    df1 = df_1[i].values.tolist()
    df2 = df_2[i].values.tolist()
    print(acf(df1, 1))
    # print(acf(df2, 1))

