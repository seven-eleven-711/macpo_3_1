import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

df_1 = pd.read_excel("C:\\Users\\admin\\Desktop\\1\\result+\\results.xlsx", sheet_name='Sheet1')
df_2 = pd.read_excel("C:\\Users\\admin\\Desktop\\1\\result+\\results.xlsx", sheet_name='Sheet2')
df_3 = pd.read_excel("C:\\Users\\admin\\Desktop\\1\\result+\\results.xlsx", sheet_name='Sheet8')
ref = {'h1': 1.505, 'h2': 0.934, 'h3': 0.607, 'h4': 0.37, 'h5': 0.252,
       't12': 17.832, 't23': 11.762, 't34': 7.928, 't45': 4.94}

I = 0

# 全局设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

fig = plt.figure(figsize=(14, 20), constrained_layout=True)

plt.style.use('seaborn-paper')
plt.tight_layout()
for i in ['h1', 'h2', 'h3', 'h4', 'h5']:

    I = I+1
    df1 = df_1[i].values.tolist()
    df2 = df_2[i].values.tolist()
    df3 = df_3[i].values.tolist()
    ref_ = [ref[i] for n in np.arange(0, 5.1, 0.1)]
    ax1 = fig.add_subplot(5, 1, I)

    x = np.arange(0, 5.1, 0.1)
    x0 = np.arange(0, 5.05, 0.05)

    ax1.plot(x, df1, label="HAPPO-L", color='k', linewidth=3)
    ax1.plot(x, df3, label="IPPO-L", color='g', linewidth=3, alpha=0.8)
    ax1.plot(x0, df2, label="PI", color='b', linewidth=3, linestyle='--', alpha=0.8)
    ax1.plot(x, ref_, label="ref", color='r', linewidth=3)
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.tick_params(labelsize=20)
    ax1.set_xticks(np.arange(0, 5.1, step=1))
    # plt.title("加速", fontdict={"family": "SimSun", "size": 30, "color": "k"})
    if I==5:
        ax1.set_xticklabels([f'{i}' for i in np.arange(0, 5.1, 1)])
        # plt.xticks(np.arange(0, 5.1, step=1))
        plt.xlabel("时间/", fontdict={"family": "SimSun", "size": 20, "color": "k"}, horizontalalignment='center')
        # 添加额外的文本
        plt.text(0.535, -0.18, "s", fontsize=25, fontfamily="Times New Roman", ha="center",
                 transform=plt.gca().transAxes)

    plt.ylabel(i + "/mm", fontdict={"family": "Times New Roman", "size": 20, "color": "k"})
    plt.gca().yaxis.set_major_formatter('{:.3f}'.format)
    if I ==1:
        legend = plt.legend(prop={'size': 18}, loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=4)
        for handle in legend.legendHandles:
            handle.set_alpha(0.8)
    plt.grid()
plt.savefig(
            "C:\\Users\\admin\\Desktop\\1\\" + 'H' + ".svg",
            bbox_inches="tight")


I = 0
fig = plt.figure(figsize=(14, 20), constrained_layout=True)

plt.style.use('seaborn-paper')
plt.tight_layout()
for i in ['t12', 't23', 't34', 't45']:

    I = I+1
    df1 = df_1[i].values.tolist()
    df2 = df_2[i].values.tolist()
    df3 = df_3[i].values.tolist()
    ref_ = [ref[i] for n in np.arange(0, 5.1, 0.1)]
    ax1 = fig.add_subplot(5, 1, I)
    # 设置英文字体为 Times New Roman

    x = np.arange(0, 5.1, 0.1)
    x0 = np.arange(0, 5.05, 0.05)

    ax1.plot(x, df1, label="HAPPO-L", color='k', linewidth=3)
    ax1.plot(x, df3, label="IPPO-L", color='g', linewidth=3, alpha=0.8)
    ax1.plot(x0, df2, label="PI", color='b', linewidth=3, linestyle='--', alpha=0.8)
    ax1.plot(x, ref_, label="ref", color='r', linewidth=3)
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.tick_params(labelsize=20)
    ax1.set_xticks(np.arange(0, 5.1, step=1))
    # plt.title("加速", fontdict={"family": "SimSun", "size": 30, "color": "k"})
    if I==4:
        ax1.set_xticklabels([f'{i}' for i in np.arange(0, 5.1, 1)])
        # plt.xticks(np.arange(0, 5.1, step=1))
        plt.xlabel("时间/", fontdict={"family": "SimSun", "size": 20, "color": "k"}, horizontalalignment='center')
        # 添加额外的文本
        plt.text(0.535, -0.18, "s", fontsize=25, fontfamily="Times New Roman", ha="center",
                 transform=plt.gca().transAxes)

    plt.ylabel(i + "/kN", fontsize=20)

    plt.gca().yaxis.set_major_formatter('{:.1f}'.format)
    # 只在第一个子图显示图例
    if I ==1:
        legend = plt.legend(prop={'size': 18}, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=4)
        for handle in legend.legendHandles:
            handle.set_alpha(0.8)
    plt.grid()
plt.savefig(
            "C:\\Users\\admin\\Desktop\\1\\" + 'T' + ".svg",
            bbox_inches="tight")

