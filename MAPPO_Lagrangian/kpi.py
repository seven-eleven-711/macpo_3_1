import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_1 = pd.read_excel("C:\\Users\\admin\\Desktop\\results.xlsx", sheet_name='Sheet1')
df_2 = pd.read_excel("C:\\Users\\admin\\Desktop\\results.xlsx", sheet_name='Sheet2')
df_3 = pd.read_excel("C:\\Users\\admin\\Desktop\\results.xlsx", sheet_name='Sheet8')
ref = {'h1': 1.505, 'h2': 0.934, 'h3': 0.607, 'h4': 0.37, 'h5': 0.252,
       't12': 17.832, 't23': 11.762, 't34': 7.928, 't45': 4.94}
t = ['t12', 't23', 't34', 't45']
m1 = []
m2 = []
m3 = []
s1 = []
s2 = []
s3 = []
for i in ['h1', 'h2', 'h3', 'h4', 'h5']:
    df1 = df_1[i].values.tolist()
    df2 = df_2[i].values.tolist()
    df3 = df_3[i].values.tolist()
    max_percent_error_pi = max([abs(value - ref[i]) * 100 for value in df2])
    max_percent_error_rl = max([abs(value - ref[i]) * 100 for value in df1])
    max_percent_error_irl = max([abs(value - ref[i]) * 100 for value in df3])
    m1.append(max_percent_error_pi)
    m2.append(max_percent_error_rl)
    m3.append(max_percent_error_irl)

    total_pi_deviation = []
    total_rl_deviation = []
    total_irl_deviation = []
    for n in range(len(df1)):
        total_pi_deviation = [abs(value - ref[i]) for value in df2]
        total_rl_deviation = [abs(value - ref[i]) for value in df1]
        total_irl_deviation = [abs(value - ref[i]) for value in df3]
    pi_std = sum(total_pi_deviation) / len(df1)
    rl_std = sum(total_rl_deviation) / len(df1)
    irl_std = sum(total_irl_deviation) / len(df1)
    s1.append(pi_std)
    s2.append(rl_std)
    s3.append(irl_std)
print(m1, m2, m3)
print(s1, s2, s3)
categories = ['h1', 'h2', 'h3', 'h4', 'h5']

plt.figure(figsize=(14, 8))
# 全局设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
bar_width = 0.2  # 条形宽度
bar_positions_group1 = np.arange(5)
bar_positions_group2 = bar_positions_group1 + bar_width
bar_positions_group3 = bar_positions_group1 + bar_width * 2
plt.tick_params(labelsize=32)
plt.bar(bar_positions_group1, m1, width=bar_width, label='PI', color='#2171B5', edgecolor='black', linewidth=2)
plt.bar(bar_positions_group2, m3, width=bar_width, label='IPPO-L', color='#96C37D', edgecolor='black', linewidth=2)
plt.bar(bar_positions_group3, m2, width=bar_width, label='HAPPO-L', color='#D95135', edgecolor='black', linewidth=2)
plt.xlabel('出口厚度', fontdict={"family": "SimSun", "size": 32, "color": "k"})
plt.ylabel('MEP/%', fontsize=32)
# plt.title('减速', fontsize=30, fontdict={"family": "SimSun", "size": 30, "color": "k"})
plt.legend(prop={'size': 32})
plt.xticks(bar_positions_group1 + bar_width, categories)
plt.savefig('C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\results\\Deceleration\\' + 'Max_APE_h.svg', dpi=1200, bbox_inches="tight")

categories = ['h1', 'h2', 'h3', 'h4', 'h5']
plt.figure(figsize=(14, 8))
bar_width = 0.2  # 条形宽度
bar_positions_group1 = np.arange(5)
bar_positions_group2 = bar_positions_group1 + bar_width
bar_positions_group3 = bar_positions_group1 + bar_width * 2
plt.tick_params(labelsize=32)
plt.bar(bar_positions_group1, s1, width=bar_width, label='PI', color='#2171B5', edgecolor='black', linewidth=2)
plt.bar(bar_positions_group2, s3, width=bar_width, label='IPPO-L', color='#96C37D', edgecolor='black', linewidth=2)
plt.bar(bar_positions_group3, s2, width=bar_width, label='HAPPO-L', color='#D95135', edgecolor='black', linewidth=2)
plt.xlabel('出口厚度', fontdict={"family": "SimSun", "size": 32, "color": "k"})
plt.ylabel('MAD/mm', fontsize=32)
# plt.title('减速', fontsize=30, fontdict={"family": "SimSun", "size": 30, "color": "k"})
plt.legend(prop={'size': 32})
plt.xticks(bar_positions_group1 + bar_width, categories)
plt.savefig('C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\results\\Deceleration\\' + 'MAE_h.svg', dpi=1200, bbox_inches="tight")


m1 = []
m2 = []
m3 = []
s1 = []
s2 = []
s3 = []
for i in ['t12', 't23', 't34', 't45']:
    df1 = df_1[i].values.tolist()
    df2 = df_2[i].values.tolist()
    df3 = df_3[i].values.tolist()
    max_percent_error_pi = max([abs(value - ref[i]) * 100 for value in df2])
    max_percent_error_rl = max([abs(value - ref[i]) * 100 for value in df1])
    max_percent_error_irl = max([abs(value - ref[i]) * 100 for value in df3])
    m1.append(max_percent_error_pi)
    m2.append(max_percent_error_rl)
    m3.append(max_percent_error_irl)

    total_pi_deviation = []
    total_rl_deviation = []
    total_irl_deviation = []
    for n in range(len(df1)):
        total_pi_deviation = [abs(value - ref[i]) for value in df2]
        total_rl_deviation = [abs(value - ref[i]) for value in df1]
        total_irl_deviation = [abs(value - ref[i]) for value in df3]
    pi_std = sum(total_pi_deviation) / len(df1)
    rl_std = sum(total_rl_deviation) / len(df1)
    irl_std = sum(total_irl_deviation) / len(df1)
    s1.append(pi_std)
    s2.append(rl_std)
    s3.append(irl_std)
categories = ['t12', 't23', 't34', 't45']
plt.figure(figsize=(14, 8))
bar_width = 0.2  # 条形宽度
bar_positions_group1 = np.arange(4)
bar_positions_group2 = bar_positions_group1 + bar_width
bar_positions_group3 = bar_positions_group1 + bar_width * 2
plt.tick_params(labelsize=32)
plt.bar(bar_positions_group1, m1, width=bar_width, label='PI', color='#2171B5', edgecolor='black', linewidth=2)
plt.bar(bar_positions_group2, m3, width=bar_width, label='IPPO-L', color='#96C37D', edgecolor='black', linewidth=2)
plt.bar(bar_positions_group3, m2, width=bar_width, label='HAPPO-L', color='#D95135', edgecolor='black', linewidth=2)
plt.xlabel('机架间张力', fontdict={"family": "SimSun", "size": 32, "color": "k"})
plt.ylabel('MEP/%', fontsize=32)
# plt.title('减速', fontsize=30, fontdict={"family": "SimSun", "size": 30, "color": "k"})
plt.legend(prop={'size': 32})
plt.xticks(bar_positions_group1 + bar_width / 2, categories)
plt.savefig('C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\results\\Deceleration\\' + 'Max_APE_t.svg', dpi=1200, bbox_inches="tight")


categories = ['t12', 't23', 't34', 't45']
plt.figure(figsize=(14, 8))
bar_width = 0.2  # 条形宽度
bar_positions_group1 = np.arange(4)
bar_positions_group2 = bar_positions_group1 + bar_width
bar_positions_group3 = bar_positions_group1 + bar_width * 2
plt.tick_params(labelsize=32)
plt.bar(bar_positions_group1, s1, width=bar_width, label='PI', color='#2171B5', edgecolor='black', linewidth=2)
plt.bar(bar_positions_group2, s3, width=bar_width, label='IPPO-L', color='#96C37D', edgecolor='black', linewidth=2)
plt.bar(bar_positions_group3, s2, width=bar_width, label='HAPPO-L', color='#D95135', edgecolor='black', linewidth=2)

plt.xlabel('机架间张力', fontdict={"family": "SimSun", "size": 32, "color": "k"})
plt.ylabel('MAD/kN', fontsize=32)
# plt.title('减速', fontsize=30, fontdict={"family": "SimSun", "size": 30, "color": "k"})
plt.legend(prop={'size': 32})
plt.xticks(bar_positions_group1 + bar_width / 2, categories)
plt.savefig('C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\results\\Deceleration\\' + 'MAE_t.svg', dpi=1200, bbox_inches="tight")
