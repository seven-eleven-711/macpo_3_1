import pandas as pd
import matplotlib.pyplot as plt

file_path = "C:\\Users\\admin\\Desktop\\1\\episode_costs_aver_costs.csv"
df = pd.read_csv(file_path)
fig, ax = plt.subplots(figsize=(16, 9))

# 全局设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# plt.figure(figsize=(16, 9))
# df["HAPPO"].plot()
# df["MAPPO-L"].plot()
# df["IPPO-L"].plot()
ax.plot(df["HAPPO"], label='HAPPO', c='b')
ax.plot(df["MAPPO-L"], label='HAPPO-L', c='orange')
ax.plot(df["IPPO-L"], label='IPPO-L', c='g')
# 设置 x 轴和 y 轴的刻度标签字体
for label in ax.get_xticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(14)

for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(14)

plt.tick_params(labelsize=25)
# plt.title("减速", fontdict={"family": "SimSun", "size": 30, "color": "k"})
plt.xlabel("训练周期", fontdict={"family": "SimSun", "size": 25, "color": "k"})
plt.text(0.58, -0.1, "(×3)", fontsize=25, fontfamily="Times New Roman", ha="center",
         transform=plt.gca().transAxes)
plt.ylabel("平均代价值", fontdict={"family": "SimSun", "size": 25, "color": "k"})
plt.legend(prop={'size': 22})
plt.grid()
plt.savefig(
    "C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\results\\Deceleration\\cost.svg",
    dpi=1200, bbox_inches='tight')

##
file_path = "C:\\Users\\admin\\Desktop\\1\\episode_rewards_aver_rewards.csv"
df = pd.read_csv(file_path)
fig, ax = plt.subplots(figsize=(16, 9))

ax.plot(df["HAPPO"], label='HAPPO', c='b')
ax.plot(df["MAPPO-L"], label='HAPPO-L', c='orange')
ax.plot(df["IPPO-L"], label='IPPO-L', c='g')
# 设置 x 轴和 y 轴的刻度标签字体
for label in ax.get_xticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(14)

for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(14)

plt.tick_params(labelsize=25)
# plt.title("减速", fontdict={"family": "SimSun", "size": 30, "color": "k"})
plt.xlabel("训练周期", fontdict={"family": "SimSun", "size": 25, "color": "k"})
plt.text(0.58, -0.1, "(×3)", fontsize=25, fontfamily="Times New Roman", ha="center",
         transform=plt.gca().transAxes)
plt.ylabel("平均奖励值", fontdict={"family": "SimSun", "size": 25, "color": "k"})
plt.legend(prop={'size': 22})
plt.grid()
plt.savefig(
    "C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\results\\Deceleration\\reward.svg",
    dpi=1200, bbox_inches='tight')
