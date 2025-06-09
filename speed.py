import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

speed_usl = []
speed_lsl = []
for t in range(101):
    if t <= 5 or t >= 95:
        speed_usl.append(1.0+0.1)
    elif 55 >= t >= 45:
        speed_usl.append(0.1+0.1)
    elif 45 >= t > 5:
        speed_usl.append(-0.0225*t + 1.1125+0.1)
    else:
        speed_usl.append(0.0225*t - 1.0375)

    if 55 >= t >= 45:
        speed_lsl.append(0.0+0.1)
    elif t <= 5 or t >= 95:
        speed_lsl.append(0.9+0.1)
    elif 45 >= t > 5:
        speed_lsl.append(-0.0225*t + 1.0125+0.1)
    else:
        speed_lsl.append(0.0225*t - 1.1375)
x = np.arange(101)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['font.sans-serif'] = ['Times New Roman']

plt.figure(figsize=(14, 8))
plt.plot(speed_lsl, label="轧辊速度设定下限", linewidth=3)
plt.plot(speed_usl, label="轧辊速度设定上限", linewidth=3)
plt.fill_between(x, speed_lsl, speed_usl, facecolor='#90EE90', alpha=0.3)

plt.tick_params(labelsize=20)
# plt.title("Time history of set-point of ith stand’s roll speed", fontsize=40)
plt.xlabel("time/s", fontsize=28)
plt.ylabel("$U_{vi}^t$"+'/m/s', fontsize=26)

plt.legend(loc="lower right", prop={'family': 'SimSun', 'size': 16})
plt.xticks([5, 45, 55, 95], ['$t_1$', '$t_2$', '$t_3$', '$t_4$'], fontsize=25, fontname='Times New Roman')
plt.yticks([1.05, 0.15], ['$U_{vi}^1$', '$U_{vi}^2$'], fontsize=18, fontname='Times New Roman')
plt.grid()
plt.savefig(
            "C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\results\\" + 'speed' + ".svg",
            bbox_inches="tight")

plt.show()

