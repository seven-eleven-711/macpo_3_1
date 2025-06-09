import os

import numpy
import pandas as pd
import numpy as np
import torch
import random

from MAPPO_Lagrangian.mappo_lagrangian.envs.env_model import NeuralNetwork
from MAPPO_Lagrangian.mappo_lagrangian.config import get_config


class AgentEnvInteract:
    def __init__(self):
        super(AgentEnvInteract, self).__init__()

        h0_path = "C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\envs\\h0_input.xlsx"
        h0_input = pd.read_excel(h0_path, sheet_name='h0').values.squeeze().tolist()
        self.h0_normal = [(h - 2.15549)/(2.21369 - 2.15549) for h in h0_input]

        # data_max = pd.read_excel("C:\\Users\\admin\\Desktop\\macpo_3_0\\MAPPO_Lagrangian\\mappo_lagrangian\\env_build\\train_data\\input_max.xlsx", sheet_name='sheet_1').values.squeeze()
        # data_min = pd.read_excel("C:\\Users\\admin\\Desktop\\macpo_3_0\\MAPPO_Lagrangian\\mappo_lagrangian\\env_build\\train_data\\input_min.xlsx", sheet_name='sheet_1').values.squeeze()
        # self.data_normal_max = {'T12': 18.7613, 'T23': 12.3941, 'T34': 8.26798, 'T45': 5.33566, 'H1': 1.5217, 'H2': 0.944350234, 'H3': 0.616127552, 'H4': 0.375513, 'H5': 0.25541}
        # self.data_normal_min = {'T12': 17.7052, 'T23': 11.724, 'T34': 7.86168, 'T45': 4.82258, 'H1': 1.49235, 'H2': 0.921710748, 'H3': 0.600961, 'H4': 0.365711, 'H5': 0.252242}
        #
        # self.t_ref = {'T12': 17.832, 'T23': 11.762, 'T34': 7.928, 'T45': 4.94}
        # self.h_ref = {'H1': 1.505, 'H2': 0.934, 'H3': 0.607, 'H4': 0.37, 'H5': 0.252}
        # self.nt_ref = []
        # self.nh_ref = []
        # for key, value in self.t_ref.items():
        #     self.nt_ref.append((value - self.data_normal_min[key])/(self.data_normal_max[key] - self.data_normal_min[key]))
        # for key, value in self.h_ref.items():
        #     if (value - self.data_normal_min[key]) >= 0:
        #         self.nh_ref.append((value - self.data_normal_min[key])/(self.data_normal_max[key] - self.data_normal_min[key]))
        #     else:
        #         self.nh_ref.append(.0)
        self.h_ref = [0.344874591, 0.480845966, 0.429947959, 0.550069371, 0.383262411]
        self.t_ref = [0.162738409, 0.045827303, 0.091857228, 0.2288532]

        self.train_h0 = []

        # h_list = {'H1': [0, 1, 5, 6, 10, 16, 17],
        #           'H2': [1, 2, 6, 7, 11, 17, 18],
        #           'H3': [2, 3, 7, 8, 12, 18, 19],
        #           'H4': [3, 4, 8, 9, 13, 19, 20],
        #           'H5': [4, 9, 14, 20, 21]
        #           }
        h_list = {'H1': [5, 6, 10, 16, 17],
                  'H2': [6, 7, 11, 17, 18],
                  'H3': [7, 8, 12, 18, 19],
                  'H4': [8, 9, 13, 19, 20],
                  'H5': [9, 14, 20, 21]
                  }

        t_list = {'T01': [0, 5, 10, 11, 17],
                  'T12': [0, 1, 5, 6, 10, 11, 16, 17, 18],
                  'T23': [1, 2, 6, 7, 11, 12, 17, 18, 19],
                  'T34': [2, 3, 7, 8, 12, 13, 18, 19, 20],
                  'T45': [3, 4, 8, 9, 13, 14, 19, 20, 21],
                  'T56': [4, 9, 14, 15, 20]
                  }
        # 变量列表
        self.label_list = {**h_list, **t_list}
        self.model_list = {}
        for key, value in self.label_list.items():
            output_model = NeuralNetwork(len(value), 22 - len(value))
            model_path = os.path.join("C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\envs\\predict_model", key + ".pth")
            output_model.load_state_dict(torch.load(model_path))
            self.model_list[key] = output_model

        self.raw_data = self.data_read()

    # 随机初始化状态，返回初始每个智能体状态（5维列表）和全部智能体共享状态（22）
    # 目前固定初始值
    def state_init(self, model):
        if not model:
            t0 = random.sample(range(0, len(self.h0_normal) - 51), 1)[0]
            self.train_h0 = self.h0_normal[t0: t0 + 51]
        else:
            self.train_h0 = self.h0_normal[0: 51]
        sub_state = []
        sub_s = []

        # todo:加上辊缝试试
        # todo: 随机抽取初始
        share_state = [0.968929216, 0.971409334, 0.983769966, 0.98863901, 1.0,
                       0.453559267, 0.546228562, 0.479968745, 0.486629791, 0.54476495,
                       0.58814433, 0.200109051, 0.353058614, 0.286584685, 0.703990859, 0.424964539,
                       0.590136185, 0.136377733, 0.085383502, 0.127162426, 0.229379434, 0.522816901]
        # 5 or 10
        share_state[10] = self.train_h0[0]

        for i in range(5):
            if i != 4:
                sub_s.append(share_state[i])
                sub_s.append(share_state[i + 1])
                sub_s.append(share_state[i + 5])
                sub_s.append(share_state[i + 6])
                sub_s.append(share_state[i + 10])
                sub_s.append(share_state[i + 11])
                sub_s.append(share_state[i + 16])
                sub_s.append(share_state[i + 17])
            else:
                sub_s.append(share_state[i])
                sub_s.append(share_state[i - 1])
                sub_s.append(share_state[i + 5])
                sub_s.append(share_state[i + 4])
                sub_s.append(share_state[i + 10])
                sub_s.append(share_state[i + 11])
                sub_s.append(share_state[i + 16])
                sub_s.append(share_state[i + 17])
            # sub_s.append(.02)
            sub_state.append(np.array(sub_s))
            sub_s.clear()
        return sub_state, share_state, share_state

    def state_trans(self, aa, ss, t):
        # 厚度张力
        state = []
        for j in [4, 6]:
            for i in range(5):
                state.append(ss[i][j])
            state.append(ss[4][j + 1])

        a_t = []
        rolling_v = []
        rolling_s = []
        # 动作处理

        for i in range(2):
            for j in range(5):
                if i == 0:
                    a_t.append(aa[j][i])
                    rolling_v.append(aa[j][i])
                else:
                    a_t.append(aa[j][i])
                    rolling_s.append(aa[j][i])

        s_s = np.array(state)

        a_a = np.array(a_t)

        roll_v = self.action_transfer(rolling_v)
        roll_s = self.action_transfer(rolling_s)

        a_s = np.hstack((a_a, s_s))
        inputs = a_s

        outputs = []
        outputs.insert(0, self.train_h0[t + 1])
        for key, value in self.label_list.items():
            output_model = NeuralNetwork(len(value), 22 - len(value))
            model_path = os.path.join("C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\envs\\predict_model", key + ".pth")
            output_model.load_state_dict(torch.load(model_path))

            main_param = [param for i, param in enumerate(inputs) if i in value]
            aux_param = [param for i, param in enumerate(inputs) if i not in value]
            one_output = output_model(torch.Tensor(main_param), torch.Tensor(aux_param))
            outputs.append(one_output.detach()[0].double())

        next_state = np.hstack((roll_v, roll_s, outputs)).tolist()

        new_sub_state = []
        sub_s = []
        for i in range(5):
            if i != 4:
                sub_s.append(next_state[i])
                sub_s.append(next_state[i + 1])
                sub_s.append(next_state[i + 5])
                sub_s.append(next_state[i + 6])
                sub_s.append(next_state[i + 10])
                sub_s.append(next_state[i + 11])
                sub_s.append(next_state[i + 16])
                sub_s.append(next_state[i + 17])
            else:

                sub_s.append(next_state[i])
                sub_s.append(next_state[i - 1])
                sub_s.append(next_state[i + 5])
                sub_s.append(next_state[i + 4])
                sub_s.append(next_state[i + 10])
                sub_s.append(next_state[i + 11])
                sub_s.append(next_state[i + 16])
                sub_s.append(next_state[i + 17])

            new_sub_state.append(np.array(sub_s))
            sub_s.clear()
        return new_sub_state, next_state, next_state

    # 可能对动作有影响
    @staticmethod
    def action_transfer(a):
        for i, k in enumerate(a):
            # a[i] = k + np.random.uniform(-0.02, 0.02)
            a[i] = k + np.random.normal(0, 0.02)
        return a

    # 由于数据做了归一化处理，则对于reward来讲，如果使用差值的平方会使得奖励值很小，差值就表示着对于指标精度的衡量，此时如果设置精度过高将导致奖励值无法上升
    def reward(self, s, t):
        parser = get_config()
        all_args = parser.parse_args()
        sub_r = []

        thickness_ref = self.h_ref
        tension_ref = self.t_ref

        s = np.array(s)

        for j in range(5):
            if j != 4:
                # reward_ = (s[j][3] - thickness_ref[j]) ** 2 + (s[j][5] - tension_ref[j]) ** 2  # + ratio[j] * (s[4][5] - thickness_ref[4]) ** 2
                reward_ = abs(s[j][5] - thickness_ref[j]) ** 2 + abs(s[j][7] - tension_ref[j]) ** 2
            if j == 4:
                # reward_ = (s[j][3] - thickness_ref[j]) ** 2
                reward_ = abs(s[j][5] - thickness_ref[j]) ** 2

            if reward_ <= 0.1:
                r = - reward_

            elif 0.1 <= reward_ < 0.2:
                r = - reward_ - 0.05

            elif reward_ >= 0.2:
                r = - reward_ - 0.1
            # r = - reward_
            sub_r.append([r])

        # total_r = (0.9 * sub_r[0][0] + 0.9 * sub_r[1][0] + 0.9 * sub_r[2][0] + 0.9*sub_r[3][0] + sub_r[4][0]) / 5  # best=0.94
        total_r = (sub_r[0][0] + sub_r[1][0] + sub_r[2][0] + sub_r[3][0] + sub_r[4][0]) / 5
        share_r = [[total_r], [total_r], [total_r], [total_r], [total_r]]

        return share_r
        # return sub_r

    @staticmethod
    def data_read():
        df = pd.read_excel(
            'C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\envs\\pi_data.xlsx',
            sheet_name='Sheet1')
        data = df.values

        pi_data = {'h1': data[:, 11].tolist(),
                   'h2': data[:, 12].tolist(),
                   'h3': data[:, 13].tolist(),
                   'h4': data[:, 14].tolist(),
                   'h5': data[:, 15].tolist(),
                   'h0': data[:, 10].tolist(),
                   't01': data[:, 16].tolist(),
                   't12': data[:, 17].tolist(),
                   't23': data[:, 18].tolist(),
                   't34': data[:, 19].tolist(),
                   't45': data[:, 20].tolist(),
                   't56': data[:, 21].tolist()
                   }
        return pi_data
