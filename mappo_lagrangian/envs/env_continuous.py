import numpy as np
from gym import spaces
import pandas as pd
import math
from MAPPO_Lagrangian.mappo_lagrangian.envs.env_run import AgentEnvInteract


class ColdRolling(object):
    """对于连续动作环境的封装"""

    def __init__(self):
        self.interaction = AgentEnvInteract()
        self.agent_num = 5  # 设置智能体的个数
        self.obs_dim = 8  # 设置智能体的观测纬度
        self.action_dim = 2  # 设置智能体的动作纬度

        self.ep = 0
        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.share_policy_space = []

        share_obs_dim = 17+5
        share_po_dim = 17+5
        total_action_space = []
        for agent in range(self.agent_num):
            # physical action space
            u_action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)

            total_action_space.append(u_action_space)

            # total action space
            self.action_space.append(total_action_space[0])

            # observation space
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,),
                                                     dtype=np.float32))  # low=-np.inf, high=+np.inf

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.agent_num)]

        self.share_policy_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_po_dim,),
                                              dtype=np.float32) for _ in range(self.agent_num)]

    def step(self, actions, old_obs, t):
        """
        输入actions纬度：
        # actions shape = (5, 5, 2)
        # 5个线程的环境，里面有5个智能体，每个智能体的动作是一个one_hot的5维编码
        """
        n_t = t

        sub_agent_obs, share_agent_ht, share_agent_vt = self.interaction.state_trans(actions, old_obs, n_t)
        # self.save_data(ins, outs)

        if t <= 5:
            speed_usl = 1
        elif t >= 45:
            speed_usl = 0.1
        else:
            speed_usl = -0.0225*t + 1.1125

        if t >= 45:
            speed_lsl = 0
        elif t <= 5:
            speed_lsl = 0.9
        else:
            speed_lsl = -0.0225*t + 1.0125
        # if t <= 10:
        #     speed_usl = 1
        # elif t >= 90:
        #     speed_usl = 0.1
        # else:
        #     speed_usl = -0.01125 * t + 1.1125
        #
        # if t >= 90:
        #     speed_lsl = 0
        # elif t <= 10:
        #     speed_lsl = 0.9
        # else:
        #     speed_lsl = -0.01125 * t + 1.0125
        # sub_agent_reward, sub_agent_cost = self.interaction.reward_cost(sub_agent_obs, ref)
        sub_agent_reward = self.interaction.reward(sub_agent_obs, t)

        sub_agent_cost = []
        sub_agent_done = []
        sub_agent_info = []

        for i in range(self.agent_num):

            sub_agent_done.append(False)

            if speed_lsl <= share_agent_vt[i] <= speed_usl:
                sub_agent_cost.append([.0])
            else:
                v_error = math.exp(abs(share_agent_vt[i] - speed_lsl) + abs(share_agent_vt[i] - speed_usl))
                sub_agent_cost.append([v_error])

            sub_agent_info.append({})

        return np.stack(sub_agent_obs), np.stack(share_agent_ht), np.stack(share_agent_vt), np.stack(sub_agent_reward), np.stack(sub_agent_cost), np.stack(sub_agent_done), sub_agent_info

    def reset(self, model):
        sub_agent_obs, share_agent_ht, share_agent_vt = self.interaction.state_init(model)
        return np.stack(sub_agent_obs), np.stack(share_agent_ht), np.stack(share_agent_vt)

    def seed(self, param):
        pass

    def save_data(self, data1, data2):

        ep = self.ep
        # list转dataframe
        # data_1 = pd.DataFrame(data1.detach().numpy().tolist(), columns=[ep])
        # data_2 = pd.DataFrame(data2.detach().numpy().tolist(), columns=[ep])
        # self.ep += 1
        # 保存到本地excel
        # data_1.to_excel("C:\\Users\\user\\Desktop\\macpo\\MACPO\\macpo\\envs\\data_space.xlsx", index=False, sheet_name='Sheet1', startcol=self.ep)
        # data_2.to_excel("C:\\Users\\user\\Desktop\\macpo\\MACPO\\macpo\\envs\\data_space.xlsx", index=False, sheet_name='Sheet2', startcol=self.ep)
        data_1 = data1.detach().numpy().tolist()
        data_2 = data2.detach().numpy().tolist()
        f1 = open("C:\\Users\\user\\Desktop\\macpo_new1\\MACPO\\macpo\\envs\\data_space_in.txt", "a")
        f2 = open("C:\\Users\\user\\Desktop\\macpo_new1\\MACPO\\macpo\\envs\\data_space_out.txt", "a")
        for line in data_1:
            f1.write(str(line) + " ")
        f1.write('\n')
        f1.close()

        for line in data_2:
            f2.write(str(line) + " ")
        f2.write('\n')
        f2.close()
