import os
import time
from itertools import chain

import pandas as pd
import wandb
import numpy as np
from functools import reduce
import torch
from matplotlib import pyplot as plt
from proplot import rc
from MAPPO_Lagrangian.mappo_lagrangian.runner.separated.base_runner_mappo_lagr import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MujocoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(MujocoRunner, self).__init__(config)
        # self.algorithm_name = config.all_args.algorithm_name
        self.average_reward = []
        self.average_cost = []

    def run(self):
        # 环境重启
        sub_obs, _, _ = self.warmup()
        old_obs = sub_obs
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        train_episode_costs = [0 for _ in range(self.n_rollout_threads)]
        train_agent_costs = [0 for _ in range(self.num_agents)]

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            done_episodes_rewards = []
            done_episodes_costs = []

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, \
                    rnn_states_cost = self.collect(step)

                # Obser reward cost and next obs
                obs, share_obs_ht, share_obs_vt, rewards, costs, dones, infos = self.envs.step(actions, old_obs, step)
                old_obs = obs
                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                cost_env = np.mean(costs, axis=1).flatten()
                cost_agent = np.mean(costs, axis=0).flatten()
                train_episode_rewards += reward_env
                train_episode_costs += cost_env
                train_agent_costs += cost_agent

                for t in range(self.n_rollout_threads):
                    done_episodes_rewards.append(train_episode_rewards[t])
                    train_episode_rewards[t] = 0
                    done_episodes_costs.append(train_episode_costs[t])
                    train_episode_costs[t] = 0

                data = obs, share_obs_ht, share_obs_vt, rewards, costs, dones, infos, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic, cost_preds, rnn_states_cost  # fixme: it's important!!!

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                # if len(done_episodes_rewards) > 0:
                aver_episode_rewards = np.mean(done_episodes_rewards)
                aver_episode_costs = np.mean(done_episodes_costs)
                aver_episode_gent_costs = [c/50 for c in train_agent_costs]
                for t in range(self.num_agents):
                    train_agent_costs[t] = 0
                # self.return_aver_cost(aver_episode_costs)
                self.return_aver_cost(aver_episode_gent_costs)
                print("average rewards: {}, average costs: {}".format(aver_episode_rewards, aver_episode_costs))
                # 记录奖励值和代价值
                self.average_reward.append(aver_episode_rewards)
                self.average_cost.append(aver_episode_costs)
                self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards}, episode)
                self.writter.add_scalars("train_episode_costs", {"aver_costs": aver_episode_costs}, episode)
                if episode == 1999:
                    df_1 = pd.DataFrame(self.average_reward, columns=['average_reward'])
                    df_2 = pd.DataFrame(self.average_cost, columns=['average_cost'])
                    df_1.to_excel('C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\r.xlsx', index=False, sheet_name="Sheet1")
                    df_2.to_excel('C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\c.xlsx', index=False, sheet_name="Sheet1")
                # if episode == 3999:
                #     df_1 = pd.DataFrame(self.average_reward, columns=['average_reward'])
                #     df_2 = pd.DataFrame(self.average_cost, columns=['average_cost'])
                #     df_1.to_excel('C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\r.xlsx', index=False, sheet_name="Sheet1")
                #     df_2.to_excel('C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\c.xlsx', index=False, sheet_name="Sheet1")

            # eval
            if episode % 200 == 0:
                self.control(int(episode / 200))

    def return_aver_cost(self, aver_episode_costs):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_insert(aver_episode_costs[agent_id])
        # if not self.use_centralized_V:
        #     for agent_id in range(self.num_agents):
        #         self.buffer[agent_id].return_aver_insert(aver_episode_costs[agent_id])
        # else:
        #     for agent_id in range(self.num_agents):
        #         self.buffer[agent_id].return_aver_insert(aver_episode_costs)

    def warmup(self):
        # reset env
        obs, share_ht, share_vt = self.envs.reset(False)
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            # print(share_obs[:, agent_id])
            # self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            # self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].share_ht[0] = share_ht.copy()
            self.buffer[agent_id].share_vt[0] = share_vt.copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
        return obs, share_ht, share_vt

    @torch.no_grad()
    def collect(self, step):
        # values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, \
        # rnn_states_cost = self.collect(step)

        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        cost_preds_collector = []
        rnn_states_cost_collector = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic, cost_pred, rnn_state_cost \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_ht[step],
                                                            self.buffer[agent_id].share_vt[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            rnn_states_cost=self.buffer[agent_id].rnn_states_cost[step]
                                                            )
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
            cost_preds_collector.append(_t2n(cost_pred))
            rnn_states_cost_collector.append(_t2n(rnn_state_cost))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)
        cost_preds = np.array(cost_preds_collector).transpose(1, 0, 2)
        rnn_states_cost = np.array(rnn_states_cost_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost

    def insert(self, data):
        obs, share_obs_ht, share_obs_vt, rewards, costs, dones, infos, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost = data  # fixme:!!!
        # print("insert--rewards", rewards)
        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        rnn_states_cost[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_cost.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs_ht,
                                         share_obs_vt,
                                         obs[:, agent_id],
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], None,
                                         active_masks[:, agent_id], None,
                                         costs=costs[:, agent_id],
                                         cost_preds=cost_preds[:, agent_id],
                                         rnn_states_cost=rnn_states_cost[:, agent_id])

    def log_train(self, train_infos, total_num_steps):
        # 打印1#智能体的奖励值
        print("average_step_rewards is {}.".format(np.mean(self.buffer[0].rewards)))
        train_infos[0][0]["average_step_rewards"] = 0
        for agent_id in range(self.num_agents):
            train_infos[0][agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[0][agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    @torch.no_grad()
    def control(self, e_num):

        tension = []
        thickness = []
        rolling_speed = []
        rolling_aspeed = []
        roll_sew = []
        for i in range(6):
            tension.append([])
            thickness.append([])
            rolling_speed.append([])
            rolling_aspeed.append([])
            roll_sew.append([])

        eval_obs, eval_ht, eval_vt = self.eval_envs.reset(True)
        for i in range(5):
            rolling_speed[i].append(eval_vt[0][i])
            # roll_sew[i].append(eval_vt[0][i+5])

        for j in range(5):
            thickness[j].append(eval_obs[0][j][5])
            tension[j].append(eval_obs[0][j][7])
        # thickness[4].append(eval_obs[0][4][5])
        thickness[5].append(eval_ht[0][10])     # h0

        old_obs = eval_obs

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        step = 0

        while step < self.episode_length:
            control_actions_collector = []
            # eval_rnn_states_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                control_actions, temp_rnn_state = self.trainer[agent_id].policy.act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True)

                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                control_actions_collector.append(_t2n(control_actions))

                # rolling_aspeed[agent_id].append(control_actions[0][0])
                # roll_sew[agent_id].append(control_actions[0][1])

            control_actions_all = np.array(control_actions_collector).transpose(1, 0, 2)

            # Obser reward and next obs
            eval_obs, eval_ht, eval_vt, eval_rewards, eval_cost, eval_dones, eval_infos = self.eval_envs.step(
                control_actions_all, old_obs,
                step)
            step += 1
            old_obs = eval_obs
            # print(eval_obs)
            for i in range(5):
                rolling_speed[i].append(eval_vt[0][i])
                # roll_sew[i].append(eval_vt[0][i + 5])

            for j in range(5):
                thickness[j].append(eval_obs[0][j][5])
                tension[j].append(eval_obs[0][j][7])
            # thickness[4].append(eval_obs[0][4][5])
            thickness[5].append(eval_ht[0][10])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

        eval_env_infos = {'h1': thickness[0],
                          'h2': thickness[1],
                          'h3': thickness[2],
                          'h4': thickness[3],
                          'h5': thickness[4],
                          'h0': thickness[5],
                          't12': tension[0],
                          't23': tension[1],
                          't34': tension[2],
                          't45': tension[3],
                          'v1': rolling_speed[0],
                          'v2': rolling_speed[1],
                          'v3': rolling_speed[2],
                          'v4': rolling_speed[3],
                          'v5': rolling_speed[4],
                          # 辊速动作
                          # 辊缝动作设定
                          # 's1': roll_sew[0],
                          # 's2': roll_sew[1],
                          # 's3': roll_sew[2],
                          # 's4': roll_sew[3],
                          # 's5': roll_sew[4]
                          }

        self.plot_results(eval_env_infos, step, e_num)

    @torch.no_grad()
    def data_read(self):
        df = pd.read_excel(
            'C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\envs\\pi_data.xlsx',
            sheet_name='Sheet1', nrows=51)
        data = df.values

        pi_data = {'h1': data[:, 0].tolist(),
                   'h2': data[:, 1].tolist(),
                   'h3': data[:, 2].tolist(),
                   'h4': data[:, 3].tolist(),
                   'h5': data[:, 4].tolist(),
                   # 'h0': data[:, 10].tolist(),
                   # 't01': data[:, 16].tolist(),
                   't12': data[:, 5].tolist(),
                   't23': data[:, 6].tolist(),
                   't34': data[:, 7].tolist(),
                   't45': data[:, 8].tolist(),
                   # 't56': data[:, 21].tolist(),
                   # # 辊速动作
                   # 'v1': data[:, 0].tolist(),
                   # 'v2': data[:, 1].tolist(),
                   # 'v3': data[:, 2].tolist(),
                   # 'v4': data[:, 3].tolist(),
                   # 'v5': data[:, 4].tolist(),
                   # # 辊缝动作设定
                   # 's1': data[:, 5].tolist(),
                   # 's2': data[:, 6].tolist(),
                   # 's3': data[:, 7].tolist(),
                   # 's4': data[:, 8].tolist(),
                   # 's5': data[:, 9].tolist()
                   }
        return pi_data

    @torch.no_grad()
    def plot_results(self, env_infos, step, e_num):

        speed_usl = []
        speed_lsl = []
        for t in range(51):
            if t <= 5:
                speed_usl.append(1.0)
            elif t >= 45:
                speed_usl.append(0.1)
            else:
                speed_usl.append(-0.0225*t + 1.1125)

            if t >= 45:
                speed_lsl.append(0.0)
            elif t <= 5:
                speed_lsl.append(0.9)
            else:
                speed_lsl.append(-0.0225*t + 1.0125)

        normalization_params = []
        pi_data = self.data_read()
        # self.data_fit(eval_env_infos)
        eval_env_infos = self.data_fit(env_infos)
        # self.log_env(eval_env_infos, step)

        ref = {'h1': 1.505, 'h2': 0.934, 'h3': 0.607, 'h4': 0.37, 'h5': 0.252,
               't12': 17.832, 't23': 11.762, 't34': 7.928, 't45': 4.94}

        plt.figure()
        for key in ['h0', 'v1', 'v2', 'v3', 'v4', 'v5']:        # , 's1', 's2', 's3', 's4', 's5'

            # values_pi = pi_data.get(key)
            values_rl = eval_env_infos.get(key)

            plt.xlabel('time', fontsize=20)  # 显示x轴标签
            plt.ylabel(key, fontsize=20)  # 显示y轴标签
            # plt.plot(values_pi, 'black', label='pi')
            plt.tick_params(labelsize=11)
            plt.plot(values_rl, 'red', label='mappo_l')
            plt.legend(loc=1)
            plt.grid()  # 显示网格

            if key in ['v1', 'v2', 'v3', 'v4', 'v5']:
                plt.plot(speed_usl, color='green', linestyle='--')
                plt.plot(speed_lsl, color='green', linestyle='--')

            figure_path = "C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\test_picture"  # 创建一个文件夹保存结果
            figure_save_path = os.path.join(figure_path, str(e_num))
            if not os.path.exists(figure_save_path):
                os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
            plt.savefig(os.path.join(figure_save_path, key+'.png'))  # 分别命名图片
            # plt.show()
            plt.clf()
        plt.close()

        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(nrows=5, ncols=6)
        for key in ['h1', 'h2', 'h3', 'h4', 'h5', 't12', 't23', 't34', 't45']:
            values_pi = pi_data.get(key)
            values_rl = eval_env_infos.get(key)
            # 控制效果对比图
            control_chart = fig.add_subplot(gs[:, :5])
            plt.tick_params(labelsize=11)
            control_chart.set_xlabel('time/s', fontsize=20)
            if key in ['h1', 'h2', 'h3', 'h4', 'h5']:
                control_chart.set_ylabel(key + '/mm', fontsize=20)
            else:
                control_chart.set_ylabel(key + '/T', fontsize=20)
            control_chart.plot(np.linspace(0.0, 5.0, 51), values_pi, 'blue', label='pi', linewidth=2, linestyle='--', alpha=0.5)
            control_chart.plot(np.linspace(0.0, 5.0, 51), values_rl, 'red', label='macpo', linewidth=2, alpha=0.7)
            plt.xticks(np.linspace(0.0, 5.0, 6))
            control_chart.axhline(ref[key])
            control_chart.legend(loc=2)
            control_chart.grid()  # 显示网格

            # 结果指标对比
            # 计算最大误差百分比
            max_percent_error_pi = max([abs(value - ref[key]) * 100 for value in values_pi])
            max_percent_error_rl = max([abs(value - ref[key]) * 100 for value in values_rl[1:]])
            percent_bias = [max_percent_error_pi, max_percent_error_rl]
            index1 = fig.add_subplot(gs[:2, 5])
            # 子图：最大偏差百分比
            index1.set_title('Max_APE')
            index1.set_ylabel('%')
            index1.bar(x=['pi', 'mappo_l'], height=percent_bias, color=['red', 'blue'], alpha=0.4)

            # 计算标准差
            total_pi_deviation = []
            total_rl_deviation = []
            for i in range(len(values_pi)):
                total_pi_deviation = [abs(value - ref[key]) for value in values_pi]
                total_rl_deviation = [abs(value - ref[key]) for value in values_rl[1:]]
            pi_std = sum(total_pi_deviation) / len(values_pi)
            rl_std = sum(total_rl_deviation) / len(values_pi)
            std = [pi_std, rl_std]
            index2 = fig.add_subplot(gs[3:5, 5])
            # 子图：标准差
            index2.set_title('STD')
            if key in ['h1', 'h2', 'h3', 'h4', 'h5']:
                index2.set_ylabel('mm')
            else:
                control_chart.set_ylabel('T')
            index2.bar(x=['pi', 'mappo_l'], height=std, color=['red', 'blue'], alpha=0.4)
            plt.tight_layout()
            figure_path = "C:\\Users\\admin\\Desktop\\macpo_3_1\\MAPPO_Lagrangian\\mappo_lagrangian\\scripts\\test_picture"  # 创建一个文件夹保存结果
            figure_save_path = os.path.join(figure_path, str(e_num))
            if not os.path.exists(figure_save_path):
                os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
            plt.savefig(os.path.join(figure_save_path, key+'.png'))  # 分别命名图片
            # plt.show()
            plt.clf()
            df = pd.DataFrame(env_infos)
            df.to_excel(figure_save_path + ".xlsx", index=False)
        plt.close()
        return

    @staticmethod
    def data_fit(eval_env_infos):
        edge = {'h0': [2.15549, 2.21369], 'h1': [1.49235, 1.52903], 'h2': [0.919962755, 0.945309535], 'h3': [0.600405128, 0.615743896],
                'h4': [0.36326, 0.375513], 'h5': [0.250649, 0.254174],
                't12': [17.6221, 18.9119], 't23': [11.724, 12.5532], 't34': [7.86168, 8.58367],
                't45': [4.82258, 5.33566],
                'v1': [98.5674, 197.889], 'v2': [155.649, 316.051], 'v3': [239.695, 484.981], 'v4': [398.417, 810.617],
                'v5': [602.299, 1200]}

        for key in ['h1', 'h2', 'h3', 'h4', 'h5', 't12', 't23', 't34', 't45']:
            normal_edge = edge[key]
            denormal_data = [a * (normal_edge[1] - normal_edge[0]) + normal_edge[0] for a in eval_env_infos[key]]
            eval_env_infos[key] = denormal_data

        return eval_env_infos
