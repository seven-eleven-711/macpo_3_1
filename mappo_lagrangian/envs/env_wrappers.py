import numpy as np


# single env
# 生成多个并行环境
class DummyVecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]        # 每一个环境都是类ContinuousActionEnv
        env = self.envs[0]
        self.num_envs = len(env_fns)        # 线程数
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.share_policy_space = env.share_policy_space
        self.action_space = env.action_space
        self.actions = None     # 初始化一个动作空间
        self.old_env = None     # 保存上一时刻状态
        # self.act_ss = None

    def step(self, actions, old_env, step):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.actions = actions
        self.old_env = old_env
        # self.step_async(actions)
        step_now = step
        return self.step_wait(step_now)

    # def step_async(self, actions):

    def step_wait(self, t):
        results = [env.step(a, s, t) for (a, s, env) in zip(self.actions, self.old_env, self.envs)]
        obs, share_obs_ht, share_obs_vt, rews, cost, dones, infos = map(np.array, zip(*results))      # 对结果列表操作，返回数组

        self.actions = None
        return obs, share_obs_ht, share_obs_vt, rews, cost, dones, infos

    def reset(self, model):
        sub_obs = []
        share_ht = []
        share_vt = []
        # 多线分别初始化
        for env in self.envs:
            sub, share_v, share_c = env.reset(model)
            sub_obs.append(sub)
            share_ht.append(share_v)
            share_vt.append(share_c)
        # sub_obs, share_obs = [env.reset() for env in self.envs]
        return np.array(sub_obs), np.array(share_ht), np.array(share_vt)
        # return sub_obs, share_obs
