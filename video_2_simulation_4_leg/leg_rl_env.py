import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from pathlib import Path

# 引入物理引擎参数
g = 9.81
L1, L2 = 0.45, 0.45
m1, m2 = 10.0, 4.0


class LegRLEnv(gym.Env):
    def __init__(self, dataset_path, model_path, max_steps=100):
        super().__init__()

        # 1. 加载真实数据 (作为参考目标)
        data = np.load(dataset_path, allow_pickle=True)
        self.ref_X = data["X"]
        self.ref_Y = data["Y"]
        self.max_steps = min(max_steps, len(self.ref_X))

        # 2. 动作空间：4 维连续力矩 [Hip_L, Knee_L, Hip_R, Knee_R]
        self.action_space = spaces.Box(low=-200.0, high=200.0, shape=(4,), dtype=np.float32)

        # 3. 观察空间：20 维特征
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.current_step = 0
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # 随机选一段数据作为起始点
        start_idx = np.random.randint(0, len(self.ref_X) - self.max_steps)
        self.ref_X_seq = self.ref_X[start_idx: start_idx + self.max_steps]

        self.state = self.ref_X_seq[0].copy()
        return self.state, {}

    def step(self, action):
        # 这里的 action 就是 DNN 输出的力矩 (未标准化)

        # 1. 用正向动力学计算下一帧状态
        # (为了训练速度，这里用简化的物理更新，真实项目中需要接入完整的 RK4)
        qdd = action / 15.0  # 简化的加速度假想
        new_state = self.state.copy()
        new_state[16:20] = qdd  # 更新加速度特征
        new_state[8:12] += new_state[12:16] * 0.01  # 更新角度

        self.state = new_state
        self.current_step += 1

        # 2. 计算奖励 (Reward)
        # 理想状态 vs 当前状态 (MSE)
        target_state = self.ref_X_seq[self.current_step]
        tracking_error = np.mean((new_state - target_state) ** 2)

        # 高度检查 (模拟摔倒)
        hips_height = new_state[0]
        is_fallen = hips_height < 0.6  # 假设正常高度 0.9 左右

        # 能量惩罚
        energy_penalty = np.mean(action ** 2) * 0.0001

        reward = -tracking_error - energy_penalty
        if is_fallen:
            reward -= 100.0  # 摔倒重罚

        terminated = is_fallen or (self.current_step >= self.max_steps - 1)
        truncated = False

        return self.state, reward, terminated, truncated, {}