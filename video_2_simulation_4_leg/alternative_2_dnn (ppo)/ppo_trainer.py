import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import math
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ==================== 1. 物理环境 ====================
g = 9.81
L1, L2 = 0.45, 0.45
m1, m2 = 10.0, 4.0


def forward_kinematics(q):
    th1, th2 = q
    abs_th1 = th1
    abs_th2 = th1 - th2
    p_hip = np.array([0.0, 0.0])
    p_knee = p_hip + L1 * np.array([np.sin(abs_th1), -np.cos(abs_th1)])
    p_foot = p_knee + L2 * np.array([np.sin(abs_th2), -np.cos(abs_th2)])
    return p_knee, p_foot


def jacobian(f, q, eps=1e-5):
    q = np.asarray(q, dtype=float);
    n = len(q)
    f0 = np.asarray(f(q), dtype=float);
    J = np.zeros((len(f0), n))
    for i in range(n):
        qp = q.copy();
        qp[i] += eps
        qm = q.copy();
        qm[i] -= eps
        J[:, i] = (f(qp) - f(qm)) / (2 * eps)
    return J


def mass_matrix(q):
    p1_fn = lambda qq: forward_kinematics(qq)[0]
    p2_fn = lambda qq: forward_kinematics(qq)[1]
    Jv1 = jacobian(p1_fn, q);
    Jv2 = jacobian(p2_fn, q)
    return m1 * Jv1.T @ Jv1 + m2 * Jv2.T @ Jv2


def gravity_term(q):
    p1_fn = lambda qq: forward_kinematics(qq)[0]
    p2_fn = lambda qq: forward_kinematics(qq)[1]
    Jv1 = jacobian(p1_fn, q);
    Jv2 = jacobian(p2_fn, q)
    G = np.array([0.0, -g])
    return -(m1 * Jv1.T @ G + m2 * Jv2.T @ G)


class LegRLEnv:
    def __init__(self, dataset_path, scaler_path, max_steps=200):
        data = np.load(dataset_path, allow_pickle=True)
        self.ref_X = data["X"]
        self.max_steps = max_steps

        # 【修复 1】加载归一化器
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
        self.scaler_X = scalers["scaler_X"]

    def reset(self):
        start_idx = np.random.randint(0, len(self.ref_X) - self.max_steps)
        self.seq_X = self.ref_X[start_idx: start_idx + self.max_steps]

        self.current_step = 0
        self.q = np.array([self.seq_X[0, 8], self.seq_X[0, 9]])
        self.qd = np.array([0.0, 0.0])

        # 返回归一化后的状态
        raw_state = self.seq_X[0].copy()
        self.state_obs = self.scaler_X.transform(raw_state.reshape(1, -1))[0]
        return self.state_obs

    def step(self, action):
        tau = np.clip(action[:2] * 20.0, -100.0, 100.0)
        dt = 0.01
        for _ in range(3):
            M = mass_matrix(self.q)
            G = gravity_term(self.q)
            try:
                qdd = np.linalg.solve(M, tau - G - self.qd * 5.0)
            except:
                qdd = np.zeros(2)
            self.qd = np.clip(self.qd + qdd * dt, -20.0, 20.0)
            self.q = np.clip(self.q + self.qd * dt, -math.pi * 2, math.pi * 2)

        self.current_step += 1
        target = self.seq_X[self.current_step].copy()

        raw_state = target.copy()
        raw_state[8], raw_state[9] = self.q[0], self.q[1]
        raw_state[12], raw_state[13] = self.qd[0], self.qd[1]

        # 归一化新状态
        self.state_obs = self.scaler_X.transform(raw_state.reshape(1, -1))[0]

        target_q = np.array([target[8], target[9]])
        error = np.sum((self.q - target_q) ** 2)

        _, p_foot = forward_kinematics(self.q)
        is_fallen = p_foot[1] < -1.5 or np.abs(self.q[0] - target_q[0]) > 1.5

        if not hasattr(self, 'last_action'):
            self.last_action = np.zeros_like(action)

            # 惩罚力矩突变
        action_smoothness_penalty = np.sum((action - self.last_action) ** 2) * 0.1
        self.last_action = action

        # 【修复 2】使用 exp 限制 reward 范围，防止积分爆炸
        reward = np.exp(-error * 10.0) - action_smoothness_penalty # 误差越小，reward 越接近 1
        if is_fallen:
            reward = -10.0  # 摔跤固定扣 10 分，绝对不给天文数字
            done = True
        else:
            done = self.current_step >= self.max_steps - 1
        torque_penalty = np.sum(action ** 2) * 0.01
        reward = np.exp(-error * 10.0) - torque_penalty

        return self.state_obs, reward, done


# ==================== 2. PPO 网络 ====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim=20, action_dim=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) - 2.0)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared(state)
        return self.actor_mean(x), self.actor_log_std.exp().expand_as(self.actor_mean(x)), self.critic(x)


# ==================== 3. 训练配置 ====================
DATA_PATH = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/features/leg_dataset.npz"
SCALER_PATH = Path.home() / "Desktop/Physical_Intelligence/video_2_simulation_4_leg/scalers.pkl"
MODEL_PATH = Path.home() / "Desktop/Physical_Intelligence/video_2_simulation_4_leg/dnn_torque_model.pt"

env = LegRLEnv(str(DATA_PATH), str(SCALER_PATH), max_steps=150)
policy = ActorCritic(20, 4).to(DEVICE)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)

GAMMA = 0.99
CLIP_EPS = 0.2
K_EPOCHS = 5
STEPS_PER_EPOCH = 1024

# 加载 DNN 权重
try:
    dnn_weights = torch.load(MODEL_PATH, map_location=DEVICE)
    policy.shared[0].weight.data = dnn_weights['net.0.weight'].data
    policy.shared[0].bias.data = dnn_weights['net.0.bias'].data
    policy.shared[2].weight.data = dnn_weights['net.3.weight'].data
    policy.shared[2].bias.data = dnn_weights['net.3.bias'].data
    print("✅ 成功加载 DNN 预训练权重！")
except Exception as e:
    print(f"⚠️ 未找到预训练权重：{e}")


# ==================== 4. PPO 核心 ====================
def train():
    reward_history = []
    print("开始 PPO 强化学习训练...")

    for iteration in range(150):
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        state = env.reset()
        ep_reward = 0

        for step in range(STEPS_PER_EPOCH):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mean, std, value = policy(state_t)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            action_np = action.cpu().numpy()[0]
            next_state, reward, done = env.step(action_np)

            states.append(state);
            actions.append(action_np)
            log_probs.append(log_prob.item())
            rewards.append(reward);
            dones.append(done);
            values.append(value.item())

            ep_reward += reward
            state = next_state
            if done:
                state = env.reset()
                break

        # 计算 Returns
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + GAMMA * R * (1 - d)
            returns.insert(0, R)

        states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions_t = torch.FloatTensor(np.array(actions)).to(DEVICE)
        old_log_probs_t = torch.FloatTensor(np.array(log_probs)).to(DEVICE)
        returns_t = torch.FloatTensor(np.array(returns)).to(DEVICE)

        # PPO 更新
        for _ in range(K_EPOCHS):
            mean, std, values_pred = policy(states_t)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)

            advantages = returns_t - values_pred.squeeze().detach()
            # 【修复 3】优势函数必须标准化
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            # 【修复 4】给 Critic Loss 加上 Clamp，防止梯度爆炸
            critic_loss = nn.MSELoss()(values_pred.squeeze(), returns_t)
            critic_loss = torch.clamp(critic_loss, max=10.0)

            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        reward_history.append(ep_reward)
        if iteration % 5 == 0:
            print(f"Iter {iteration:3d} | Episode Reward: {ep_reward:.1f} | Steps: {len(states)}")

    torch.save(policy.state_dict(), "ppo_leg_policy.pt")
    print("✅ 完美！PPO 策略已保存为 ppo_leg_policy.pt")
    plt.plot(reward_history)
    plt.title("PPO 训练奖励曲线")
    plt.xlabel("Iteration")
    plt.ylabel("Episode Reward")
    plt.show()


if __name__ == "__main__":
    train()