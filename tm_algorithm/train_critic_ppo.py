import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class Env3D_RL:
    def __init__(self):
        self.mass = 1.0
        self.damping = 0.8
        self.target = np.array([10.0, 10.0, 5.0])
        self.start = np.array([0.0, 0.0, 0.0])
        self.obstacles = [
            [3.0, 3.0, 0.0, 1.2, 6.0],
            [6.0, 4.0, 0.0, 1.5, 5.0],
            [2.0, 7.0, 0.0, 1.0, 7.0],
            [7.0, 7.0, 0.0, 1.2, 4.0],
            [5.0, 8.0, 0.0, 1.0, 6.0],
        ]
        self.dt = 0.05
        self.max_steps = 150
        self.reset()

    def reset(self):
        self.pos = self.start.copy()
        self.vel = np.zeros(3)
        self.step_cnt = 0
        self.prev_dist = np.linalg.norm(self.pos - self.target)  # 记录初始距离
        return self.get_state()

    def get_state(self):
        rel_target = self.target - self.pos
        return np.concatenate([self.pos, self.vel, rel_target])

    def check_collision(self, pos, safe_radius=0.5):
        for ox, oy, oz, r, h in self.obstacles:
            dist_xy = math.sqrt((pos[0] - ox) ** 2 + (pos[1] - oy) ** 2)
            if dist_xy < (r + safe_radius) and (oz - safe_radius <= pos[2] <= oz + h + safe_radius):
                return True
        if pos[0] < -5 or pos[0] > 15 or pos[1] < -5 or pos[1] > 15 or pos[2] < -2 or pos[2] > 12:
            return True
        return False

    def step(self, force):
        force = np.clip(force, -15.0, 15.0)
        self.vel = np.clip(self.vel, -5.0, 5.0)

        acc = force / self.mass - self.damping * self.vel
        self.vel += acc * self.dt
        self.pos += self.vel * self.dt
        self.step_cnt += 1

        current_dist = np.linalg.norm(self.pos - self.target)

        # 【核心修改：距离引导奖励】
        # 如果靠近了终点，奖励正数；如果远离了终点，奖励负数
        reward = (self.prev_dist - current_dist) * 10.0
        self.prev_dist = current_dist

        done = False

        if self.check_collision(self.pos):
            reward = -50.0  # 撞墙重罚，直接死
            done = True
        elif current_dist < 1.0:
            reward = 200.0  # 到终点给巨大奖励
            done = True
        elif self.step_cnt >= self.max_steps:
            reward = -20.0  # 超时惩罚（比撞墙轻一点点）
            done = True

        return self.get_state(), reward, done


# ==================== PPO 网络 ====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim=9, action_dim=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.actor = nn.Linear(256, action_dim)
        self.actor_std = nn.Parameter(torch.zeros(1, action_dim) - 1.0)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.actor_std.exp().expand_as(self.actor(h)), self.critic(h)


# ==================== 训练主循环 ====================
def train():
    env = Env3D_RL()
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    print("开始带有距离引导的训练...")
    for episode in range(3000):
        state = env.reset()
        done = False
        states, actions, rewards, log_probs, values = [], [], [], [], []

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mean, std, value = model(state_t)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

            next_state, reward, done = env.step(action.detach().numpy()[0])

            states.append(state)
            actions.append(action.detach().numpy()[0])
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            state = next_state

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))
        old_log_probs_t = torch.FloatTensor(np.array(log_probs))
        returns_t = torch.FloatTensor(np.array(returns))

        for _ in range(5):
            mean, std, values_pred = model(states_t)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions_t).sum(-1)

            advantages = returns_t - values_pred.squeeze().detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values_pred.squeeze(), returns_t)

            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        if episode % 200 == 0:
            print(f"Episode {episode}: Total Reward = {sum(rewards):.1f}")

    torch.save(model.shared.state_dict(), "tm_shared.pth")
    torch.save(model.critic.state_dict(), "tm_critic.pth")
    print("✅ 训练完成！权重已保存。")


if __name__ == "__main__":
    train()