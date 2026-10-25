import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

state_dim = 3
action_dim = 1
hidden = 64
lr = 3e-4
gamma = 0.95
lambda_gae = 0.95
clip_eps = 0.2
epochs_per_update = 10
episode_len = 50
n_episodes = 2000

device = torch.device('cpu')

class EmotionEnv:
    def __init__(self):
        self.state = None

    def reset(self):
        v = np.random.uniform(-1, 1)
        a = np.random.uniform(0, 1)
        f = np.random.uniform(0, 1)
        self.state = np.array([v, a, f], dtype=np.float32)
        return self.state

    def s_target(self):
        v, a, f = self.state
        target = 1.0 + 0.6 * v + 0.3 * a - 0.7 * f
        return np.clip(target, 0.3, 2.0)

    def step(self, action_scale):
        s = float(action_scale[0])
        target = self.s_target()
        r = -(s - target) ** 2
        r -= 0.05 * s * s
        self.state[0] += np.random.normal(0, 0.05)
        self.state[0] = np.clip(self.state[0], -1, 1)
        self.state[1] += np.random.normal(0, 0.03)
        self.state[1] = np.clip(self.state[1], 0, 1)

        return self.state.copy(), r

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(hidden, action_dim)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mu_head(h)
        v = self.v_head(h)
        return mu, v

    def get_action(self, state):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            mu, v = self.forward(s)
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            action_clipped = torch.clamp(action, 0.3, 2.0)
        return action_clipped.cpu().numpy()[0], log_prob.item(), v.item()

    def evaluate(self, states, actions):
        mu, v = self.forward(states)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, v.squeeze(-1)

def compute_gae(rewards, values, dones, gamma=gamma, lam=lambda_gae):
    advantages = []
    gae = 0
    values = values + [0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return advantages, returns

if __name__ == "__main__":
    env = EmotionEnv()
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    reward_history = []

    for episode in range(n_episodes):
        state = env.reset()
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        for t in range(episode_len):
            action, log_prob, value = model.get_action(state)
            next_state, reward = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(0.0)

            state = next_state

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        old_log_probs_t = torch.tensor(log_probs, dtype=torch.float32, device=device)

        advantages, returns = compute_gae(rewards, values, dones)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        for _ in range(epochs_per_update):
            new_log_probs, entropy, values_pred = model.evaluate(states_t, actions_t)
            ratio = (new_log_probs - old_log_probs_t).exp()

            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred, returns_t)
            entropy_bonus = entropy.mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_reward = sum(rewards)
        reward_history.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode:4d} | total_reward = {total_reward:+.3f}")

        torch.save(model.state_dict(), "ppo_modulator.pt")
