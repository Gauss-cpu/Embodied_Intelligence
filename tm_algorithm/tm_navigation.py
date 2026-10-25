import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


# ==================== 1. 3D 物理环境 (原版一模一样) ====================
class Env3D:
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

    def check_collision(self, pos, safe_radius=0.5):
        for ox, oy, oz, r, h in self.obstacles:
            dist_xy = math.sqrt((pos[0] - ox) ** 2 + (pos[1] - oy) ** 2)
            if dist_xy < (r + safe_radius) and (oz - safe_radius <= pos[2] <= oz + h + safe_radius):
                return True
        return False

    def step(self, state, force):
        pos = state[:3]
        vel = state[3:]

        force = np.clip(force, -30.0, 30.0)

        acc = force / self.mass - self.damping * vel
        new_vel = vel + acc * self.dt
        new_pos = pos + new_vel * self.dt
        return np.concatenate([new_pos, new_vel])


# ==================== 2. 加载训练好的 PPO Critic ====================
class CriticNet(nn.Module):
    def __init__(self, state_dim=9):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        return self.critic(self.shared(x))


print("加载 PPO Critic 大脑...")
critic_brain = CriticNet()
# 这里读取你刚刚训练出来的权重文件
critic_brain.shared.load_state_dict(torch.load("tm_shared.pth", weights_only=True))
critic_brain.critic.load_state_dict(torch.load("tm_critic.pth", weights_only=True))
critic_brain.eval()


# ==================== 3. TM 算法 (融合 PPO 评估) ====================
class TM_Algorithm_3D:
    def __init__(self, env, num_samples=100, horizon_steps=12):
        self.env = env
        self.num_samples = num_samples
        self.horizon_steps = horizon_steps
        self.lambda_energy = 0.05

    def get_action(self, state, urgency):
        pos = state[:3]
        vel = state[3:]

        candidate_forces = []
        target_dir = (self.env.target - pos)
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-6)

        for _ in range(self.num_samples):
            noise = np.random.normal(0, 3.0, 3)
            force = target_dir * 10.0 + noise
            candidate_forces.append(force)

        safe_actions = []
        collision_trajs = []

        # 第一层 Safety
        for force in candidate_forces:
            temp_state = state.copy()
            # 【修改】：这里保存完整的 6 维状态，为了以后给 Critic 算速度
            trajectory = [temp_state.copy()]
            is_safe = True

            for _ in range(self.horizon_steps):
                temp_state = self.env.step(temp_state, force)
                trajectory.append(temp_state.copy())
                if self.env.check_collision(temp_state[:3]):
                    is_safe = False
                    break

            if is_safe:
                safe_actions.append((force, trajectory))
            else:
                collision_trajs.append(trajectory)

        if len(safe_actions) == 0:
            return -vel * 5.0, [], collision_trajs

        # 第二层 Optimal
        best_force = np.zeros(3)
        best_score = -float('inf')
        best_traj = []

        for force, trajectory in safe_actions:
            final_state = trajectory[-1]
            final_pos = final_state[:3]
            final_vel = final_state[3:]

            # 【核心替换】：组装 9 维特征喂给神经网络
            rel_target = self.env.target - final_pos
            critic_input = np.concatenate([final_pos, final_vel, rel_target])

            state_tensor = torch.FloatTensor(critic_input).unsqueeze(0)
            with torch.no_grad():
                # 神经网络直接给出长期价值打分
                q_val = critic_brain(state_tensor).item()

            energy_penalty = self.lambda_energy * np.sum(force ** 2)

            # 综合得分 = 大脑打分 - 能耗
            score = q_val - energy_penalty

            if score > best_score:
                best_score = score
                best_force = force
                best_traj = trajectory

        return best_force, best_traj, collision_trajs


# ==================== 4. 渲染 ====================
def main():
    env = Env3D()
    tm = TM_Algorithm_3D(env)

    state = np.concatenate([env.start, [0.0, 0.0, 0.0]])
    history_pos = [state[:3].copy()]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*env.target, c='g', marker='*', s=400, label='Target')

    for ox, oy, oz, r, h in env.obstacles:
        z_vals = np.linspace(oz, oz + h, 10)
        theta = np.linspace(0, 2 * np.pi, 15)
        for zv in z_vals:
            ax.plot(ox + r * np.cos(theta), oy + r * np.sin(theta), [zv] * 15, color='red', alpha=0.1)

    path_line, = ax.plot([], [], [], 'b-', lw=3, label="Robot Path")
    robot_dot, = ax.plot([], [], [], 'bo', ms=12)
    best_pred, = ax.plot([], [], [], 'y--', lw=2)
    coll_lines = [ax.plot([], [], [], 'r--', alpha=0.1, lw=1)[0] for _ in range(10)]

    urgency_text = ax.text2D(0.05, 0.95, "Urgency: 0.0", transform=ax.transAxes, fontsize=14, color='purple')
    score_text = ax.text2D(0.05, 0.90, "Force: 0.0", transform=ax.transAxes, fontsize=14, color='black')

    ax.legend(loc='lower right')
    max_steps = 150
    step_cnt = 0

    def update(frame):
        nonlocal state, step_cnt
        dist_to_target = np.linalg.norm(state[:3] - env.target)

        if step_cnt >= max_steps or dist_to_target < 0.8:
            ax.set_title("Target Reached", fontsize=18, color='green')
            return path_line, robot_dot

        urgency = 0.1
        urgency_text.set_text(f"Urgency: {urgency:.2f}")

        best_force, best_traj, coll_trajs = tm.get_action(state, urgency)

        score_text.set_text(f"Force: {np.linalg.norm(best_force):.2f}")

        if len(best_traj) > 0:
            bx = [p[0] for p in best_traj]
            by = [p[1] for p in best_traj]
            bz = [p[2] for p in best_traj]
            best_pred.set_data(bx, by)
            best_pred.set_3d_properties(bz)

        for i, line in enumerate(coll_lines):
            if i < len(coll_trajs):
                cx = [p[0] for p in coll_trajs[i]]
                cy = [p[1] for p in coll_trajs[i]]
                cz = [p[2] for p in coll_trajs[i]]
                line.set_data(cx, cy)
                line.set_3d_properties(cz)
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        state = env.step(state, best_force)
        history_pos.append(state[:3].copy())
        step_cnt += 1

        hx = [p[0] for p in history_pos]
        hy = [p[1] for p in history_pos]
        hz = [p[2] for p in history_pos]

        path_line.set_data(hx, hy)
        path_line.set_3d_properties(hz)
        robot_dot.set_data([hx[-1]], [hy[-1]])
        robot_dot.set_3d_properties([hz[-1]])

        ax.set_xlim(state[0] - 6, state[0] + 6)
        ax.set_ylim(state[1] - 6, state[1] + 6)
        ax.set_zlim(max(0, state[2] - 4), state[2] + 6)

        return path_line, robot_dot, best_pred, *coll_lines, urgency_text, score_text

    ani = animation.FuncAnimation(fig, update, frames=range(max_steps), interval=50, blit=False)
    plt.show()


if __name__ == "__main__":
    main()