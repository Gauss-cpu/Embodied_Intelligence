import json
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
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ==================== 1. 加载骨骼数据 ====================
ROOT = Path.home() / "Desktop/Physical_Intelligence"
FULL_BODY_DIR = ROOT / "dataset_cmu/full_body_3d/008"
full_body_file = list(FULL_BODY_DIR.rglob("*.json"))[0]

with open(full_body_file) as f:
    full_body_data = json.load(f)

traj_3d = full_body_data["trajectory"]
frames_count = len(traj_3d)
local_joints_data = []

SCALE = 0.015
for f in traj_3d:
    hips = np.array([f["Hips"][0], f["Hips"][1], f["Hips"][2]])
    frame_joints = {}
    for joint in ["Hips", "Spine", "Neck", "LeftArm", "LeftForeArm", "LeftHand",
                  "RightArm", "RightForeArm", "RightHand",
                  "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]:
        p = f[joint]
        rel_x = (p[0] - hips[0]) * SCALE
        rel_y = (p[2] - hips[2]) * SCALE
        rel_z = (p[1] - hips[1]) * SCALE
        frame_joints[joint] = np.array([rel_x, rel_y, rel_z])
    local_joints_data.append(frame_joints)


# ==================== 2. 定义与加载网络 ====================
class UrgencyNet(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


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


print("加载 Urgency 与 Critic 大脑...")
text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
urgency_brain = UrgencyNet().to(DEVICE)
critic_brain = CriticNet().to(DEVICE)

try:
    # 【注意路径】：加载上级目录 urgency_policy 里的模型
    urgency_path = ROOT / "urgency_policy/urgency_model.pth" if (
                ROOT / "urgency_policy/urgency_model.pth").exists() else Path("../urgency_policy/urgency_model.pth")
    urgency_brain.load_state_dict(torch.load(urgency_path, map_location=DEVICE, weights_only=True))

    # 加载当前目录的 Critic 模型
    critic_brain.shared.load_state_dict(torch.load("tm_shared.pth", map_location=DEVICE, weights_only=True))
    critic_brain.critic.load_state_dict(torch.load("tm_critic.pth", map_location=DEVICE, weights_only=True))
    print("✅ 所有模型加载成功！")
except Exception as e:
    print(f"⚠️ 模型加载失败: {e}")
    exit()

urgency_brain.eval()
critic_brain.eval()


# ==================== 3. 物理环境 ====================
class Env3D:
    def __init__(self):
        self.mass = 1.0
        self.damping = 0.8
        self.max_speed = 1.5
        self.target = np.array([10.0, 10.0, 0.0])
        self.start = np.array([0.0, 0.0, 1.5])

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
        acc[2] = 0
        new_vel = vel + acc * self.dt
        new_vel[2] = 0

        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed:
            new_vel = (new_vel / speed) * self.max_speed

        new_pos = pos + new_vel * self.dt
        new_pos[2] = 1.5
        return np.concatenate([new_pos, new_vel])


# ==================== 4. TM 算法 ====================
class TM_Algorithm_3D:
    def __init__(self, env, num_samples=200, horizon_steps=20):
        self.env = env
        self.num_samples = num_samples
        self.horizon_steps = horizon_steps

    def get_action(self, state, urgency):
        pos = state[:3]
        vel = state[3:]
        target_dir = (self.env.target - pos)
        target_dir[2] = 0
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-6)

        candidate_forces = []
        for _ in range(self.num_samples):
            noise = np.random.normal(0, 4.0, 3)
            noise[2] = 0
            force = target_dir * (15.0 + 10.0 * urgency) + noise
            candidate_forces.append(force)

        safe_actions = []
        for force in candidate_forces:
            temp_state = state.copy()
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

        if len(safe_actions) == 0:
            return -vel * 5.0, []

        best_force = np.zeros(3)
        best_score = -float('inf')
        best_traj = []
        current_dist = np.linalg.norm(pos - self.env.target)

        for force, trajectory in safe_actions:
            final_state = trajectory[-1]
            final_pos = final_state[:3]
            final_vel = final_state[3:]

            rel_target = self.env.target - final_pos
            critic_input = np.concatenate([final_pos, final_vel, rel_target])

            state_tensor = torch.FloatTensor(critic_input).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_val = critic_brain(state_tensor).item()

            final_dist = np.linalg.norm(final_pos - self.env.target)
            delta_D = current_dist - final_dist
            dist_bonus = delta_D * 50.0 * (1.0 + urgency)

            energy_penalty = 0.05 * np.sum(force ** 2)
            score = q_val + dist_bonus - energy_penalty

            if score > best_score:
                best_score = score
                best_force = force
                best_traj = trajectory

        return best_force, best_traj


# ==================== 5. 渲染 ====================
def main():
    env = Env3D()
    tm = TM_Algorithm_3D(env)

    state = np.concatenate([env.start, [0.0, 0.0, 0.0]])
    history_pos = [state[:3].copy()]
    frame_idx = 0
    is_success = False

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(env.target[0], env.target[1], env.target[2], c='g', marker='*', s=800)

    for ox, oy, oz, r, h in env.obstacles:
        z_vals = np.linspace(oz, oz + h, 10)
        theta = np.linspace(0, 2 * np.pi, 15)
        for zv in z_vals:
            ax.plot(ox + r * np.cos(theta), oy + r * np.sin(theta), [zv] * 15, color='red', alpha=0.1)

    path_line, = ax.plot([], [], [], 'b-', lw=2, alpha=0.5)
    best_pred, = ax.plot([], [], [], 'y--', lw=2)

    line_spine, = ax.plot([], [], [], 'o-', lw=5, color='#2c3e50', ms=6)
    line_pelvis, = ax.plot([], [], [], '-', lw=4, color='#8e44ad')
    line_larm, = ax.plot([], [], [], 'o-', lw=3, color='#e67e22', ms=4)
    line_rarm, = ax.plot([], [], [], 'o-', lw=3, color='#d35400', ms=4)
    line_lleg, = ax.plot([], [], [], 'o-', lw=4, color='#3498db', ms=5)
    line_rleg, = ax.plot([], [], [], 'o-', lw=4, color='#e74c3c', ms=5)

    max_steps = 1000
    step_cnt = 0

    def draw_figure(current_pos, heading, local_joints):
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        def get_global(joint_name):
            rel = local_joints[joint_name]
            return current_pos + R @ rel

        h = get_global("Hips")
        sp = get_global("Spine")
        n = get_global("Neck")
        la, lf, lh = get_global("LeftArm"), get_global("LeftForeArm"), get_global("LeftHand")
        ra, rf, rh = get_global("RightArm"), get_global("RightForeArm"), get_global("RightHand")
        lu, ll, lft = get_global("LeftUpLeg"), get_global("LeftLeg"), get_global("LeftFoot")
        ru, rl, rft = get_global("RightUpLeg"), get_global("RightLeg"), get_global("RightFoot")

        line_spine.set_data([h[0], sp[0], n[0]], [h[1], sp[1], n[1]])
        line_spine.set_3d_properties([h[2], sp[2], n[2]])
        line_pelvis.set_data([lu[0], ru[0]], [lu[1], ru[1]])
        line_pelvis.set_3d_properties([lu[2], ru[2]])

        line_larm.set_data([n[0], la[0], lf[0], lh[0]], [n[1], la[1], lf[1], lh[1]])
        line_larm.set_3d_properties([n[2], la[2], lf[2], lh[2]])
        line_rarm.set_data([n[0], ra[0], rf[0], rh[0]], [n[1], ra[1], rf[1], rh[1]])
        line_rarm.set_3d_properties([n[2], ra[2], rf[2], rh[2]])

        line_lleg.set_data([lu[0], ll[0], lft[0]], [lu[1], ll[1], lft[1]])
        line_lleg.set_3d_properties([lu[2], ll[2], lft[2]])
        line_rleg.set_data([ru[0], rl[0], rft[0]], [ru[1], rl[1], rft[1]])
        line_rleg.set_3d_properties([ru[2], rl[2], rft[2]])

    def update(frame):
        nonlocal state, step_cnt, frame_idx, is_success

        if is_success:
            ax.set_title("Target Reached! 绝杀成功！", fontsize=20, color='green')
            return path_line, best_pred, line_spine, line_pelvis, line_larm, line_rarm, line_lleg, line_rleg

        if step_cnt >= max_steps:
            horizontal_dist = math.sqrt((state[0] - env.target[0]) ** 2 + (state[1] - env.target[1]) ** 2)
            ax.set_title(f"Failed: Time Out! (Dist: {horizontal_dist:.2f})", fontsize=20, color='red')
            return path_line, best_pred, line_spine, line_pelvis, line_larm, line_rarm, line_lleg, line_rleg

        # ==================== 核心替换区 ====================
        # 1. 模拟外部事件输入（前半程悠闲，后半程逃命）
        if step_cnt < max_steps * 0.4:
            event = "Taking a leisurely walk"
        else:
            event = "A dog is chasing me, run for my life!"

        # 2. 模型推理得出 Sigmoid 紧急度
        emb = text_encoder.encode([event], convert_to_tensor=True).to(DEVICE)
        with torch.no_grad():
            urgency = urgency_brain(emb).item()

        # 3. 根据紧急度修改物理引擎的速度上限
        if urgency > 0.6:
            env.max_speed = 2.5  # 狂奔
        else:
            env.max_speed = 0.8  # 漫步
        # ====================================================

        # TM 导航
        best_force, best_traj = tm.get_action(state, urgency)
        state = env.step(state, best_force)
        step_cnt += 1

        # 渲染更新
        vel = state[3:]
        heading = math.atan2(vel[1], vel[0]) if np.linalg.norm(vel) > 0.1 else 0.0
        frame_idx = (frame_idx + (3 if urgency > 0.6 else 1)) % frames_count
        current_local = local_joints_data[frame_idx]

        draw_figure(state[:3], heading, current_local)
        history_pos.append(state[:3].copy())

        hx = [p[0] for p in history_pos]
        hy = [p[1] for p in history_pos]
        hz = [p[2] for p in history_pos]
        path_line.set_data(hx, hy)
        path_line.set_3d_properties(hz)

        if len(best_traj) > 0:
            bx = [p[0] for p in best_traj]
            by = [p[1] for p in best_traj]
            bz = [p[2] for p in best_traj]
            best_pred.set_data(bx, by)
            best_pred.set_3d_properties(bz)

        horizontal_dist = math.sqrt((state[0] - env.target[0]) ** 2 + (state[1] - env.target[1]) ** 2)

        if horizontal_dist < 0.5:
            is_success = True
            state[0] = env.target[0]
            state[1] = env.target[1]
            draw_figure(state[:3], heading, current_local)
            history_pos[-1] = state[:3].copy()
            path_line.set_data([p[0] for p in history_pos], [p[1] for p in history_pos])
            path_line.set_3d_properties([p[2] for p in history_pos])
            best_pred.set_data([], [])
            best_pred.set_3d_properties([])
            ax.set_title("Target Reached!", fontsize=20, color='green')
            return path_line, best_pred, line_spine, line_pelvis, line_larm, line_rarm, line_lleg, line_rleg

        # 动态标题
        title_color = 'red' if urgency > 0.6 else 'green'
        ax.set_title(
            f"Event: {event}\nUrgency: {urgency:.2f} | Speed: {env.max_speed:.1f} | Dist: {horizontal_dist:.2f}m",
            fontsize=14, color=title_color)

        mid_x = (state[0] + env.target[0]) / 2
        mid_y = (state[1] + env.target[1]) / 2
        ax.set_xlim(mid_x - 7, mid_x + 7)
        ax.set_ylim(mid_y - 7, mid_y + 7)
        ax.set_zlim(0, 4)

        return path_line, best_pred, line_spine, line_pelvis, line_larm, line_rarm, line_lleg, line_rleg

    ani = animation.FuncAnimation(fig, update, frames=range(max_steps), interval=40, blit=False)
    plt.show()


if __name__ == "__main__":
    main()