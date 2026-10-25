import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
from pathlib import Path

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ==================== 物理参数 ====================
g = 9.81
L1, L2 = 0.45, 0.45
m1, m2 = 10.0, 4.0

def forward_kinematics_2d(q):
    th1, th2 = q
    abs_th1 = th1
    abs_th2 = th1 - th2
    p_hip = np.array([0.0, 0.0])
    p_knee = p_hip + L1 * np.array([np.sin(abs_th1), -np.cos(abs_th1)])
    p_foot = p_knee + L2 * np.array([np.sin(abs_th2), -np.cos(abs_th2)])
    return p_knee, p_foot

def jacobian(f, q, eps=1e-5):
    q = np.asarray(q, dtype=float); n = len(q)
    f0 = np.asarray(f(q), dtype=float); J = np.zeros((len(f0), n))
    for i in range(n):
        qp = q.copy(); qp[i] += eps
        qm = q.copy(); qm[i] -= eps
        J[:, i] = (f(qp) - f(qm)) / (2 * eps)
    return J

def mass_matrix(q):
    p1_fn = lambda qq: forward_kinematics_2d(qq)[0]
    p2_fn = lambda qq: forward_kinematics_2d(qq)[1]
    Jv1 = jacobian(p1_fn, q); Jv2 = jacobian(p2_fn, q)
    return m1 * Jv1.T @ Jv1 + m2 * Jv2.T @ Jv2

def gravity_term(q):
    p1_fn = lambda qq: forward_kinematics_2d(qq)[0]
    p2_fn = lambda qq: forward_kinematics_2d(qq)[1]
    Jv1 = jacobian(p1_fn, q); Jv2 = jacobian(p2_fn, q)
    G = np.array([0.0, -g])
    return -(m1 * Jv1.T @ G + m2 * Jv2.T @ G)

# ==================== 加载模型 ====================
class ActorCritic(nn.Module):
    def __init__(self, state_dim=20, action_dim=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) - 1.0)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared(state)
        return self.actor_mean(x), self.actor_log_std.exp().expand_as(self.actor_mean(x)), self.critic(x)

PPO_MODEL_PATH = Path.home() / "Desktop/Physical_Intelligence/video_2_simulation_4_leg/ppo_leg_policy.pt"
SCALER_PATH = Path.home() / "Desktop/Physical_Intelligence/video_2_simulation_4_leg/scalers.pkl"
DATA_PATH = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/features/leg_dataset.npz"

policy = ActorCritic(20, 4).to(DEVICE)
try:
    policy.load_state_dict(torch.load(PPO_MODEL_PATH, map_location=DEVICE))
    print("✅ PPO 策略加载成功！")
except:
    print("⚠️ 使用 DNN 权重兜底...")
    dnn_weights = torch.load(Path.home() / "Desktop/Physical_Intelligence/video_2_simulation_4_leg/dnn_torque_model.pt", map_location=DEVICE)
    policy.shared[0].weight.data = dnn_weights['net.0.weight'].data
    policy.shared[0].bias.data = dnn_weights['net.0.bias'].data
    policy.shared[2].weight.data = dnn_weights['net.3.weight'].data
    policy.shared[2].bias.data = dnn_weights['net.3.bias'].data

with open(SCALER_PATH, "rb") as f:
    scaler_X = pickle.load(f)["scaler_X"]

# ==================== 选取测试数据 ====================
data = np.load(DATA_PATH, allow_pickle=True)
ref_X = data["X"]
seq_len = 150
start_idx = 100000
seq_X = ref_X[start_idx : start_idx + seq_len]

q = np.array([seq_X[0, 8], seq_X[0, 9]])
qd = np.array([0.0, 0.0])
q_history = [q.copy()]

print("开始物理推演...")
dt = 0.01

for step in range(seq_len - 1):
    raw_state = seq_X[step].copy()
    raw_state[8], raw_state[9] = q[0], q[1]
    raw_state[12], raw_state[13] = qd[0], qd[1]

    state_norm = scaler_X.transform(raw_state.reshape(1, -1))[0]
    state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mean, _, _ = policy(state_t)
        action = mean.cpu().numpy()[0]

    tau = np.clip(action[:2] * 20.0, -100.0, 100.0)

    for _ in range(3):
        M = mass_matrix(q)
        G = gravity_term(q)
        try:
            qdd = np.linalg.solve(M, tau - G - qd * 5.0)
        except:
            qdd = np.zeros(2)

        qd = np.clip(qd + qdd * dt, -20.0, 20.0)
        q = np.clip(q + qd * dt, -math.pi*2, math.pi*2)

    if not np.all(np.isfinite(q)):
        print(f"❌ 第 {step} 帧发生 NaN 爆炸！")
        break

    q_history.append(q.copy())

# ==================== 横向 3D 动画 ====================
fig = plt.figure(figsize=(10, 6))  # 把画幅改宽，适应横向
ax = fig.add_subplot(111, projection='3d')

# 【关键修改】坐标轴映射：X 是前后，Z 是上下，Y 全是 0
# 这样相机只要从 Y 轴看过去，就是完美的侧视图
ax.set_xlim(-1.0, 1.0) # 前后范围
ax.set_ylim(-0.1, 0.1) # 左右范围（压扁）
ax.set_zlim(-1.2, 0.2) # 上下范围
ax.set_box_aspect([2, 0.1, 1.5])

# 视角：从正侧面看（azim=0，看向 -Y 方向）
ax.view_init(elev=0, azim=-90)

# 画一条地面基准线
ax.plot([-1.0, 1.0], [0, 0], [0, 0], 'k--', lw=2, alpha=0.7)

line, = ax.plot([], [], [], 'o-', lw=5, color='#2ecc71', ms=10)
joint_hip, = ax.plot([], [], [], 'o', color='red', ms=8)

def draw(i):
    q = q_history[i]
    p_knee, p_foot = forward_kinematics_2d(q)

    # X = 前后，Y = 0（固定），Z = 上下
    xs = [0, p_knee[0], p_foot[0]]
    ys = [0, 0, 0]
    zs = [0, p_knee[1], p_foot[1]]

    line.set_data(xs, ys)
    line.set_3d_properties(zs)

    joint_hip.set_data([0], [0])
    joint_hip.set_3d_properties([0])

    ax.set_title(f"PPO 在线物理控制 (帧 {i})", fontsize=14)
    return line, joint_hip,

ani = animation.FuncAnimation(fig, draw, frames=range(0, len(q_history)),
                              interval=20, blit=False)
plt.show()