import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
from sentence_transformers import SentenceTransformer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ROOT = Path.home() / "Desktop/Physical_Intelligence"
AMYGDALA_DIR = ROOT / "amygdala_layer/event2score"


# ==================== 1. 加载杏仁核 (Amygdala) ====================
class ValenceNet(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


print("加载 Amygdala 模型...")
_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
_valence_model = ValenceNet().to(DEVICE)
_valence_model.load_state_dict(torch.load(AMYGDALA_DIR / "valence_model.pt", map_location=DEVICE))
_valence_model.eval()


def get_event_valence(event_text):
    emb = _encoder.encode([event_text], convert_to_tensor=True).to(DEVICE)
    with torch.no_grad():
        return float(_valence_model(emb).item())


# ==================== 2. 物理参数 ====================
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
    p1_fn = lambda qq: forward_kinematics_2d(qq)[0]
    p2_fn = lambda qq: forward_kinematics_2d(qq)[1]
    Jv1 = jacobian(p1_fn, q);
    Jv2 = jacobian(p2_fn, q)
    return m1 * Jv1.T @ Jv1 + m2 * Jv2.T @ Jv2


def gravity_term(q):
    p1_fn = lambda qq: forward_kinematics_2d(qq)[0]
    p2_fn = lambda qq: forward_kinematics_2d(qq)[1]
    Jv1 = jacobian(p1_fn, q);
    Jv2 = jacobian(p2_fn, q)
    G = np.array([0.0, -g])
    return -(m1 * Jv1.T @ G + m2 * Jv2.T @ G)


# ==================== 3. 加载 DNN 力矩模型 ====================
class TorqueNet(nn.Module):
    def __init__(self, input_dim=20, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


DNN_PATH = ROOT / "video_2_simulation_4_leg/dnn_torque_model.pt"
SCALER_PATH = ROOT / "video_2_simulation_4_leg/scalers.pkl"
DATA_PATH = ROOT / "dataset_cmu/features/leg_dataset.npz"

dnn_model = TorqueNet(20, 4).to(DEVICE)
dnn_model.load_state_dict(torch.load(DNN_PATH, map_location=DEVICE))
dnn_model.eval()

with open(SCALER_PATH, "rb") as f:
    scaler_X = pickle.load(f)["scaler_X"]


# ==================== 4. 情绪物理推演 ====================
def simulate_leg_emotion(valence):
    data = np.load(DATA_PATH, allow_pickle=True)
    ref_X = data["X"]
    seq_len = 150
    start_idx = 100000
    seq_X = ref_X[start_idx: start_idx + seq_len]

    q = np.array([seq_X[0, 8], seq_X[0, 9]])  # 髋, 膝
    qd = np.array([0.0, 0.0])
    q_history = [q.copy()]

    dt = 0.01
    emotion_gain = 1.0 + 0.8 * valence

    for step in range(seq_len - 1):
        raw_state = seq_X[step].copy()

        target_q = np.array([raw_state[8], raw_state[9]]) * emotion_gain

        obs_state = raw_state.copy()
        obs_state[8], obs_state[9] = q[0], q[1]
        obs_state[12], obs_state[13] = qd[0], qd[1]

        state_norm = scaler_X.transform(obs_state.reshape(1, -1))[0]
        state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            tau_base = dnn_model(state_t).cpu().numpy()[0][:2]

        tau_ff = np.clip(tau_base * emotion_gain, -200.0, 200.0)

        for _ in range(3):
            M = mass_matrix(q)
            G = gravity_term(q)

            Kp = 300.0
            Kd = 30.0
            tau_fb = Kp * (target_q - q) - Kd * qd

            tau_total = tau_ff + tau_fb

            try:
                qdd = np.linalg.solve(M, tau_total - G)
            except:
                qdd = np.zeros(2)

            qd = np.clip(qd + qdd * dt, -30.0, 30.0)
            q = np.clip(q + qd * dt, -math.pi * 2, math.pi * 2)

        if not np.all(np.isfinite(q)):
            break

        q_history.append(q.copy())

    return q_history


# ==================== 5. 执行对比推演 ====================
# 直接用英文 Prompt 喂给模型
event_happy = "I got first place in class, I am so happy!"
val_happy = get_event_valence(event_happy)
print(f"事件: '{event_happy}' -> valence: {val_happy:+.2f}")

event_sad = "I fell down and it hurts so much."
val_sad = get_event_valence(event_sad)
print(f"事件: '{event_sad}' -> valence: {val_sad:+.2f}")

print("推演高兴状态...")
q_happy = simulate_leg_emotion(val_happy)
print("推演难过状态...")
q_sad = simulate_leg_emotion(val_sad)

# ==================== 6. 双屏动画对比 ====================
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for ax, title in zip([ax1, ax2], [f"高兴 (valence={val_happy:+.2f})", f"难过 (valence={val_sad:+.2f})"]):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-1.2, 0.2)
    ax.set_box_aspect([2, 0.1, 1.5])
    ax.view_init(elev=0, azim=-90)
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k--', lw=2, alpha=0.5)
    ax.set_title(title, fontsize=16)

line_h, = ax1.plot([], [], [], 'o-', lw=5, color='#f1c40f', ms=8)
line_s, = ax2.plot([], [], [], 'o-', lw=5, color='#3498db', ms=8)


def draw(i):
    qh = q_happy[i]
    kh, fh = forward_kinematics_2d(qh)
    line_h.set_data([0, kh[0], fh[0]], [0, 0, 0])
    line_h.set_3d_properties([0, kh[1], fh[1]])

    qs = q_sad[i]
    ks, fs = forward_kinematics_2d(qs)
    line_s.set_data([0, ks[0], fs[0]], [0, 0, 0])
    line_s.set_3d_properties([0, ks[1], fs[1]])

    return line_h, line_s


max_frames = min(len(q_happy), len(q_sad))
ani = animation.FuncAnimation(fig, draw, frames=range(max_frames), interval=20, blit=False)
plt.tight_layout()
plt.show()