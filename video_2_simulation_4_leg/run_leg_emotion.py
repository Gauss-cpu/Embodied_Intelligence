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
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ROOT = Path.home() / "Desktop/Physical_Intelligence"
AMYGDALA_DIR = ROOT / "amygdala_layer/event2score"

sys.path.append(str(ROOT / "video_2_simulation_4_leg"))
from leg_dynamics import step_dynamics


# 1. loading amygdala
class ValenceNet(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x): return self.net(x)


print("loading amygdala model...")
_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
_valence_model = ValenceNet().to(DEVICE)
_valence_model.load_state_dict(torch.load(AMYGDALA_DIR / "valence_model.pt", map_location=DEVICE))
_valence_model.eval()


def get_event_valence(event_text):
    emb = _encoder.encode([event_text], convert_to_tensor=True).to(DEVICE)
    with torch.no_grad():
        return float(_valence_model(emb).item())


# 2. kinematics
L1, L2 = 0.45, 0.45


def forward_kinematics_2d(q):
    th1, th2 = q
    p_hip = np.array([0.0, 0.0])
    p_knee = p_hip + L1 * np.array([np.sin(th1), -np.cos(th1)])
    p_foot = p_knee + L2 * np.array([np.sin(th1 + th2), -np.cos(th1 + th2)])
    return p_knee, p_foot


# 3. loading dnn
class TorqueNet(nn.Module):
    def __init__(self, input_dim=20, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x): return self.net(x)


DNN_PATH = ROOT / "video_2_simulation_4_leg/dnn_torque_model.pt"
SCALER_PATH = ROOT / "video_2_simulation_4_leg/scalers.pkl"
DATA_PATH = ROOT / "dataset_cmu/features/leg_dataset.npz"

print("loading DNN...")
dnn_model = TorqueNet(20, 4).to(DEVICE)
dnn_model.load_state_dict(torch.load(DNN_PATH, map_location=DEVICE))
dnn_model.eval()

with open(SCALER_PATH, "rb") as f:
    scaler_X = pickle.load(f)["scaler_X"]


# 4. simulation
def simulate_leg_emotion(valence):
    data = np.load(DATA_PATH, allow_pickle=True)
    ref_X = data["X"]
    seq_len = 150
    start_idx = 100000
    seq_X = ref_X[start_idx: start_idx + seq_len]

    q = np.array([seq_X[0, 8], seq_X[0, 9]])  # 初始角度
    qd = np.array([seq_X[0, 12], seq_X[0, 13]])  # 初始角速度
    q_history = [q.copy()]

    dt = 0.01
    emotion_gain = 1.0 + 0.2 * valence

    for step in range(seq_len - 1):
        raw_state = seq_X[step].copy()
        target_q = np.array([raw_state[8], raw_state[9]])

        obs_state = raw_state.copy()
        obs_state[8], obs_state[9] = q[0], q[1]
        obs_state[12], obs_state[13] = qd[0], qd[1]

        state_norm = scaler_X.transform(obs_state.reshape(1, -1))[0]
        state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            tau_base = dnn_model(state_t).cpu().numpy()[0][:2]

        tau_ff = np.clip(tau_base * emotion_gain, -200.0, 200.0)

        for _ in range(3):
            Kp = 300.0
            Kd = 30.0
            tau_fb = Kp * (target_q - q) - Kd * qd
            tau_total = np.clip(tau_ff + tau_fb, -250.0, 250.0)

            q, qd, qdd = step_dynamics(q, qd, tau_total, dt=dt, friction=5.0)

            qd = np.clip(qd, -15.0, 15.0)
            q = np.clip(q, -math.pi * 1.5, math.pi * 1.5)

        if not np.all(np.isfinite(q)): break
        q_history.append(q.copy())

    return q_history


# 5. execute comparative simulation
event_happy = "I won the lottery and got first place!"
val_happy = get_event_valence(event_happy)
print(f"event: '{event_happy}' -> valence: {val_happy:+.2f}")

event_sad = "My dog died and I lost my job."
val_sad = get_event_valence(event_sad)
print(f"event: '{event_sad}' -> valence: {val_sad:+.2f}")

print("simulating happiness...")
q_happy = simulate_leg_emotion(val_happy)
print("simulating sadness...")
q_sad = simulate_leg_emotion(val_sad)

# 6. simulation animation
fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for ax, title in zip([ax1, ax2], [f"高兴 (valence={val_happy:+.2f})", f"难过 (valence={val_sad:+.2f})"]):
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-1.2, 0.2)
    ax.set_box_aspect([2, 0.1, 1.5])
    ax.view_init(elev=10, azim=-90)
    ax.plot(np.array([-1.0, 1.0]), np.array([0, 0]), np.array([0, 0]), 'k--', lw=2, alpha=0.5)
    ax.set_title(title, fontsize=16)

line_h, = ax1.plot([], [], [], 'o-', lw=5, color='#f1c40f', ms=10, markeredgecolor='white')
line_s, = ax2.plot([], [], [], 'o-', lw=5, color='#3498db', ms=10, markeredgecolor='white')


def draw(i):
    qh = q_happy[i]
    kh, fh = forward_kinematics_2d(qh)
    xh = np.array([0, kh[0], fh[0]])
    yh = np.array([0, 0, 0])
    zh = np.array([0, kh[1], fh[1]])
    line_h.set_data(xh, yh)
    line_h.set_3d_properties(zh)

    qs = q_sad[i]
    ks, fs = forward_kinematics_2d(qs)
    xs = np.array([0, ks[0], fs[0]])
    ys = np.array([0, 0, 0])
    zs = np.array([0, ks[1], fs[1]])
    line_s.set_data(xs, ys)
    line_s.set_3d_properties(zs)

    return line_h, line_s


max_frames = min(len(q_happy), len(q_sad))
ani = animation.FuncAnimation(fig, draw, frames=max_frames, interval=30, blit=False)
plt.tight_layout()
plt.show()

def get_leg_trajectory(valence):
    data = np.load(DATA_PATH, allow_pickle=True)
    ref_X = data["X"]
    seq_len = 120
    start_idx = 100000
    seq_X = ref_X[start_idx: start_idx + seq_len]

    q = np.array([seq_X[0, 8], seq_X[0, 9]])
    qd = np.array([seq_X[0, 12], seq_X[0, 13]])
    q_history = [q.copy()]

    dt = 0.01
    emotion_gain = 1.0 + 0.2 * valence

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
            Kp = 300.0
            Kd = 30.0
            tau_fb = Kp * (target_q - q) - Kd * qd
            tau_total = np.clip(tau_ff + tau_fb, -250.0, 250.0)

            q, qd, qdd = step_dynamics(q, qd, tau_total, dt=dt, friction=15.0)
            qd = np.clip(qd, -15.0, 15.0)
            q = np.clip(q, -math.pi * 1.5, math.pi * 1.5)

        if not np.all(np.isfinite(q)): break
        q_history.append(q.copy())

    return np.array(q_history)