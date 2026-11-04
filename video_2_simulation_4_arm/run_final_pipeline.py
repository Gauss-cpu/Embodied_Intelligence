import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import numpy as np
import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from scipy.signal import savgol_filter

ROOT = Path.home() / "Desktop/Physical_Intelligence"
AMYGDALA_DIR = ROOT / "amygdala_layer/event2score"
sys.path.append(str(ROOT / "video_2_simulation_4_arm"))
from arm_dynamics import mass_matrix, gravity_term, step_dynamics

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 1. amygdala model
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

def event_to_weight(event_text):
    emb = _encoder.encode([event_text], convert_to_tensor=True).to(DEVICE)
    with torch.no_grad():
        return float(_valence_model(emb).item())

# 2. Calculating pipeline
def get_arm_trajectory(json_path, valence):
    with open(json_path) as f:
        data = json.load(f)
    traj = data["trajectory"]
    N = min(len(traj), 120)
    dt_sim = 1.0 / 30.0

    q_initial = np.array([f["q"] for f in traj[:N]])
    q_initial_smooth = q_initial.copy()
    if N >= 7:
        for j in range(8):
            q_initial_smooth[:, j] = savgol_filter(q_initial[:, j], 7, 2)

    for i in [0, 2, 4, 7]:
        q_initial_smooth[:, i] = 0.0

    trajectory_gain = 1.0 + 0.3 * valence
    q_emotional_target = q_initial_smooth.copy()
    q_emotional_target[:, 1] *= trajectory_gain
    q_emotional_target[:, 3] *= trajectory_gain

    qd_target = np.gradient(q_emotional_target, axis=0) / dt_sim
    qd_target = np.clip(qd_target, -10.0, 10.0)
    qdd_target = np.gradient(qd_target, axis=0) / dt_sim
    qdd_target = np.clip(qdd_target, -50.0, 50.0)

    tau_emotional_ff = np.zeros((N, 8))
    for i in range(N):
        M = mass_matrix(q_emotional_target[i])
        G = gravity_term(q_emotional_target[i])
        tau_emotional_ff[i] = M @ qdd_target[i] + G

    torque_boost = 1.0 + 0.15 * abs(valence)
    tau_emotional_ff *= torque_boost
    tau_emotional_ff = np.clip(tau_emotional_ff, -150, 150)

    y = np.zeros(16)
    y[:8] = q_emotional_target[0]
    y[8:] = qd_target[0]
    q_sim = []

    for i in range(N):
        tau = tau_emotional_ff[i]
        q_tgt = q_emotional_target[i]
        qd_tgt = qd_target[i]
        Kp, Kd = 80.0, 8.0
        sub_steps = 5
        dt_sub = dt_sim / sub_steps

        for _ in range(sub_steps):
            q, qd = y[:8], y[8:]
            tau_fb = Kp * (q_tgt - q) + Kd * (qd_tgt - qd)
            tau_total = np.clip(tau + tau_fb, -250, 250)
            q_new, qd_new, _ = step_dynamics(q, qd, tau_total, dt=dt_sub, friction=5.0)
            y[8:] = np.clip(qd_new, -15.0, 15.0)
            y[:8] = q_new

        q_sim.append(y[:8].copy())

    return np.array(q_sim)

if __name__ == "__main__":
    dataset = ROOT / "dataset_penn"
    default_q = dataset / "penn_baseball_pitch/video_0020_q.json"
    q_sim_test = get_arm_trajectory(str(default_q), "今天拿了全班第一，很高兴")
    print(f"计算完成，数据形状: {q_sim_test.shape}")