import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import numpy as np
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

ROOT = Path.home() / "Desktop/Physical_Intelligence"
sys.path.append(str(ROOT / "video_2_simulation_4_leg"))
sys.path.append(str(ROOT / "video_2_simulation_4_arm"))
sys.path.append(str(ROOT / "tm_algorithm"))

from run_final_pipeline import get_arm_trajectory
from run_leg_emotion import get_leg_trajectory
from tm_navigation_with_full_body import Env3D, TM_Algorithm_3D

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class ValenceNet(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x): return self.net(x)


class UrgencyNet(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x): return self.net(x)


print("loading NLP brain...")
_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
_valence_model = ValenceNet().to(DEVICE)
_valence_model.load_state_dict(torch.load(ROOT / "amygdala_layer/event2score/valence_model.pt", map_location=DEVICE))
_valence_model.eval()

_urgency_model = UrgencyNet().to(DEVICE)
_urgency_model.load_state_dict(torch.load(ROOT / "urgency_policy/urgency_model.pth", map_location=DEVICE))
_urgency_model.eval()


def get_emotions(text):
    emb = _encoder.encode([text], convert_to_tensor=True).to(DEVICE)
    with torch.no_grad():
        val = float(_valence_model(emb).item())
        urg = float(_urgency_model(emb).item())
    return val, urg

event_text = "I am taking a leisurely walk in the park."
print(f"events: {event_text}")
valence, urgency = get_emotions(event_text)
print(f"Valence: {valence:+.2f} | Urgency: {urgency:.2f}")

print("rehearsing leg and arm trajectory...")
q_arm = get_arm_trajectory(str(ROOT / "dataset_penn/penn_baseball_pitch/video_0020_q.json"), valence)
q_leg = get_leg_trajectory(valence)

print("rehearsing TM algorithm...")
env = Env3D()
tm = TM_Algorithm_3D(env, num_samples=50, horizon_steps=10)
state = np.concatenate([env.start, [0.0, 0.0, 0.0]])
nav_path_x, nav_path_y, nav_path_z = [], [], []

for _ in range(120):
    best_force, best_traj = tm.get_action(state, urgency)
    state = env.step(state, best_force)
    nav_path_x.append(state[0])
    nav_path_y.append(state[1])
    nav_path_z.append(state[2])

print(f"length of arm trajectories: {len(q_arm)}, length of leg trajectories: {len(q_leg)}, length of navigation pathways: {len(nav_path_x)}")
max_frames = min(120, len(q_arm), len(q_leg))

fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"Affective Embodied AI | Event: '{event_text}'", fontsize=20, fontweight='bold')

# upper-left: navigation
ax_nav = fig.add_subplot(221, projection='3d')
ax_nav.set_title(f"TM Navigation (Urgency: {urgency:.2f})", fontsize=14)
ax_nav.set_xlim(0, 12);
ax_nav.set_ylim(0, 12);
ax_nav.set_zlim(0, 5)
ax_nav.view_init(elev=30, azim=-45)
ax_nav.scatter(env.target[0], env.target[1], env.target[2], c='g', marker='*', s=300)
nav_path_line, = ax_nav.plot([], [], [], 'b-', lw=2)
nav_dot, = ax_nav.plot([], [], [], 'bo', ms=10)

# upper-right：arm
ax_arm = fig.add_subplot(222)
ax_arm.set_title("8-DOF Arm Physics", fontsize=14)
ax_arm.set_xlim(0, max_frames);
ax_arm.set_ylim(-200, 200)
arm_line, = ax_arm.plot([], [], 'r-', lw=2)

# lower-left：leg
ax_leg = fig.add_subplot(223)
ax_leg.set_title("2-DOF Leg Physics", fontsize=14)
ax_leg.set_xlim(0, max_frames);
ax_leg.set_ylim(-180, 180)
leg_line, = ax_leg.plot([], [], 'g-', lw=2)

# lower-right：emotions
ax_emo = fig.add_subplot(224)
ax_emo.set_title("Affective State Monitor", fontsize=14)
ax_emo.set_xlim(0, max_frames);
ax_emo.set_ylim(-1.1, 1.1)
emo_line, = ax_emo.plot([], [], 'm-', lw=2)
ax_emo.axhline(0, color='k', linestyle='--', alpha=0.5)


def update(frame):
    if frame >= max_frames:
        ani.event_source.stop()
        return nav_path_line, nav_dot, arm_line, leg_line, emo_line

    # 1. TM navigation
    nav_path_line.set_data(nav_path_x[:frame], nav_path_y[:frame])
    nav_path_line.set_3d_properties(nav_path_z[:frame])

    nav_dot.set_data([nav_path_x[frame]], [nav_path_y[frame]])
    nav_dot.set_3d_properties([nav_path_z[frame]])

    # 2. arm physics
    arm_line.set_data(np.arange(frame), np.degrees(q_arm[:frame, 1]))

    # 3. leg physics
    leg_line.set_data(np.arange(frame), np.degrees(q_leg[:frame, 0]))

    # 4. emotion monitor
    emo_line.set_data(np.arange(frame), [urgency] * frame)

    return nav_path_line, nav_dot, arm_line, leg_line, emo_line

ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=50, blit=False)
plt.tight_layout()
plt.show()