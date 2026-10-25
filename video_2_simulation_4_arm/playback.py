import json
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# === 从 arm3d.py 复制这些函数进来 ===
# rot_x, rot_y, rot_z
# R_shoulder, R_elbow, R_wrist
# forward_kinematics

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def R_shoulder(q):
    return rot_z(q[0]) @ rot_y(-q[1]) @ rot_z(q[2])
def R_elbow(q):
    return R_shoulder(q) @ rot_x(q[3]) @ rot_z(q[4])
def R_wrist(q):
    return R_elbow(q) @ rot_x(q[5]) @ rot_y(q[6]) @ rot_z(q[7])

L1, L2, L3 = 1.0, 0.8, 0.15

def forward_kinematics(q):
    Rs = R_shoulder(q); Re = R_elbow(q); Rw = R_wrist(q)
    local_down = np.array([0.0, 0.0, -1.0])
    upper_dir = Rs @ local_down
    fore_dir  = Re @ local_down
    hand_dir  = Rw @ local_down
    p_shoulder = np.zeros(3)
    p_elbow = p_shoulder + L1 * upper_dir
    p_wrist = p_elbow + L2 * fore_dir
    p_end   = p_wrist + L3 * hand_dir
    return p_elbow, p_wrist, p_end

# === 读轨迹 ===
import sys
from pathlib import Path

# 从命令行读要播哪个 json，不传就随机挑一个
DATASET = Path.home() / "Desktop/Physical_Intelligence/dataset_penn"
if len(sys.argv) > 1:
    json_path = Path(sys.argv[1])
else:
    # 随机挑一个
    all_json = sorted(DATASET.glob("penn_*/*.json"))
    print(f"数据集里共 {len(all_json)} 条轨迹")
    json_path = all_json[0]

print(f"播放: {json_path.name}")
with open(json_path) as f:
    data = json.load(f)

# 兼容两种格式：旧的 q_trajectory 是 list，新的是 dict 带 trajectory
if isinstance(data, dict):
    q_traj = data["trajectory"]
    print(f"  类别: {data.get('category')}, video_id: {data.get('video_id')}, "
          f"帧数: {len(q_traj)}")
else:
    q_traj = data

# 关键：把 q1 强制为 0（因为不可信）
qs = []
for qt in q_traj:
    q = list(qt["q"])
    q[0] = 0.0          # 关掉 q1
    qs.append(np.array(q))

print(f"加载 {len(qs)} 帧轨迹")

# === 动画 ===
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(-2.2, 0.5)
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.view_init(elev=20, azim=-60)

line, = ax.plot([], [], [], 'o-', lw=3, color='#4caf50', ms=8)

def draw(i):
    q = qs[i]
    pe, pw, pend = forward_kinematics(q)
    line.set_data_3d([0, pe[0], pw[0], pend[0]],
                     [0, pe[1], pw[1], pend[1]],
                     [0, pe[2], pw[2], pend[2]])
    return line,

ani = animation.FuncAnimation(fig, draw, frames=len(qs),
                              interval=40, blit=False, repeat=True)
plt.show()