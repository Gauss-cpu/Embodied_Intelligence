import json
import numpy as np
import math
import matplotlib

matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

ROOT = Path.home() / "Desktop/Physical_Intelligence"
FULL_BODY_DIR = ROOT / "dataset_cmu/full_body_3d/008"
full_body_file = list(FULL_BODY_DIR.rglob("*.json"))[0]

with open(full_body_file) as f:
    full_body_data = json.load(f)

traj_3d = full_body_data["trajectory"]
fps = full_body_data["fps"]
seq_len = len(traj_3d)

# ==================== 核心映射修改 ====================
# 如果动作反了，你只需要改这里的正负号！
FLIP_X = 1  # 左右镜像 (如果左右腿顺拐了，改成 -1)
FLIP_Y = 1  # 深度镜像 (如果你觉得他在倒着走，改成 -1)
FLIP_Z = 1  # 上下颠倒 (一般不用改)


def get_xyz(f, joint):
    p = f[joint]
    # p[0] 是左右，p[2] 是前后，p[1] 是上下
    # 我们把前后放到屏幕的横轴，上下放到屏幕的纵轴，左右作为深度
    x_plot = p[2] * FLIP_X  # 把前后 (CMU Z) 放到屏幕的 X 轴
    y_plot = p[0] * FLIP_Y  # 把左右 (CMU X) 放到屏幕的 Y 轴
    z_plot = p[1] * FLIP_Z  # 上下 (CMU Y) 放到屏幕的 Z 轴
    return x_plot, y_plot, z_plot


# ==================== 动态视野计算 ====================
all_x, all_y, all_z = [], [], []
for f in traj_3d:
    for j_name in ["Hips", "LeftFoot", "RightFoot", "LeftHand", "RightHand", "Neck"]:
        x, y, z = get_xyz(f, j_name)
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

margin = 5
x_min, x_max = min(all_x) - margin, max(all_x) + margin
y_min, y_max = min(all_y) - margin, max(all_y) + margin
z_min, z_max = min(all_z) - margin, max(all_z) + margin

# ==================== 渲染 ====================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
ax.set_box_aspect([1, 1, 1.5])

# 【关键】让相机正对着“前后”和“上下”所在的平面，这样就能看到人往右走
ax.view_init(elev=0, azim=-90)
ax.grid(False)
ax.set_title("全身动力学整合 (修复视角)", fontsize=16)

# 线条
line_spine, = ax.plot([], [], [], 'o-', lw=6, color='#2c3e50', ms=8)
line_pelvis, = ax.plot([], [], [], '-', lw=5, color='#8e44ad')
line_larm, = ax.plot([], [], [], 'o-', lw=4, color='#e67e22', ms=6)
line_rarm, = ax.plot([], [], [], 'o-', lw=4, color='#d35400', ms=6)
line_lleg, = ax.plot([], [], [], 'o-', lw=6, color='#3498db', ms=8)
line_rleg, = ax.plot([], [], [], 'o-', lw=6, color='#e74c3c', ms=8)


def draw(i):
    f = traj_3d[i]

    h_x, h_y, h_z = get_xyz(f, "Hips")
    s_x, s_y, s_z = get_xyz(f, "Spine")
    n_x, n_y, n_z = get_xyz(f, "Neck")

    # 1. 脊柱
    line_spine.set_data([h_x, s_x, n_x], [h_y, s_y, n_y])
    line_spine.set_3d_properties([h_z, s_z, n_z])

    # 2. 骨盆
    lu_x, lu_y, lu_z = get_xyz(f, "LeftUpLeg")
    ru_x, ru_y, ru_z = get_xyz(f, "RightUpLeg")
    line_pelvis.set_data([lu_x, ru_x], [lu_y, ru_y])
    line_pelvis.set_3d_properties([lu_z, ru_z])

    # 3. 左臂
    la_x, la_y, la_z = get_xyz(f, "LeftArm")
    lf_x, lf_y, lf_z = get_xyz(f, "LeftForeArm")
    lh_x, lh_y, lh_z = get_xyz(f, "LeftHand")
    line_larm.set_data([n_x, la_x, lf_x, lh_x], [n_y, la_y, lf_y, lh_y])
    line_larm.set_3d_properties([n_z, la_z, lf_z, lh_z])

    # 4. 右臂
    ra_x, ra_y, ra_z = get_xyz(f, "RightArm")
    rf_x, rf_y, rf_z = get_xyz(f, "RightForeArm")
    rh_x, rh_y, rh_z = get_xyz(f, "RightHand")
    line_rarm.set_data([n_x, ra_x, rf_x, rh_x], [n_y, ra_y, rf_y, rh_y])
    line_rarm.set_3d_properties([n_z, ra_z, rf_z, rh_z])

    # 5. 左腿
    ll_x, ll_y, ll_z = get_xyz(f, "LeftLeg")
    lf2_x, lf2_y, lf2_z = get_xyz(f, "LeftFoot")
    line_lleg.set_data([lu_x, ll_x, lf2_x], [lu_y, ll_y, lf2_y])
    line_lleg.set_3d_properties([lu_z, ll_z, lf2_z])

    # 6. 右腿
    rl_x, rl_y, rl_z = get_xyz(f, "RightLeg")
    rf2_x, rf2_y, rf2_z = get_xyz(f, "RightFoot")
    line_rleg.set_data([ru_x, rl_x, rf2_x], [ru_y, rl_y, rf2_y])
    line_rleg.set_3d_properties([ru_z, rl_z, rf2_z])

    return line_spine, line_pelvis, line_larm, line_rarm, line_lleg, line_rleg,


ani = animation.FuncAnimation(fig, draw, frames=range(0, seq_len, 2),
                              interval=1000 / (fps / 2), blit=False)
plt.tight_layout()
plt.show()