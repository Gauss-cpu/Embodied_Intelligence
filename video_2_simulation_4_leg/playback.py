import json
import sys
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def play_json_3d(json_path):
    with open(json_path) as f:
        data = json.load(f)

    traj = data["trajectory"]
    fps = data["fps"]
    print(f"播放: {json_path.name}, 帧数: {len(traj)}")

    # ================= 动态计算视野范围 =================
    # 遍历所有帧，找出 X, Y, Z 的最大最小值
    xs, ys, zs = [], [], []
    for f in traj:
        for joint in ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]:
            xs.append(f[joint][0])
            ys.append(f[joint][1])
            zs.append(f[joint][2])

    margin = 10  # 边缘留白
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin
    z_min, z_max = min(zs) - margin, max(zs) + margin

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 应用动态视野，把范围缩小到火柴人身上
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # 坐标轴间距自动调整，看得很清楚
    ax.set_xlabel('X (Left/Right)')
    ax.set_ylabel('Y (Up)')
    ax.set_zlabel('Z (Forward)')

    # 视角：从侧面看
    ax.view_init(elev=10, azim=-90)

    # ================= 绘制骨骼（线） =================
    line_L, = ax.plot([], [], [], '-', lw=2, color='#3498db', alpha=0.7)  # 左腿线
    line_R, = ax.plot([], [], [], '-', lw=2, color='#e74c3c', alpha=0.7)  # 右腿线

    # ================= 绘制质点（散点球） =================
    # 用 scatter 画，可以设置极大尺寸，且不受线条遮挡影响
    # s=150 代表球的面积，非常醒目
    pt_hip, = ax.plot([], [], [], 'o', color='black', ms=15, markeredgecolor='white')

    # 左腿质点（蓝）
    pt_lu, = ax.plot([], [], [], 'o', color='#3498db', ms=12, markeredgecolor='white')
    pt_ll, = ax.plot([], [], [], 'o', color='#3498db', ms=12, markeredgecolor='white')
    pt_lf, = ax.plot([], [], [], 'o', color='#3498db', ms=12, markeredgecolor='white')

    # 右腿质点（红）
    pt_ru, = ax.plot([], [], [], 'o', color='#e74c3c', ms=12, markeredgecolor='white')
    pt_rl, = ax.plot([], [], [], 'o', color='#e74c3c', ms=12, markeredgecolor='white')
    pt_rf, = ax.plot([], [], [], 'o', color='#e74c3c', ms=12, markeredgecolor='white')

    # 质点名称文字标注
    text_hip = ax.text(0, 0, 0, "Hips", color='black', fontsize=12, fontweight='bold')
    text_lu = ax.text(0, 0, 0, "L_Knee", color='#3498db', fontsize=10)
    text_ll = ax.text(0, 0, 0, "L_Ankle", color='#3498db', fontsize=10)
    text_ru = ax.text(0, 0, 0, "R_Knee", color='#e74c3c', fontsize=10)
    text_rl = ax.text(0, 0, 0, "R_Ankle", color='#e74c3c', fontsize=10)

    def draw(i):
        f = traj[i]
        hip = f["Hips"]
        lu, ll, lf = f["LeftUpLeg"], f["LeftLeg"], f["LeftFoot"]
        ru, rl, rf = f["RightUpLeg"], f["RightLeg"], f["RightFoot"]

        # 更新线
        line_L.set_data([hip[0], lu[0], ll[0], lf[0]], [hip[1], lu[1], ll[1], lf[1]])
        line_L.set_3d_properties([hip[2], lu[2], ll[2], lf[2]])

        line_R.set_data([hip[0], ru[0], rl[0], rf[0]], [hip[1], ru[1], rl[1], rf[1]])
        line_R.set_3d_properties([hip[2], ru[2], rl[2], rf[2]])

        # 更新质点
        pt_hip.set_data([hip[0]], [hip[1]])
        pt_hip.set_3d_properties([hip[2]])

        pt_lu.set_data([lu[0]], [lu[1]])
        pt_lu.set_3d_properties([lu[2]])
        pt_ll.set_data([ll[0]], [ll[1]])
        pt_ll.set_3d_properties([ll[2]])
        pt_lf.set_data([lf[0]], [lf[1]])
        pt_lf.set_3d_properties([lf[2]])

        pt_ru.set_data([ru[0]], [ru[1]])
        pt_ru.set_3d_properties([ru[2]])
        pt_rl.set_data([rl[0]], [rl[1]])
        pt_rl.set_3d_properties([rl[2]])
        pt_rf.set_data([rf[0]], [rf[1]])
        pt_rf.set_3d_properties([rf[2]])

        # 更新文字位置
        text_hip.set_position((hip[0], hip[1]))
        text_hip.set_3d_properties(hip[2] + 5)

        text_lu.set_position((lu[0], lu[1]))
        text_lu.set_3d_properties(lu[2] + 5)
        text_ll.set_position((ll[0], ll[1]))
        text_ll.set_3d_properties(ll[2] + 5)

        text_ru.set_position((ru[0], ru[1]))
        text_ru.set_3d_properties(ru[2] + 5)
        text_rl.set_position((rl[0], rl[1]))
        text_rl.set_3d_properties(rl[2] + 5)

        return line_L, line_R, pt_hip, pt_lu, pt_ll, pt_lf, pt_ru, pt_rl, pt_rf, text_hip, text_lu, text_ll, text_ru, text_rl

    ani = animation.FuncAnimation(fig, draw, frames=range(0, len(traj), 2),
                                  interval=1000 / (fps / 2), blit=False)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        OUT_DIR = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/processed_legs_3d"
        files = list(OUT_DIR.rglob("*.json"))
        if not files: sys.exit()
        file_path = files[0]

    play_json_3d(file_path)