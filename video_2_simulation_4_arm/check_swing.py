import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

DATASET = Path.home() / "Desktop/Physical_Intelligence/dataset_penn"

files = {
    "棒球投球": DATASET / "penn_baseball_pitch/video_0020_q.json",
    "引体向上": DATASET / "penn_pullup/video_1252_q.json",
    "网球正手": DATASET / "penn_tennis_forehand/video_2007_q.json",
    "挺举":     DATASET / "penn_clean_and_jerk/video_0715_q.json",
    "保龄球":   DATASET / "penn_bowl/video_0520_q.json",
}

# 如果某个文件不存在，跳过
files = {k: v for k, v in files.items() if v.exists()}

if not files:
    print(f"没找到任何文件，检查 {DATASET}")
    exit()

fig, axes = plt.subplots(len(files), 1, figsize=(12, 3 * len(files)))
if len(files) == 1:
    axes = [axes]

for i, (name, path) in enumerate(files.items()):
    with open(path) as f:
        data = json.load(f)
    q_traj = data["trajectory"]
    t = np.array([qf["t"] for qf in q_traj])
    q2 = np.degrees([qf["q"][1] for qf in q_traj])
    q4 = np.degrees([qf["q"][3] for qf in q_traj])

    axes[i].plot(t, q2, label="q2 (抬升)", lw=1.5)
    axes[i].plot(t, q4, label="q4 (肘)", lw=1.5)
    axes[i].set_ylabel("角度 (°)")
    axes[i].set_title(f"{name} — {path.name}  ({len(q_traj)} 帧)")
    axes[i].legend(loc="upper right")
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(0, 200)

    # 打印统计
    print(f"{name:10s}: q2 ∈ [{q2.min():.1f}, {q2.max():.1f}]  振幅 {q2.max()-q2.min():.1f}°")
    print(f"{'':10s}  q4 ∈ [{q4.min():.1f}, {q4.max():.1f}]  振幅 {q4.max()-q4.min():.1f}°")

axes[-1].set_xlabel("时间 (s)")
plt.tight_layout()
plt.savefig("swing_check_penn.png", dpi=120)
print("\n已保存: swing_check_penn.png")
plt.show()