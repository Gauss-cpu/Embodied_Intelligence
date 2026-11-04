import json
import numpy as np
import math
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys

ROOT = Path.home() / "Desktop/Physical_Intelligence"
sys.path.append(str(ROOT / "video_2_simulation_4_arm"))
from arm_dynamics import compute_inverse_dynamics


def analyze_trajectory(json_path, save_path="tracking_result.png"):
    with open(json_path) as f:
        data = json.load(f)
    traj = data["trajectory"]
    cat = data.get("category", "?")
    N = len(traj)
    dt = 1.0 / 30.0

    # savgol filter
    q_desired = np.array([qf["q"] for qf in traj])
    t_array = np.arange(N) * dt

    q_smooth = q_desired.copy()
    if N >= 7:
        for j in range(8):
            q_smooth[:, j] = savgol_filter(q_desired[:, j], 7, 2)
    elif N >= 5:
        for j in range(8):
            q_smooth[:, j] = savgol_filter(q_desired[:, j], 5, 2)
    else:
        q_smooth = q_desired.copy()

    # calculate speed
    qd = np.gradient(q_smooth, axis=0) / dt
    # calculate acceleration
    qdd = np.gradient(qd, axis=0) / dt

    qdd = np.clip(qdd, -100, 100)

    # lock down joints that are unclear or ambiguous
    for i in [0, 2, 4, 7]:
        q_smooth[:, i] = 0.0
        qd[:, i] = 0.0
        qdd[:, i] = 0.0

    print(f"trajectories: {cat}, {N} frames")
    print("calculating inverse dynamics...")

    tau_all = np.zeros((N, 8))

    for i in range(N):
        q = q_smooth[i]
        qd_val = qd[i]
        qdd_val = qdd[i]

        # using lagrange equation
        tau = compute_inverse_dynamics(q, qd_val, qdd_val)
        tau_all[i] = tau

        if i % 20 == 0:
            print(f"  {i}/{N}")

    print("completed")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t_array, np.degrees(q_smooth[:, 1]), 'b-', label="q2", lw=2)
    axes[0].plot(t_array, np.degrees(q_smooth[:, 3]), 'r-', label="q4", lw=2)
    axes[0].set_ylabel("joint angles (°)")
    axes[0].legend();
    axes[0].grid(True, alpha=0.3)

    for j in range(8):
        axes[1].plot(t_array, tau_all[:, j], label=f"τ{j + 1}")
    axes[1].set_ylabel("joint torque τ (N·m)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(ncol=4);
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\nsaved: {save_path}")
    plt.show()

    out_data = {
        "video_id": data.get("video_id", "?"),
        "category": data.get("category", "?"),
        "tau": tau_all.tolist()
    }
    with open("tau_result.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"saved: tau_result.json")

    return q_smooth, tau_all


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        dataset = ROOT / "dataset_penn"
        path = sorted((dataset / "penn_baseball_pitch").glob("*.json"))[0]

    analyze_trajectory(str(path))