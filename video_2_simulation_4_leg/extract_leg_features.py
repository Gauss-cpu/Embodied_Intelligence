import json
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path
sys.path.append(str(Path.home() / "Desktop/Physical_Intelligence/video_2_simulation_4_leg"))
from leg_dynamics import compute_inverse_dynamics

IN_DIR = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/processed_legs_3d"
OUT_DIR = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# physical hyperparameter
m_thigh = 10.0
m_shank = 4.0
m_total = 2 * (m_thigh + m_shank) + 30.0  # approximately 30.0 kg weight for the upper body

# dimensionality reduction
def compute_angle(p1, p2):
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return np.arctan2(dz, -dy)

def process_file(json_path):
    with open(json_path) as f:
        data = json.load(f)

    traj = data["trajectory"]
    fps = data["fps"]
    dt = 1.0 / fps
    N = len(traj)
    if N < 10: return None

    hips = np.array([f["Hips"] for f in traj])
    lk = np.array([f["LeftUpLeg"] for f in traj])
    lf = np.array([f["LeftLeg"] for f in traj])
    lft = np.array([f["LeftFoot"] for f in traj])

    rk = np.array([f["RightUpLeg"] for f in traj])
    rf = np.array([f["RightLeg"] for f in traj])
    rft = np.array([f["RightFoot"] for f in traj])

    # 1. Barycenter of thigh and shank
    com_thigh_L = (hips + lk) / 2
    com_shank_L = (lk + lf) / 2
    com_thigh_R = (hips + rk) / 2
    com_shank_R = (rk + rf) / 2

    # 2. Barycenter of hips and legs
    com_total = (com_thigh_L * m_thigh + com_shank_L * m_shank +
                 com_thigh_R * m_thigh + com_shank_R * m_shank + hips * 30.0) / m_total

    # 3. The angle relative towards the ground
    q_hip_L = np.array([compute_angle(hips[i], lk[i]) for i in range(N)])
    q_knee_L = np.array([compute_angle(lk[i], lf[i]) for i in range(N)])
    q_hip_R = np.array([compute_angle(hips[i], rk[i]) for i in range(N)])
    q_knee_R = np.array([compute_angle(rk[i], rf[i]) for i in range(N)])

    # 4. Savitzky-Golay smoothing filter
    window = min(11, N if N % 2 == 1 else N - 1)
    if window >= 5:
        hips_s = savgol_filter(hips, window, 2, axis=0)
        com_s = savgol_filter(com_total, window, 2, axis=0)
        q_hip_L_s = savgol_filter(q_hip_L, window, 2)
        q_knee_L_s = savgol_filter(q_knee_L, window, 2)
        q_hip_R_s = savgol_filter(q_hip_R, window, 2)
        q_knee_R_s = savgol_filter(q_knee_R, window, 2)
        lft_s = savgol_filter(lft, window, 2, axis=0)
        rft_s = savgol_filter(rft, window, 2, axis=0)
    else:
        hips_s, com_s = hips, com_total
        q_hip_L_s, q_knee_L_s = q_hip_L, q_knee_L
        q_hip_R_s, q_knee_R_s = q_hip_R, q_knee_R
        lft_s, rft_s = lft, rft

    # 5. calculate speed and acceleration
    def derivative(arr):
        return np.gradient(arr, dt, axis=0)

    v_hips = derivative(hips_s)
    v_com = derivative(com_s)
    vq_hip_L = derivative(q_hip_L_s)
    vq_knee_L = derivative(q_knee_L_s)
    vq_hip_R = derivative(q_hip_R_s)
    vq_knee_R = derivative(q_knee_R_s)

    aq_hip_L = derivative(vq_hip_L)
    aq_knee_L = derivative(vq_knee_L)
    aq_hip_R = derivative(vq_hip_R)
    aq_knee_R = derivative(vq_knee_R)

    # 6. extract features (a state vector for every frame)
    # list: Hips_Y, Hips_Z, Com_Y, Com_Z,
    # L_Knee_Height, R_Knee_Height, L_Foot_Height, R_Foot_Height,
    # q_hip_L, q_knee_L, q_hip_R, q_knee_R,
    # vq_hip_L, vq_knee_L, vq_hip_R, vq_knee_R,
    # aq_hip_L, aq_knee_L, aq_hip_R, aq_knee_R

    features = np.column_stack([
        hips_s[:, 1], hips_s[:, 2],
        com_s[:, 1], com_s[:, 2],
        lk[:, 1], rk[:, 1],
        lft_s[:, 1], rft_s[:, 1],
        q_hip_L_s, q_knee_L_s, q_hip_R_s, q_knee_R_s,
        vq_hip_L, vq_knee_L, vq_hip_R, vq_knee_R,
        aq_hip_L, aq_knee_L, aq_hip_R, aq_knee_R
    ])

    targets = np.zeros((N, 4))  # [tau_hip_L, tau_knee_L, tau_hip_R, tau_knee_R]

    for i in range(N):
        # left leg
        q_L = np.array([q_hip_L_s[i], q_knee_L_s[i]])
        qd_L = np.array([vq_hip_L[i], vq_knee_L[i]])
        qdd_L = np.array([aq_hip_L[i], aq_knee_L[i]])
        tau_L = compute_inverse_dynamics(q_L, qd_L, qdd_L)

        # right leg
        q_R = np.array([q_hip_R_s[i], q_knee_R_s[i]])
        qd_R = np.array([vq_hip_R[i], vq_knee_R[i]])
        qdd_R = np.array([aq_hip_R[i], aq_knee_R[i]])
        tau_R = compute_inverse_dynamics(q_R, qd_R, qdd_R)

        targets[i, 0] = tau_L[0]
        targets[i, 1] = tau_L[1]
        targets[i, 2] = tau_R[0]
        targets[i, 3] = tau_R[1]

    return features, targets, json_path.stem


if __name__ == "__main__":
    files = list(IN_DIR.rglob("*.json"))
    print(f"find {len(files)} files，starting to extract...")

    X_all, Y_all, names = [], [], []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, res in enumerate(executor.map(process_file, files)):
            if res is not None:
                feats, targets, name = res
                X_all.append(feats)
                Y_all.append(targets)
                names.append(name)
            if (i + 1) % 200 == 0:
                print(f"processed {i + 1}/{len(files)}")

    # 拼接所有数据
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    print(f"\nsize of feature matrix X : {X_all.shape} (sample number, feature dimension)")
    print(f"size of label matrix Y : {Y_all.shape} (sample number, targeted torque dimension)")

    np.savez_compressed(OUT_DIR / "leg_dataset.npz", X=X_all, Y=Y_all, names=names)
    print(f"saved to: {OUT_DIR / 'leg_dataset.npz'}")