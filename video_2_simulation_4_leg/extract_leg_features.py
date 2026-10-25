import json
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor
import os

IN_DIR = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/processed_legs_3d"
OUT_DIR = Path.home() / "Desktop/Physical_Intelligence/dataset_cmu/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 质量假设
m_thigh = 10.0  # 大腿
m_shank = 4.0  # 小腿
m_total = 2 * (m_thigh + m_shank) + 30.0  # 加上上半身粗略估计 30kg


def compute_angle(p1, p2):
    """计算两点之间相对于垂直向下（Y轴向下）的角度"""
    # 在 CMU 坐标系中，通常是 Y向上，Z向前。我们计算矢状面（Z-Y平面）的角度
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    # 返回极角，限制在 -pi 到 pi
    return np.arctan2(dz, -dy)


def process_file(json_path):
    with open(json_path) as f:
        data = json.load(f)

    traj = data["trajectory"]
    fps = data["fps"]
    dt = 1.0 / fps
    N = len(traj)
    if N < 10: return None

    # 提取原始 3D 坐标 (N, 3)
    hips = np.array([f["Hips"] for f in traj])
    lk = np.array([f["LeftUpLeg"] for f in traj])
    lf = np.array([f["LeftLeg"] for f in traj])
    lft = np.array([f["LeftFoot"] for f in traj])

    rk = np.array([f["RightUpLeg"] for f in traj])
    rf = np.array([f["RightLeg"] for f in traj])
    rft = np.array([f["RightFoot"] for f in traj])

    # 1. 计算大腿和小腿的质心 (CoM) 位置
    com_thigh_L = (hips + lk) / 2
    com_shank_L = (lk + lf) / 2
    com_thigh_R = (hips + rk) / 2
    com_shank_R = (rk + rf) / 2

    # 2. 计算全身质心 (简化为只算腿部+骨盆)
    com_total = (com_thigh_L * m_thigh + com_shank_L * m_shank +
                 com_thigh_R * m_thigh + com_shank_R * m_shank + hips * 30.0) / m_total

    # 3. 计算关节角度 (矢状面)
    q_hip_L = np.array([compute_angle(hips[i], lk[i]) for i in range(N)])
    q_knee_L = np.array([compute_angle(lk[i], lf[i]) for i in range(N)])
    q_hip_R = np.array([compute_angle(hips[i], rk[i]) for i in range(N)])
    q_knee_R = np.array([compute_angle(rk[i], rf[i]) for i in range(N)])

    # 4. Savitzky-Golay 平滑滤波（非常重要，消除微分的毛刺）
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

    # 5. 一阶导（速度）和二阶导（加速度）
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

    # 6. 提取高级特征 (每一帧组合成一个状态向量)
    # 特征列表: Hips_Y, Hips_Z, Com_Y, Com_Z,
    # L_Knee_Height, R_Knee_Height, L_Foot_Height, R_Foot_Height,
    # q_hip_L, q_knee_L, q_hip_R, q_knee_R,
    # vq_hip_L, vq_knee_L, vq_hip_R, vq_knee_R,
    # aq_hip_L, aq_knee_L, aq_hip_R, aq_knee_R

    features = np.column_stack([
        hips_s[:, 1], hips_s[:, 2],  # 骨盆 Y(高), Z(前)
        com_s[:, 1], com_s[:, 2],  # 重心 Y, Z
        lk[:, 1], rk[:, 1],  # 左右膝高度
        lft_s[:, 1], rft_s[:, 1],  # 左右脚高度 (接触特征)
        q_hip_L_s, q_knee_L_s, q_hip_R_s, q_knee_R_s,
        vq_hip_L, vq_knee_L, vq_hip_R, vq_knee_R,
        aq_hip_L, aq_knee_L, aq_hip_R, aq_knee_R
    ])

    # =========================================================
    # 【伪标签】逆动力学 Torque
    # 这里为了简化，用基础的 M(q)*qdd + G(q) 来算出目标 Torque
    # 后续可以直接用物理引擎追踪
    # =========================================================
    # 粗略惯量 (仅作为监督学习的伪标签目标)
    # tau = I * alpha + m * g * r
    # 针对膝和髋进行简化计算
    tau_hip_L = 15.0 * aq_hip_L + 30.0 * np.sin(q_hip_L_s)
    tau_knee_L = 8.0 * aq_knee_L + 15.0 * np.sin(q_knee_L_s)
    tau_hip_R = 15.0 * aq_hip_R + 30.0 * np.sin(q_hip_R_s)
    tau_knee_R = 8.0 * aq_knee_R + 15.0 * np.sin(q_knee_R_s)

    targets = np.column_stack([tau_hip_L, tau_knee_L, tau_hip_R, tau_knee_R])

    return features, targets, json_path.stem


if __name__ == "__main__":
    files = list(IN_DIR.rglob("*.json"))
    print(f"找到 {len(files)} 个文件，开始提取超级特征与力矩伪标签...")

    X_all, Y_all, names = [], [], []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, res in enumerate(executor.map(process_file, files)):
            if res is not None:
                feats, targets, name = res
                X_all.append(feats)
                Y_all.append(targets)
                names.append(name)
            if (i + 1) % 200 == 0:
                print(f"已处理 {i + 1}/{len(files)}")

    # 拼接所有数据
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    print(f"\n特征矩阵 X 大小: {X_all.shape} (样本数, 特征维度)")
    print(f"标签矩阵 Y 大小: {Y_all.shape} (样本数, 目标力矩维度)")

    np.savez_compressed(OUT_DIR / "leg_dataset.npz", X=X_all, Y=Y_all, names=names)
    print(f"已保存训练集至: {OUT_DIR / 'leg_dataset.npz'}")