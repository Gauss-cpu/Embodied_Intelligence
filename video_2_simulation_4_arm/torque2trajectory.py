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

# ==================== 动力学参数 ====================
g = 9.81
L1, L2, L3 = 1.0, 0.8, 0.15
m1, m2, m3 = 1.0, 0.8, 0.1
I_long = 0.1


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R_shoulder(q): return rot_z(q[0]) @ rot_y(-q[1]) @ rot_z(q[2])


def R_elbow(q):    return R_shoulder(q) @ rot_x(q[3]) @ rot_z(q[4])


def R_wrist(q):    return R_elbow(q) @ rot_x(q[5]) @ rot_y(q[6]) @ rot_z(q[7])


def forward_kinematics(q):
    Rs = R_shoulder(q);
    Re = R_elbow(q);
    Rw = R_wrist(q)
    ld = np.array([0.0, 0.0, -1.0])
    ud = Rs @ ld;
    fd = Re @ ld;
    hd = Rw @ ld
    ps = np.zeros(3)
    pe = ps + L1 * ud
    pw = pe + L2 * fd
    pend = pw + L3 * hd
    return ps + 0.5 * L1 * ud, pe + 0.5 * L2 * fd, pw + 0.5 * L3 * hd, pe, pw, pend


def jacobian(f, q, eps=1e-5):
    q = np.asarray(q, dtype=float)
    n = len(q)
    f0 = np.asarray(f(q), dtype=float)
    J = np.zeros((len(f0), n))
    for i in range(n):
        qp = q.copy();
        qp[i] += eps
        qm = q.copy();
        qm[i] -= eps
        J[:, i] = (f(qp) - f(qm)) / (2 * eps)
    return J


def joint_axes_world(q):
    Rs = R_shoulder(q);
    Re = R_elbow(q);
    Rw = R_wrist(q)
    return [
        np.array([0.0, 0.0, 1.0]),
        rot_z(q[0]) @ np.array([0, -1.0, 0]),
        Rs @ np.array([0, 0, 1.0]),
        Rs @ np.array([1.0, 0, 0]),
        Re @ np.array([0, 0, 1.0]),
        Re @ np.array([1.0, 0, 0]),
        Re @ rot_x(q[5]) @ np.array([0, 1.0, 0]),
        Rw @ np.array([0, 0, 1.0]),
    ]


def angular_jacobian(q, link):
    axes = joint_axes_world(q)
    if link == "upper":
        cols = axes[:3] + [np.zeros(3)] * 5
    elif link == "forearm":
        cols = axes[:5] + [np.zeros(3)] * 3
    elif link == "hand":
        cols = axes
    return np.column_stack(cols)


def inertia_world(q, link):
    if link == "upper":
        R = R_shoulder(q); m, L = m1, L1
    elif link == "forearm":
        R = R_elbow(q); m, L = m2, L2
    elif link == "hand":
        R = R_wrist(q); m, L = m3, L3
    I_local = np.diag([m * L ** 2 / 12, m * L ** 2 / 12, I_long])
    return R @ I_local @ R.T


def mass_matrix(q):
    p1_fn = lambda qq: forward_kinematics(qq)[0]
    p2_fn = lambda qq: forward_kinematics(qq)[1]
    p3_fn = lambda qq: forward_kinematics(qq)[2]
    Jv1 = jacobian(p1_fn, q);
    Jv2 = jacobian(p2_fn, q);
    Jv3 = jacobian(p3_fn, q)
    Jw1 = angular_jacobian(q, "upper")
    Jw2 = angular_jacobian(q, "forearm")
    Jw3 = angular_jacobian(q, "hand")
    I1_w = inertia_world(q, "upper")
    I2_w = inertia_world(q, "forearm")
    I3_w = inertia_world(q, "hand")
    return (m1 * Jv1.T @ Jv1 + m2 * Jv2.T @ Jv2 + m3 * Jv3.T @ Jv3
            + Jw1.T @ I1_w @ Jw1 + Jw2.T @ I2_w @ Jw2 + Jw3.T @ I3_w @ Jw3)


def gravity_term(q):
    p1_fn = lambda qq: forward_kinematics(qq)[0]
    p2_fn = lambda qq: forward_kinematics(qq)[1]
    p3_fn = lambda qq: forward_kinematics(qq)[2]
    Jv1 = jacobian(p1_fn, q);
    Jv2 = jacobian(p2_fn, q);
    Jv3 = jacobian(p3_fn, q)
    return m1 * g * Jv1[2, :] + m2 * g * Jv2[2, :] + m3 * g * Jv3[2, :]


# ==================== 物理引擎：正向积分 ====================
def dynamics(t, y, tau_seq, q_target, qd_target, dt_sim):
    """给定状态 y 和当前时刻力矩 tau，求导数"""
    q = y[:8]
    qd = y[8:]

    idx = int(round(t / dt_sim))
    idx = min(max(0, idx), len(q_target) - 1)

    tau_ff = tau_seq[idx]
    q_tgt = q_target[idx]
    qd_tgt = qd_target[idx]

    M = mass_matrix(q)
    G = gravity_term(q)

    # 【关键】：加强反馈力量
    # 之前 Kp=20 太弱了，这里用 Kp=100, Kd=10
    Kp = np.array([500, 500, 500, 500, 500, 500, 500, 500])
    Kd = np.array([50, 50, 50, 50, 50, 50, 50, 50])

    tau_fb = Kp * (q_tgt - q) + Kd * (qd_tgt - qd)
    tau_total = tau_ff + tau_fb
    friction = - qd * 2.0
    qdd = np.linalg.solve(M, tau_total + friction - G)

    try:
        qdd = np.linalg.solve(M, tau_total - G)
    except np.linalg.LinAlgError:
        qdd = np.zeros(8)

    qdd = np.clip(qdd, -200, 200)
    return np.concatenate([qd, qdd])


def simulate_from_torque(q_path, tau_path, save_path="sim_result.png"):
    with open(q_path) as f:
        data_q = json.load(f)
    with open(tau_path) as f:
        data_tau = json.load(f)

    traj = data_q["trajectory"]
    tau_ff = np.array(data_tau["tau"])

    dt_sim = 1.0 / 30.0
    N = min(len(traj), len(tau_ff))

    # 提取平滑的目标轨迹，用于给 PD 控制器当锚点
    q_raw = np.array([f["q"] for f in traj[:N]])
    q_target = q_raw.copy()
    if N >= 7:
        for j in range(8):
            q_target[:, j] = savgol_filter(q_raw[:, j], 7, 2)
    qd_target = np.gradient(q_target, axis=0) / dt_sim

    # 初始状态
    y = np.zeros(16)
    y[:8] = q_target[0]

    q_sim = []
    t_sim = []

    print(f"动作: {data_q.get('category')}, {N} 帧")
    print("物理引擎推演中 (前馈 + 反馈)...")

    for i in range(N):
        t = i * dt_sim
        k1 = dynamics(t, y, tau_ff, q_target, qd_target, dt_sim)
        k2 = dynamics(t + dt_sim / 2, y + dt_sim / 2 * k1, tau_ff, q_target, qd_target, dt_sim)
        k3 = dynamics(t + dt_sim / 2, y + dt_sim / 2 * k2, tau_ff, q_target, qd_target, dt_sim)
        k4 = dynamics(t + dt_sim, y + dt_sim * k3, tau_ff, q_target, qd_target, dt_sim)
        y = y + dt_sim / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        q_sim.append(y[:8].copy())
        t_sim.append(t)

    q_sim = np.array(q_sim)

    # 对比图
    t = [f["t"] for f in traj[:N]]
    q2_orig = np.degrees([f["q"][1] for f in traj[:N]])
    q4_orig = np.degrees([f["q"][3] for f in traj[:N]])
    q2_sim = np.degrees(q_sim[:, 1])
    q4_sim = np.degrees(q_sim[:, 3])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, q2_orig, 'b-', label="视频原始 q2", lw=2)
    axes[0].plot(t_sim, q2_sim, 'r--', label="Torque 推演 q2", lw=2)
    axes[0].set_ylabel("q2 (°)")
    axes[0].legend();
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, q4_orig, 'b-', label="视频原始 q4", lw=2)
    axes[1].plot(t_sim, q4_sim, 'r--', label="Torque 推演 q4", lw=2)
    axes[1].set_ylabel("q4 (°)")
    axes[1].set_xlabel("时间 (s)")
    axes[1].legend();
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"已保存对比图: {save_path}")
    plt.show()


if __name__ == "__main__":
    import sys

    dataset = Path.home() / "Desktop/Physical_Intelligence/dataset_penn"
    default_q = dataset / "penn_baseball_pitch/video_0020_q.json"
    q_path = sys.argv[1] if len(sys.argv) >= 3 else str(default_q)
    tau_path = sys.argv[2] if len(sys.argv) >= 3 else "tau_result.json"

    simulate_from_torque(q_path, tau_path)