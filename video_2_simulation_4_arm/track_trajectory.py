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

# ==================== arm3d 动力学（精简） ====================
g = 9.81
L1, L2, L3 = 1.0, 0.8, 0.15
m1, m2, m3 = 1.0, 0.8, 0.1
I_long = 0.1

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def R_shoulder(q): return rot_z(q[0]) @ rot_y(-q[1]) @ rot_z(q[2])
def R_elbow(q):    return R_shoulder(q) @ rot_x(q[3]) @ rot_z(q[4])
def R_wrist(q):    return R_elbow(q) @ rot_x(q[5]) @ rot_y(q[6]) @ rot_z(q[7])

def forward_kinematics(q):
    Rs = R_shoulder(q); Re = R_elbow(q); Rw = R_wrist(q)
    ld = np.array([0.0, 0.0, -1.0])
    ud = Rs @ ld; fd = Re @ ld; hd = Rw @ ld
    ps = np.zeros(3)
    pe = ps + L1 * ud
    pw = pe + L2 * fd
    pend = pw + L3 * hd
    return ps + 0.5*L1*ud, pe + 0.5*L2*fd, pw + 0.5*L3*hd, pe, pw, pend

def jacobian(f, q, eps=1e-5):
    q = np.asarray(q, dtype=float)
    n = len(q)
    f0 = np.asarray(f(q), dtype=float)
    J = np.zeros((len(f0), n))
    for i in range(n):
        qp = q.copy(); qp[i] += eps
        qm = q.copy(); qm[i] -= eps
        J[:, i] = (f(qp) - f(qm)) / (2 * eps)
    return J

def joint_axes_world(q):
    Rs = R_shoulder(q); Re = R_elbow(q); Rw = R_wrist(q)
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
    if link == "upper":     cols = axes[:3] + [np.zeros(3)] * 5
    elif link == "forearm": cols = axes[:5] + [np.zeros(3)] * 3
    elif link == "hand":    cols = axes
    return np.column_stack(cols)

def inertia_world(q, link):
    if link == "upper":     R = R_shoulder(q); m, L = m1, L1
    elif link == "forearm": R = R_elbow(q); m, L = m2, L2
    elif link == "hand":    R = R_wrist(q); m, L = m3, L3
    I_local = np.diag([m*L**2/12, m*L**2/12, I_long])
    return R @ I_local @ R.T

def mass_matrix(q):
    p1_fn = lambda qq: forward_kinematics(qq)[0]
    p2_fn = lambda qq: forward_kinematics(qq)[1]
    p3_fn = lambda qq: forward_kinematics(qq)[2]
    Jv1 = jacobian(p1_fn, q); Jv2 = jacobian(p2_fn, q); Jv3 = jacobian(p3_fn, q)
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
    Jv1 = jacobian(p1_fn, q); Jv2 = jacobian(p2_fn, q); Jv3 = jacobian(p3_fn, q)
    return m1*g*Jv1[2,:] + m2*g*Jv2[2,:] + m3*g*Jv3[2,:]


# ==================== 主逻辑：逆动力学 ====================
def analyze_trajectory(json_path, save_path="tracking_result.png"):
    with open(json_path) as f:
        data = json.load(f)
    traj = data["trajectory"]
    cat = data.get("category", "?")
    N = len(traj)
    dt = 1.0 / 30.0

    q_desired = np.array([qf["q"] for qf in traj])   # [N, 8]
    t_array = np.arange(N) * dt

    # ---- 平滑 ----
    q_smooth = q_desired.copy()
    if N >= 7:
        for j in range(8):
            q_smooth[:, j] = savgol_filter(q_desired[:, j], 7, 2)
    elif N >= 5:
        for j in range(8):
            q_smooth[:, j] = savgol_filter(q_desired[:, j], 5, 2)
    else:
        q_smooth = q_desired.copy()

    # ---- 求导（只用平滑后的）----
    qd = np.gradient(q_smooth, axis=0) / dt          # [N, 8]
    qdd = np.gradient(qd, axis=0) / dt

    # ---- clip 加速度 ----
    qdd = np.clip(qdd, -100, 100)

    # ---- 锁定不可信的关节：q1, q3, q5, q8（索引 0,2,4,7）----
    for i in [0, 2, 4, 7]:
        q_smooth[:, i] = 0.0
        qd[:, i] = 0.0
        qdd[:, i] = 0.0

    # ---- 逐帧算 τ ----
    print(f"轨迹: {cat}, {N} 帧")
    print("逐帧算逆动力学...")

    tau_all = np.zeros((N, 8))
    G_all = np.zeros((N, 8))
    M_diag_all = np.zeros((N, 8))

    for i in range(N):
        q = q_smooth[i]
        M = mass_matrix(q)
        G = gravity_term(q)
        tau = M @ qdd[i] + G
        tau_all[i] = tau
        G_all[i] = G
        M_diag_all[i] = np.diag(M)

        if i % 20 == 0:
            print(f"  {i}/{N}")

    print("完成")

    # ---- 统计 ----
    tau_norm = np.linalg.norm(tau_all, axis=1)
    print(f"\nτ 范数: 均值 {tau_norm.mean():.2f}, 最大 {tau_norm.max():.2f}")
    print(f"M 对角均值: {M_diag_all.mean(axis=0)}")
    print(f"\n各关节 τ 范围：")
    for j in range(8):
        print(f"  τ{j+1}: [{tau_all[:, j].min():+.2f}, {tau_all[:, j].max():+.2f}]  "
              f"std {tau_all[:, j].std():.3f}")

    # ---- 画图 ----
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_array, np.degrees(q_smooth[:, 1]), 'b-', label="q2", lw=2)
    axes[0].plot(t_array, np.degrees(q_smooth[:, 3]), 'r-', label="q4", lw=2)
    axes[0].set_ylabel("关节角 (°)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_array, np.degrees(qd[:, 1]), 'b-', label="q̇2", lw=2)
    axes[1].plot(t_array, np.degrees(qd[:, 3]), 'r-', label="q̇4", lw=2)
    axes[1].set_ylabel("关节速度 (°/s)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    for j in range(8):
        axes[2].plot(t_array, tau_all[:, j], label=f"τ{j+1}")
    axes[2].set_ylabel("关节力矩 τ (N·m)")
    axes[2].legend(ncol=4); axes[2].grid(True, alpha=0.3)

    for j in range(8):
        axes[3].plot(t_array, G_all[:, j], label=f"G{j+1}")
    axes[3].set_ylabel("重力项 G")
    axes[3].set_xlabel("时间 (s)")
    axes[3].legend(ncol=4); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\n已保存: {save_path}")
    plt.show()

    out_data = {
        "video_id": data.get("video_id", "?"),
        "category": data.get("category", "?"),
        "tau": tau_all.tolist()
    }
    with open("tau_result.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"已保存力矩数据: tau_result.json")
    
    return q_smooth, tau_all


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        dataset = Path.home() / "Desktop/Physical_Intelligence/dataset_penn"
        path = sorted((dataset / "penn_baseball_pitch").glob("*.json"))[0]

    analyze_trajectory(str(path))