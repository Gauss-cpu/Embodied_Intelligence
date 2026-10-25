import json
import numpy as np
import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import matplotlib

matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==================== 路径与设备 ====================
ROOT = Path.home() / "Desktop/Physical_Intelligence"
AMYGDALA_DIR = ROOT / "amygdala_layer/event2score"
DATASET = ROOT / "dataset_penn"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ==================== 1. 杏仁核模型 ====================
class ValenceNet(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


print("加载 Amygdala 模型...")
_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
_valence_model = ValenceNet().to(DEVICE)
_valence_model.load_state_dict(torch.load(AMYGDALA_DIR / "valence_model.pt", map_location=DEVICE))
_valence_model.eval()


def event_to_weight(event_text):
    emb = _encoder.encode([event_text], convert_to_tensor=True).to(DEVICE)
    with torch.no_grad():
        return float(_valence_model(emb).item())


# ==================== 2. 物理引擎核心 ====================
g = 9.81
L1, L2, L3 = 1.0, 0.8, 0.15
m1, m2, m3 = 1.0, 0.8, 0.1
I_long = 0.1


def rot_x(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R_shoulder(q): return rot_z(q[0]) @ rot_y(-q[1]) @ rot_z(q[2])


def R_elbow(q):    return R_shoulder(q) @ rot_x(q[3]) @ rot_z(q[4])


def R_wrist(q):    return R_elbow(q) @ rot_x(q[5]) @ rot_y(q[6]) @ rot_z(q[7])


def forward_kinematics(q):
    Rs = R_shoulder(q);
    Re = R_elbow(q);
    Rw = R_wrist(q)
    ld = np.array([0.0, 0.0, -1.0]);
    ud = Rs @ ld;
    fd = Re @ ld;
    hd = Rw @ ld
    ps = np.zeros(3);
    pe = ps + L1 * ud;
    pw = pe + L2 * fd;
    pend = pw + L3 * hd
    return ps + 0.5 * L1 * ud, pe + 0.5 * L2 * fd, pw + 0.5 * L3 * hd, pe, pw, pend


def jacobian(f, q, eps=1e-5):
    q = np.asarray(q, dtype=float);
    n = len(q)
    f0 = np.asarray(f(q), dtype=float);
    J = np.zeros((len(f0), n))
    for i in range(n):
        qp = q.copy();
        qp[i] += eps;
        qm = q.copy();
        qm[i] -= eps
        J[:, i] = (f(qp) - f(qm)) / (2 * eps)
    return J


def joint_axes_world(q):
    Rs = R_shoulder(q);
    Re = R_elbow(q);
    Rw = R_wrist(q)
    return [np.array([0.0, 0.0, 1.0]), rot_z(q[0]) @ np.array([0, -1.0, 0]),
            Rs @ np.array([0, 0, 1.0]), Rs @ np.array([1.0, 0, 0]),
            Re @ np.array([0, 0, 1.0]), Re @ np.array([1.0, 0, 0]),
            Re @ rot_x(q[5]) @ np.array([0, 1.0, 0]), Rw @ np.array([0, 0, 1.0])]


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
    Jw1 = angular_jacobian(q, "upper");
    Jw2 = angular_jacobian(q, "forearm");
    Jw3 = angular_jacobian(q, "hand")
    I1_w = inertia_world(q, "upper");
    I2_w = inertia_world(q, "forearm");
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


def sim_dynamics(t, y, tau_seq, q_target, qd_target, dt_sim):
    """物理引擎，采用强化 PID 追踪新目标"""
    q = y[:8];
    qd = y[8:]
    idx = min(max(0, int(round(t / dt_sim))), len(tau_seq) - 1)

    tau_ff = tau_seq[idx]
    q_tgt = q_target[idx];
    qd_tgt = qd_target[idx]

    M = mass_matrix(q);
    G = gravity_term(q)

    # 【核心改变】：加强 Kp/Kd，让机械臂像训练有素的肌肉一样死死咬住新目标
    Kp = np.array([400] * 8);
    Kd = np.array([40] * 8)
    tau_fb = Kp * (q_tgt - q) + Kd * (qd_tgt - qd)

    try:
        qdd = np.linalg.solve(M, tau_ff + tau_fb - G)
    except:
        qdd = np.zeros(8)
    return np.concatenate([qd, np.clip(qdd, -200, 200)])


# ==================== 3. 终极 Pipeline (玩法B) ====================
def run_pipeline(json_path, event_text):
    print(f"\n{'=' * 60}")
    print(f"🎬 动作轨迹: {Path(json_path).name}")
    print(f"🧠 感知事件: {event_text}")
    print(f"{'=' * 60}")

    with open(json_path) as f:
        data = json.load(f)
    traj = data["trajectory"]
    N = len(traj)
    dt_sim = 1.0 / 30.0

    # 1. 提取原始轨迹
    q_initial = np.array([f["q"] for f in traj])
    q_initial_smooth = q_initial.copy()
    if N >= 7:
        for j in range(8): q_initial_smooth[:, j] = savgol_filter(q_initial[:, j], 7, 2)

    # 2. 杏仁核计算情绪
    valence = event_to_weight(event_text)

    # 3. 【大脑意图层】：生成情绪目标轨迹（放大/缩小）
    # 为了不狂飙，轨迹增益系数设定小一点，比如 0.3
    trajectory_gain = 1.0 + 0.3 * valence
    q_emotional_target = q_initial_smooth.copy()
    q_emotional_target[:, 1] *= trajectory_gain  # 放大肩部目标
    q_emotional_target[:, 3] *= trajectory_gain  # 放大肘部目标

    print(f"[1/4] 杏仁核输出: valence = {valence:+.3f}")
    print(f"      目标轨迹增益: {trajectory_gain:.2f}x")

    # 4. 【肌肉发力层】：根据新目标重新计算前馈力矩 (摆 final)
    qd_target = np.gradient(q_emotional_target, axis=0) / dt_sim
    qdd_target = np.gradient(qd_target, axis=0) / dt_sim
    qdd_target = np.clip(qdd_target, -100, 100)

    tau_emotional_ff = np.zeros((N, 8))
    for i in range(N):
        M = mass_matrix(q_emotional_target[i])
        G = gravity_term(q_emotional_target[i])
        tau_emotional_ff[i] = M @ qdd_target[i] + G

    # 稍微加点“情绪爆表”的力矩基数（比如人激动时基础张力变高）
    torque_boost = 1.0 + 0.15 * abs(valence)
    tau_emotional_ff *= torque_boost
    print(f"[2/4] 重新生成情绪力矩 (摆 final), 增益: {torque_boost:.2f}x")

    # 5. 【物理执行层】：把力矩喂给引擎，去咬住新目标
    y = np.zeros(16);
    y[:8] = q_emotional_target[0]
    q_sim = []
    print(f"[3/4] 物理引擎（强化反馈版）推演中...")
    for i in range(N):
        t = i * dt_sim
        k1 = sim_dynamics(t, y, tau_emotional_ff, q_emotional_target, qd_target, dt_sim)
        k2 = sim_dynamics(t + dt_sim / 2, y + dt_sim / 2 * k1, tau_emotional_ff, q_emotional_target, qd_target, dt_sim)
        k3 = sim_dynamics(t + dt_sim / 2, y + dt_sim / 2 * k2, tau_emotional_ff, q_emotional_target, qd_target, dt_sim)
        k4 = sim_dynamics(t + dt_sim, y + dt_sim * k3, tau_emotional_ff, q_emotional_target, qd_target, dt_sim)
        y = y + dt_sim / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q_sim.append(y[:8].copy())
    q_sim = np.array(q_sim)

    print(f"[4/4] 推演完成！")

    # ==================== 可视化对比 ====================
    t_arr = [f["t"] for f in traj]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 图1：轨迹对比
    axes[0].plot(t_arr, np.degrees(q_initial_smooth[:, 1]), 'b-', label="原始轨迹 (视频)", lw=2)
    axes[0].plot(t_arr, np.degrees(q_emotional_target[:, 1]), 'g--', label=f"情绪目标轨迹 ({trajectory_gain:.2f}x)",
                 lw=2)
    axes[0].plot(t_arr, np.degrees(q_sim[:, 1]), 'r:', label="物理推演轨迹 (咬住目标)", lw=2.5)
    axes[0].set_ylabel("肩部 q2 (°)")
    axes[0].legend();
    axes[0].grid(True, alpha=0.3)

    # 图2：力矩对比
    axes[1].plot(t_arr, tau_emotional_ff[:, 1], 'r-', label="情绪下发力矩 τ2", lw=2)
    axes[1].set_ylabel("肩部力矩 τ2 (N·m)")
    axes[1].legend();
    axes[1].grid(True, alpha=0.3)

    # 图3：误差分析
    err = np.degrees(np.abs(q_sim[:, 1] - q_emotional_target[:, 1]))
    axes[2].plot(t_arr, err, 'k-', label="物理引擎追踪误差", lw=2)
    axes[2].set_ylabel("误差 (°)")
    axes[2].set_xlabel("时间 (s)")
    axes[2].legend();
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"玩法B：基于目标意图与强化控制的物理情绪推演\n事件: {event_text} (valence={valence:+.2f})",
                 fontsize=14)
    plt.tight_layout()

    save_name = f"emotion_result_B_{abs(hash(event_text)) % 10000}.png"
    plt.savefig(save_name, dpi=120)
    print(f"已保存对比图: {save_name}")
    plt.show()


if __name__ == "__main__":
    dataset = Path.home() / "Desktop/Physical_Intelligence/dataset_penn"
    default_q = dataset / "penn_baseball_pitch/video_0020_q.json"

    json_path = str(default_q)
    event_text = "今天拿了全班第一，很高兴"

    if len(sys.argv) >= 3:
        json_path = sys.argv[1]
        event_text = sys.argv[2]

    run_pipeline(json_path, event_text)