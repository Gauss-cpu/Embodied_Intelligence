import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from mpl_toolkits.mplot3d import Axes3D

# ==================== 参数 ====================
g = 9.81
L1, L2, L3 = 1.0, 0.8, 0.15
m1, m2, m3 = 1.0, 0.8, 0.1
I_long = 0.1

# ==================== 旋转矩阵 ====================
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# ==================== 关节旋转 ====================
def R_shoulder(q):
    """肩：绕世界Z偏航 → 绕世界Y抬升 → 绕上臂长轴自转"""
    return rot_z(q[0]) @ rot_y(-q[1]) @ rot_z(q[2])

def R_elbow(q):
    """肘：肩基础上 → 绕上臂局部X屈伸 → 绕前臂长轴自转"""
    return R_shoulder(q) @ rot_x(q[3]) @ rot_z(q[4])

def R_wrist(q):
    """腕：肘基础上 → 绕前臂局部X屈伸 → 绕Y桡尺偏 → 绕手长轴自转"""
    return R_elbow(q) @ rot_x(q[5]) @ rot_y(q[6]) @ rot_z(q[7])

# ==================== 前向运动学 ====================
def forward_kinematics(q):
    Rs = R_shoulder(q)
    Re = R_elbow(q)
    Rw = R_wrist(q)

    local_down = np.array([0.0, 0.0, -1.0])
    upper_dir = Rs @ local_down
    fore_dir = Re @ local_down
    hand_dir = Rw @ local_down

    p_shoulder = np.zeros(3)
    p_elbow = p_shoulder + L1 * upper_dir
    p_wrist = p_elbow + L2 * fore_dir
    p_end = p_wrist + L3 * hand_dir

    p_com1 = p_shoulder + 0.5 * L1 * upper_dir
    p_com2 = p_elbow + 0.5 * L2 * fore_dir
    p_com3 = p_wrist + 0.5 * L3 * hand_dir

    return p_com1, p_com2, p_com3, p_elbow, p_wrist, p_end

# ==================== 数值雅可比 ====================
def jacobian(f, q, eps=1e-6):
    q = np.asarray(q, dtype=float)
    n = len(q)
    f0 = np.asarray(f(q), dtype=float)
    J = np.zeros((len(f0), n))
    for i in range(n):
        qp = q.copy()
        qp[i] += eps
        J[:, i] = (f(qp) - f0) / eps
    return J

# ==================== 关节轴 ====================
def joint_axes_world(q):
    Rs = R_shoulder(q)
    Re = R_elbow(q)
    Rw = R_wrist(q)

    axis_0 = np.array([0.0, 0.0, 1.0])
    axis_1 = rot_z(q[0]) @ np.array([0, -1.0, 0])
    axis_2 = Rs @ np.array([0, 0, 1.0])
    axis_3 = Rs @ np.array([1.0, 0, 0])
    axis_4 = Re @ np.array([0, 0, 1.0])
    axis_5 = Re @ np.array([1.0, 0, 0])
    axis_6 = Re @ rot_x(q[5]) @ np.array([0, 1.0, 0])
    axis_7 = Rw @ np.array([0, 0, 1.0])

    return [axis_0, axis_1, axis_2, axis_3, axis_4, axis_5, axis_6, axis_7]

def angular_jacobian(q, link):
    axes = joint_axes_world(q)
    if link == "upper":
        cols = axes[:3] + [np.zeros(3)] * 5
    elif link == "forearm":
        cols = axes[:5] + [np.zeros(3)] * 3
    elif link == "hand":
        cols = axes
    else:
        raise ValueError(link)
    return np.column_stack(cols)

# ==================== 惯量 ====================
def inertia_world(q, link):
    if link == "upper":
        R = R_shoulder(q); m, L = m1, L1
    elif link == "forearm":
        R = R_elbow(q); m, L = m2, L2
    elif link == "hand":
        R = R_wrist(q); m, L = m3, L3
    else:
        raise ValueError(link)

    I_local = np.diag([m*L**2/12, m*L**2/12, I_long])
    return R @ I_local @ R.T

# ==================== 质量矩阵 ====================
def mass_matrix(q):
    p1_fn = lambda qq: forward_kinematics(qq)[0]
    p2_fn = lambda qq: forward_kinematics(qq)[1]
    p3_fn = lambda qq: forward_kinematics(qq)[2]

    Jv1 = jacobian(p1_fn, q)
    Jv2 = jacobian(p2_fn, q)
    Jv3 = jacobian(p3_fn, q)
    Jw1 = angular_jacobian(q, "upper")
    Jw2 = angular_jacobian(q, "forearm")
    Jw3 = angular_jacobian(q, "hand")

    I1_w = inertia_world(q, "upper")
    I2_w = inertia_world(q, "forearm")
    I3_w = inertia_world(q, "hand")

    M = (m1 * Jv1.T @ Jv1 + m2 * Jv2.T @ Jv2 + m3 * Jv3.T @ Jv3
         + Jw1.T @ I1_w @ Jw1 + Jw2.T @ I2_w @ Jw2 + Jw3.T @ I3_w @ Jw3)
    return M

# ==================== 重力 ====================
def gravity_term(q):
    def p1_fn(qq): return forward_kinematics(qq)[0]
    def p2_fn(qq): return forward_kinematics(qq)[1]
    def p3_fn(qq): return forward_kinematics(qq)[2]
    Jv1 = jacobian(p1_fn, q)
    Jv2 = jacobian(p2_fn, q)
    Jv3 = jacobian(p3_fn, q)
    return m1*g*Jv1[2,:] + m2*g*Jv2[2,:] + m3*g*Jv3[2,:]

# ==================== 动力学 ====================
def dynamics(t, y, tau_fn):
    q = y[:8]
    qd = y[8:]

    if not np.all(np.isfinite(y)):
        return np.zeros(16)

    M = mass_matrix(q)
    C = np.zeros(8)                # C 项关闭（慢且精度影响小）
    G = gravity_term(q)
    tau = tau_fn(t, q, qd)

    try:
        qdd = np.linalg.solve(M, tau - C - G)
    except np.linalg.LinAlgError:
        qdd = np.linalg.lstsq(M, tau - C - G, rcond=None)[0]

    if not np.all(np.isfinite(qdd)):
        qdd = np.zeros(8)

    qdd = np.clip(qdd, -100.0, 100.0)
    return np.concatenate([qd, qdd])

# ==================== 控制律：全放开 ====================
def tau_fn(t, q, qd):
    """
    所有 8 个关节全部自由驱动。
    每个关节：独立正弦 + 极轻弹簧（回到 0，防止跑飞）+ 极轻阻尼。
    不做任何"必须向前弯"之类的限制——那是 brain 层的事。
    """
    tau = np.zeros(8)

    # 每个关节独立的正弦频率和幅度，让所有关节都"活"起来
    # 幅度小一点，避免爆炸；频率错开，避免同步成一根死板
    amps     = [0.3, 2.0, 0.2, 1.0, 0.2, 0.5, 0.3, 0.2]
    freqs    = [0.3, 0.4, 0.5, 0.6, 0.5, 0.7, 0.8, 0.6]
    phases   = [0.0, 0.0, 0.5, 1.0, 0.3, 1.5, 0.7, 0.9]

    for i in range(8):
        drive = amps[i] * np.sin(2*np.pi*freqs[i]*t + phases[i])
        # 极轻回位弹簧 + 极轻阻尼（只防跑飞，不算"限制"）
        spring = -1.0 * q[i]
        damp = -0.3 * qd[i]
        tau[i] = drive + spring + damp

    return tau

# ==================== 初始状态 ====================
y = np.zeros(16)
y[:8] = [0, math.radians(20), 0, math.radians(30), 0, 0, 0, 0]

# ==================== 仿真 ====================
dt = 0.005
T_sim = 10.0
history = [y.copy()]
t_hist = [0.0]

for i in range(int(T_sim / dt)):
    t = i * dt
    k1 = dynamics(t,          y,             tau_fn)
    k2 = dynamics(t + dt/2,   y + dt/2 * k1, tau_fn)
    k3 = dynamics(t + dt/2,   y + dt/2 * k2, tau_fn)
    k4 = dynamics(t + dt,     y + dt * k3,   tau_fn)
    y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    if not np.all(np.isfinite(y)):
        print(f"NaN at step {i}, t={t:.3f}")
        break

    history.append(y.copy())
    t_hist.append(t + dt)

history = np.array(history)

# ==================== 统计 ====================
q_hist = history[:, :8]
names = ["q1肩偏航","q2肩抬升","q3上臂自转","q4肘屈","q5前臂自转","q6腕屈","q7腕偏","q8手自转"]
for j, name in enumerate(names):
    sig = np.degrees(q_hist[:, j])
    jitter = np.std(np.diff(sig))
    print(f"{name}: [{sig.min():7.2f}, {sig.max():7.2f}] jitter {jitter:.4f}")