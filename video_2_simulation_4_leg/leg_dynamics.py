import numpy as np
import math

# ==================== 物理参数 ====================
g = 9.81
l1, l2 = 0.45, 0.45          # 大腿、小腿长度 (m)
m1, m2 = 10.0, 4.0           # 大腿、小腿质量 (kg)
lc1, lc2 = l1/2, l2/2        # 质心到关节的距离 (m)
I1 = (1/12) * m1 * l1**2     # 大腿转动惯量
I2 = (1/12) * m2 * l2**2     # 小腿转动惯量

def mass_matrix(q1, q2):
    M11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * math.cos(q2)) + I1 + I2
    M12 = m2 * (lc2**2 + l1 * lc2 * math.cos(q2)) + I2
    return np.array([[M11, M12], [M12, m2 * lc2**2 + I2]])

def coriolis_matrix(q1, q2, qd1, qd2):
    h = m2 * l1 * lc2 * math.sin(q2)
    return np.array([[-h * qd2, -h * (qd1 + qd2)], [h * qd1, 0.0]])

def gravity_term(q1, q2):
    G1 = (m1 * lc1 + m2 * l1) * g * math.sin(q1) + m2 * lc2 * g * math.sin(q1 + q2)
    G2 = m2 * lc2 * g * math.sin(q1 + q2)
    return np.array([G1, G2])

def compute_inverse_dynamics(q, qd, qdd):
    """逆动力学：给定运动轨迹，算出基准力矩 (Optimal Torque)"""
    M = mass_matrix(q[0], q[1])
    C = coriolis_matrix(q[0], q[1], qd[0], qd[1])
    G = gravity_term(q[0], q[1])
    return M @ qdd + C @ qd + G

def step_dynamics(q, qd, tau, dt=0.01, friction=0.0):
    """正向动力学：给定力矩，推演下一步的物理状态"""
    M = mass_matrix(q[0], q[1])
    C = coriolis_matrix(q[0], q[1], qd[0], qd[1])
    G = gravity_term(q[0], q[1])
    rhs = tau - C @ qd - G - friction * qd
    try:
        qdd = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        qdd = np.zeros(2)
    qd_new = qd + qdd * dt
    q_new = q + qd_new * dt
    return q_new, qd_new, qdd