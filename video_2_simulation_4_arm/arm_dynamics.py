import numpy as np
import math

# physical hyperparameters
g = 9.81
L1, L2, L3 = 1.0, 0.8, 0.15
m1, m2, m3 = 1.0, 0.8, 0.1
I_long = 0.1

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
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

# through the rotation and length to calculate the coordinates of wrist, elbow, and shoulder
def forward_kinematics(q):
    Rs = R_shoulder(q); Re = R_elbow(q); Rw = R_wrist(q)
    ld = np.array([0.0, 0.0, -1.0])
    ud = Rs @ ld; fd = Re @ ld; hd = Rw @ ld
    ps = np.zeros(3)
    pe = ps + L1 * ud
    pw = pe + L2 * fd
    pend = pw + L3 * hd
    return ps + 0.5*L1*ud, pe + 0.5*L2*fd, pw + 0.5*L3*hd, pe, pw, pend

# Numerical differentiation: partial derivative matrix
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

# calculate whom is the bone circling
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

# rotation's contribution to the angular velocity of the bones
def angular_jacobian(q, link):
    axes = joint_axes_world(q)
    if link == "upper":     cols = axes[:3] + [np.zeros(3)] * 5
    elif link == "forearm": cols = axes[:5] + [np.zeros(3)] * 3
    elif link == "hand":    cols = axes
    return np.column_stack(cols)

# moment of inertia relative the world
def inertia_world(q, link):
    if link == "upper":     R = R_shoulder(q); m, L = m1, L1
    elif link == "forearm": R = R_elbow(q); m, L = m2, L2
    elif link == "hand":    R = R_wrist(q); m, L = m3, L3
    I_local = np.diag([m * L ** 2 / 12, m * L ** 2 / 12, I_long])
    return R @ I_local @ R.T

# the three terms of lagrange equation
# mass matrix is similar to kinetic energy, the formula of mass matrix is derived from kinetic energy formula
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

# gravitational potential energy
def gravity_term(q):
    p1_fn = lambda qq: forward_kinematics(qq)[0]
    p2_fn = lambda qq: forward_kinematics(qq)[1]
    p3_fn = lambda qq: forward_kinematics(qq)[2]
    Jv1 = jacobian(p1_fn, q); Jv2 = jacobian(p2_fn, q); Jv3 = jacobian(p3_fn, q)
    return m1 * g * Jv1[2, :] + m2 * g * Jv2[2, :] + m3 * g * Jv3[2, :]

def coriolis_term(q, qd):
    eps = 1e-5
    c = np.zeros(8)
    for i in range(8):
        qp = q.copy(); qp[i] += eps
        qm = q.copy(); qm[i] -= eps
        Mp = mass_matrix(qp)
        Mm = mass_matrix(qm)
        dM_dq = (Mp - Mm) / (2 * eps)
        c += np.dot(dM_dq, qd) * qd[i] - 0.5 * np.dot(qd, np.dot(dM_dq, qd))
    return c

def compute_inverse_dynamics(q, qd, qdd):
    M = mass_matrix(q)
    G = gravity_term(q)
    C = coriolis_term(q, qd)
    return M @ qdd + C + G


def step_dynamics(q, qd, tau, dt=0.01, friction=0.0):
    M = mass_matrix(q)
    G = gravity_term(q)
    C = coriolis_term(q, qd)

    rhs = tau - C - G - friction * qd
    try:
        qdd = np.linalg.solve(M, rhs)
        qdd = np.clip(qdd, -100.0, 100.0)
    except np.linalg.LinAlgError:
        qdd = np.zeros(8)

    qd_new = qd + qdd * dt
    qd_new = np.clip(qd_new, -15.0, 15.0)

    q_new = q + qd_new * dt
    return q_new, qd_new, qdd