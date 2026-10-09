"""
The nominal car: a two-axle single-track model with Pacejka (magic formula) combined-slip tires, longitudinal load
transfer as a first-order state, and a current-driven AWD driveline with a fixed front/rear torque split (open
differentials: each axle gets its share of the motor torque whatever its speed). It is the quadruped's GPU sim
analog: the model the sim stage trains in (JAX), the ILC Jacobians come from, and the trajectory optimization uses
(CasADi / IPOPT). The "real" cars are the f1tenth_gym fork's multi-body model (mb_car.py) with perturbations.

The parameters are the f1tenth_gym's F1TENTH set (mass, yaw inertia, axle distances, CoG height, friction 1.0489,
steering limits), ilc_f1tenth/params.py's driveline (wheel radius, gear ratio, motor constant), and the full-scale
MF tire's DIMENSIONLESS coefficients (shape, curvature, slip stiffness per unit load: the lateral force peaks at
~9 deg slip, as RC tires do; the Froude-scaled set in lifted_linear_tire_20261009/mb_sim peaks at ~35 deg).

State  x = [X, Y, psi, vx, vy, r, w_f, w_r, delta, dFz]      (body-frame velocities at the CoG; axle wheel speeds)
Action a = [delta_cmd, I_cmd]  (steering servo set point [rad], motor current [A]: torque mode, as a VESC in
           current control) -- the policy's normalized action is a / A_SCALE.
The servo: d delta / dt = sv_max tanh((delta_cmd - delta) / (tau_s sv_max)) (first order, rate limited).
"""
import math

import numpy as np

G = 9.81

NOMINAL = dict(
    m=3.74, Iz=0.04712, lf=0.15875, lr=0.17145, h=0.074,
    s_max=0.4189, sv_max=3.2, tau_s=0.04,
    R_w=0.051, I_w=2 * 0.9 * 0.1 * 0.043 ** 2,      # per axle: two wheels of params.py's 0.9 m_w r^2
    k_t=11.82 * 1.5 * 4 * 0.000726,                 # wheel torque per amp (gear ratio x 1.5 pole pairs lambda)
    I_max=45.0, T_se=0.5,                           # current limit (equivalent ~12 m/s^2), front torque share
    c_lt=20.0,                                      # load-transfer relaxation rate [1/s]
    mu_x=1.1739, mu_y=1.0489,                       # MF peak factors (gym F1TENTH mu; full-scale long./lat. ratio)
    C_x=1.6411, E_x=0.46403, K_x=22.303,            # MF longitudinal shape / curvature / slip stiffness per load
    C_y=1.3507, E_y=-0.0074722, K_y=21.92,          # MF lateral
    r_bx1=13.276, r_bx2=-13.778, r_cx1=1.2568,      # MF combined slip (full-scale)
    r_by1=7.1433, r_by2=9.1916, r_cy1=1.0719,
    c_roll=0.015,                                   # rolling resistance per unit load
)

NX, NA = 10, 2
A_SCALE = np.array([NOMINAL['s_max'], NOMINAL['I_max']])
STATE_NAMES = ['X', 'Y', 'psi', 'vx', 'vy', 'r', 'w_f', 'w_r', 'delta', 'dFz']


class _NP:
    sin, cos, tanh, sqrt, arctan, arctan2 = np.sin, np.cos, np.tanh, np.sqrt, np.arctan, np.arctan2

    @staticmethod
    def stack(xs):
        return np.stack(xs)


def jax_ops():
    import jax.numpy as jnp

    class _J:
        sin, cos, tanh, sqrt, arctan, arctan2 = jnp.sin, jnp.cos, jnp.tanh, jnp.sqrt, jnp.arctan, jnp.arctan2

        @staticmethod
        def stack(xs):
            return jnp.stack(xs)
    return _J


def casadi_ops():
    import casadi as ca

    class _C:
        sin, cos, tanh, sqrt, arctan, arctan2 = ca.sin, ca.cos, ca.tanh, ca.sqrt, ca.atan, ca.atan2

        @staticmethod
        def stack(xs):
            return ca.vertcat(*xs)
    return _C


def _mf(slip, Fz, mu, C, E, K, o):
    D = mu * Fz
    B = K / (C * mu)                               # K Fz / (C D): load-independent
    Bs = B * slip
    return D * o.sin(C * o.arctan(Bs - E * (Bs - o.arctan(Bs))))


def _g_x(kappa, alpha, p, o):                      # combined-slip weight on Fx (MF 5.2, no offsets)
    B = p['r_bx1'] * o.cos(o.arctan(p['r_bx2'] * kappa))
    return o.cos(p['r_cx1'] * o.arctan(B * alpha))


def _g_y(kappa, alpha, p, o):
    B = p['r_by1'] * o.cos(o.arctan(p['r_by2'] * alpha))
    return o.cos(p['r_cy1'] * o.arctan(B * kappa))


def tire_forces(x, p, o=_NP):
    """Axle forces in the wheel frames: Fx_f, Fy_f, Fx_r, Fy_r, and the loads."""
    vx, vy, r, w_f, w_r, delta, dFz = x[3], x[4], x[5], x[6], x[7], x[8], x[9]
    L = p['lf'] + p['lr']
    eps = 0.1
    vxd = o.sqrt(vx * vx + eps)
    vxf = vx * o.cos(delta) + (vy + p['lf'] * r) * o.sin(delta)
    vxfd = o.sqrt(vxf * vxf + eps)
    a_f = delta - o.arctan2(vy + p['lf'] * r, vxd)
    a_r = o.arctan2(p['lr'] * r - vy, vxd)
    k_f = (p['R_w'] * w_f - vxf) / vxfd
    k_r = (p['R_w'] * w_r - vx) / vxd
    Fz_f0, Fz_r0 = p['m'] * G * p['lr'] / L, p['m'] * G * p['lf'] / L
    floor_f, floor_r = 0.05 * Fz_f0, 0.05 * Fz_r0
    d = Fz_f0 - dFz - floor_f
    Fz_f = floor_f + 0.5 * (d + o.sqrt(d * d + floor_f * floor_f))
    d = Fz_r0 + dFz - floor_r
    Fz_r = floor_r + 0.5 * (d + o.sqrt(d * d + floor_r * floor_r))
    out = []
    for k, a, Fz in ((k_f, a_f, Fz_f), (k_r, a_r, Fz_r)):
        Fx0 = _mf(k, Fz, p['mu_x'], p['C_x'], p['E_x'], p['K_x'], o)
        Fy0 = _mf(a, Fz, p['mu_y'], p['C_y'], p['E_y'], p['K_y'], o)
        out += [Fx0 * _g_x(k, a, p, o), Fy0 * _g_y(k, a, p, o)]
    return out + [Fz_f, Fz_r]


def dynamics(x, a, p=NOMINAL, o=_NP):
    """dx/dt for the action a = [delta_cmd, I_cmd] (clipped to the servo / current limits)."""
    psi, vx, vy, r, delta = x[2], x[3], x[4], x[5], x[8]
    Fxf, Fyf, Fxr, Fyr, Fz_f, Fz_r = tire_forces(x, p, o)
    L = p['lf'] + p['lr']
    cd, sd = o.cos(delta), o.sin(delta)
    roll = p['c_roll'] * (Fz_f + Fz_r) * o.tanh(vx / 0.2)
    Fx_cg = Fxr + Fxf * cd - Fyf * sd - roll
    dc = p['s_max'] * o.tanh(a[0] / p['s_max'])                  # smooth clip of the set point
    I = p['I_max'] * o.tanh(a[1] / p['I_max'])
    T = p['k_t'] * I
    sv = p['sv_max'] * o.tanh((dc - delta) / (p['tau_s'] * p['sv_max']))
    return o.stack([
        vx * o.cos(psi) - vy * o.sin(psi),
        vx * o.sin(psi) + vy * o.cos(psi),
        r,
        Fx_cg / p['m'] + vy * r,
        (Fyr + Fyf * cd + Fxf * sd) / p['m'] - vx * r,
        (p['lf'] * (Fyf * cd + Fxf * sd) - p['lr'] * Fyr) / p['Iz'],
        (p['T_se'] * T - p['R_w'] * Fxf) / p['I_w'],
        ((1 - p['T_se']) * T - p['R_w'] * Fxr) / p['I_w'],
        sv,
        -p['c_lt'] * (x[9] - p['h'] / L * Fx_cg),
    ])


def steady_state_residual(z, R, beta, p=NOMINAL, o=_NP):
    """Drift / cornering equilibrium on a circle of radius R (left turn for R > 0) at sideslip beta:
    unknowns z = [V, w_f, w_r, delta, I, dFz]; returns the 6 nonzero state derivatives (vx, vy, r, w_f, w_r, dFz)
    at delta_cmd = delta."""
    V, w_f, w_r, delta, I, dFz = (z[i] for i in range(6))
    x = [0, 0, 0, V * o.cos(beta), V * o.sin(beta), V / R, w_f, w_r, delta, dFz]
    f = dynamics(x, [delta, I], p, o)
    return o.stack([f[3], f[4], f[5], f[6], f[7], f[9]])
