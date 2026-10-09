"""
The "real" cars: the f1tenth_gym fork's multi-body model (s-hliao/f1tenth_gym dev-humble, vehicle_dynamics_mb:
sprung body with roll/pitch, four unsprung wheels with their own spin, suspension, compliant joints, Pacejka MF
tires per corner with the full-scale tire's offsets) at F1TENTH scale, integrated by RK4 at 0.25 ms, behind the
same actuation as the nominal model (car_model.py): a steering servo (first order, rate limited, optional
transport delay) and a current-controlled motor whose torque is split front/rear (RWD by default: the drift
setup) -- the gym's accel input carries the torque (T = m R_w u1), its speed / accel clamps are lifted.

Perturbed variants (REAL_CARS) differ from the nominal only here; the sim the policy trains in never sees them.
All numerics in numba: one call flies a whole multi-lap run with the network in the loop (policy.py's numpy MLP
and obs_np), which is the analog of the quadruped's CPU MuJoCo robots.
"""
import math
import os
import sys
from dataclasses import fields

import numba
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lifted_linear_tire_20261009', 'mb_sim'))
from fork_import import (vehicle_dynamics_mb, init_mb, mb_vector, mb_index,           # noqa: E402
                         FULLSCALE_VEHICLE_PARAMETERS as FS)
from mb_params import F1TENTH_MB_PARAMETERS                                           # noqa: E402
import car_model as cm                                                                # noqa: E402

SUBSTEP = 2.5e-4
NXM = 29
_f_mb = numba.njit(cache=True)(vehicle_dynamics_mb)


def mb_params(mass=1.0, mu=1.0, ky=1.0, T_se=0.0):
    """The MB parameter vector: the F1TENTH-scale set of mb_params.py with the full-scale MF slip stiffnesses
    (dimensionless per unit load, as car_model.NOMINAL), the drive split T_se (also the motor-braking split), the
    gym's accel / speed clamps lifted (the motor current limit governs), and the perturbations."""
    P = F1TENTH_MB_PARAMETERS.with_updates(
        tire_p_ky1=FS.tire_p_ky1 * ky, tire_p_kx1=FS.tire_p_kx1,
        tire_p_dy1=F1TENTH_MB_PARAMETERS.tire_p_dy1 * mu, tire_p_dx1=FS.tire_p_dx1 * mu,
        T_se=T_se, T_sb=T_se, a_max=200.0, v_switch=200.0, v_max=50.0, v_min=-5.0,
        m_s=F1TENTH_MB_PARAMETERS.m_s * mass + (mass - 1) * (F1TENTH_MB_PARAMETERS.m_uf + F1TENTH_MB_PARAMETERS.m_ur),
        I_z_body=F1TENTH_MB_PARAMETERS.I_z_body * mass, I_Phi_s=F1TENTH_MB_PARAMETERS.I_Phi_s * mass,
        I_y_s=F1TENTH_MB_PARAMETERS.I_y_s * mass, m=F1TENTH_MB_PARAMETERS.m * mass)
    p = mb_vector(P)
    return p, P


# variants: (name, MB perturbations, actuation: k_t scale, steering offset rad, servo tau s, steering delay steps,
# current delay steps), sensing noise (pos m, yaw rad, vel m/s, rate rad/s)
REAL_CARS = {
    'real_nom': dict(mb={}, kt=1.0, d_off=0.0, tau=0.04, d_delay=0, i_delay=0),
    'real_mass': dict(mb=dict(mass=1.25), kt=1.0, d_off=0.0, tau=0.04, d_delay=0, i_delay=0),
    'real_mu': dict(mb=dict(mu=0.8), kt=1.0, d_off=0.0, tau=0.04, d_delay=0, i_delay=0),
    'real_act': dict(mb={}, kt=0.8, d_off=0.03, tau=0.07, d_delay=0, i_delay=1),
    'real_lag': dict(mb=dict(ky=0.8), kt=1.0, d_off=0.0, tau=0.05, d_delay=1, i_delay=0),
}
NOISE = np.array([0.003, 0.003, 0.005, 0.03, 0.03, 0.03, 0.5, 0.5])  # X Y psi vx vy r w_f w_r (sensing)


def car_config(name, T_se=0.0):
    c = REAL_CARS[name]
    p, P = mb_params(T_se=T_se, **c['mb'])
    return dict(name=name, p=p, P=P, kt=c['kt'] * cm.NOMINAL['k_t'], d_off=c['d_off'], tau=c['tau'],
                d_delay=c['d_delay'], i_delay=c['i_delay'], mass=P.m)


@numba.njit(cache=True)
def _deriv(x, sv, accel, p):
    u = np.empty(2)
    u[0] = sv
    u[1] = accel
    return _f_mb(x.copy(), u, p)


@numba.njit(cache=True)
def mb_step(x, d_cmd, i_cmd, p, kt, tau, dt, R_w, mass, I_max, sv_max, s_max):
    """One control period with the commands held: servo + motor current -> MB, RK4 at SUBSTEP."""
    n = int(round(dt / SUBSTEP))
    h = dt / n
    I = I_max * math.tanh(i_cmd / I_max)
    dc = s_max * math.tanh(d_cmd / s_max)
    accel = kt * I / (mass * R_w)
    for _ in range(n):
        sv = sv_max * math.tanh((dc - x[2]) / (tau * sv_max))
        k1 = _deriv(x, sv, accel, p)
        x2 = x + 0.5 * h * k1
        sv = sv_max * math.tanh((dc - x2[2]) / (tau * sv_max))
        k2 = _deriv(x2, sv, accel, p)
        x3 = x + 0.5 * h * k2
        sv = sv_max * math.tanh((dc - x3[2]) / (tau * sv_max))
        k3 = _deriv(x3, sv, accel, p)
        x4 = x + h * k3
        sv = sv_max * math.tanh((dc - x4[2]) / (tau * sv_max))
        k4 = _deriv(x4, sv, accel, p)
        x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        for i in range(23, 27):
            if x[i] < 0.0:
                x[i] = 0.0
    return x


def planar(xm):
    """MB state -> the nominal model's state [X, Y, psi, vx, vy, r, w_f, w_r, delta, dFz] (dFz not measured: 0)."""
    return np.array([xm[0], xm[1], xm[4], xm[3], xm[10], xm[5], 0.5 * (xm[23] + xm[24]),
                     0.5 * (xm[25] + xm[26]), xm[2], 0.0])


def mb_from_planar(xs, p):
    """An MB state at the planar state xs (suspension at its static equilibrium from init_mb, wheels at xs's axle
    speeds)."""
    x0 = np.array([xs[0], xs[1], xs[8], math.hypot(xs[3], xs[4]), xs[2], xs[5], math.atan2(xs[4], xs[3])])
    xm = init_mb(x0, p)
    xm[23] = xm[24] = xs[6]
    xm[25] = xm[26] = xs[7]
    return xm
