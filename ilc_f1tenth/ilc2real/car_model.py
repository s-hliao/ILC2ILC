"""
The nominal car: LLA-MPC's Fiala model (LLA-MPC-online llampc/nmpc_gen_fiala_fixed.export_model, exact=False, the
model the lab's NMPC runs on the car), term for term:
  single-track, Fiala/brush tires with combined slip sigma = sqrt(tan(alpha)^2 + kappa^2 + 1e-2) on each axle,
  peak mu Fz (static axle loads, no load transfer), ONE wheel-speed state omega_w driving BOTH axles (locked AWD:
  kappa_f and kappa_r from the same shaft), motor torque gear_ratio 1.5 pole_pairs lambda I on that shaft, against
  (Iw + Im). Parameters: llampc params.F110 (mass 4.6 kg, Iz, lf, lr, rw, Iw, Im, pole_pairs = poles / 2 = 2) and
  the NMPC's mean tire model (llampc_fiala_fixed.fiala_setup: Cf 250, Cr 225 N/rad, muf = mur = 0.6, Cro 0).
It is the quadruped's GPU-sim analog: the sim stage trains in it (JAX), the ILC Jacobians come from it, the
trajectory optimization uses it (CasADi / IPOPT). The "real" cars are the f1tenth_gym fork's multi-body model
calibrated to this car (mb_car.py).

State  x = [X, Y, psi, vx, vy, r, omega_w, I, delta]   (LLA-MPC's layout: wheel speed, motor current, steering)
Action a = [delta_cmd, I_cmd]  set points (the network's output; LLA-MPC plans the rates instead), clamped smoothly
to the ranges (sat: the identity inside them). The actuators:
  steering servo  d delta / dt = sv_max tanh((delta_cmd - delta) / (tau_s sv_max)), delta in +-0.34 rad
  current loop    d I / dt     = slew  tanh((I_cmd - I) / (tau_I slew)), slew 300 A/s, I in [-25, 50] A
(the NMPC's bounds: steer rate 3.2 rad/s, current slew 300 A/s, current -25..50 A, steering +-0.34 rad)
"""
import os

import numpy as np

G = 9.81

NOMINAL = dict(
    m=4.6, Iz=0.04712, lf=0.15875, lr=0.17145, h=0.074,
    rw=0.051, Iw=0.9 * (4 * 0.1) * 0.043 ** 2, Im=11.82 ** 2 * 5.5e-6,
    kt=11.82 * 1.5 * 2 * 0.000726,                  # gear_ratio * 1.5 * pole_pairs * lambda [N m / A]
    Cf=250.0, Cr=225.0, muf=0.6, mur=0.6, Cro=0.0,              # muf / mur: overridden by $F1T_MU (e.g. plastic tires)
    s_max=0.34, sv_max=3.2, tau_s=0.04,
    I_min=-25.0, I_max=50.0, slew=300.0, tau_I=0.02,
)

if os.environ.get('F1T_MU'):
    NOMINAL['muf'] = NOMINAL['mur'] = float(os.environ['F1T_MU'])

NX, NA = 9, 2
STATE_NAMES = ['X', 'Y', 'psi', 'vx', 'vy', 'r', 'omega_w', 'I', 'delta']
I_MID = 0.5 * (NOMINAL['I_max'] + NOMINAL['I_min'])
I_HALF = 0.5 * (NOMINAL['I_max'] - NOMINAL['I_min'])
# normalized action a_n = (a - A_OFF) / A_SCALE
A_SCALE = np.array([NOMINAL['s_max'], I_HALF])
A_OFF = np.array([0.0, I_MID])


class _NP:
    sin, cos, tan, tanh, sqrt, arctan, arctan2 = np.sin, np.cos, np.tan, np.tanh, np.sqrt, np.arctan, np.arctan2

    @staticmethod
    def where(c, a, b):
        return np.where(c, a, b)

    @staticmethod
    def softplus(z):
        return np.logaddexp(0.0, z)

    @staticmethod
    def stack(xs):
        return np.stack(xs)


def jax_ops():
    import jax.numpy as jnp

    class _J:
        sin, cos, tan, tanh, sqrt, arctan, arctan2 = jnp.sin, jnp.cos, jnp.tan, jnp.tanh, jnp.sqrt, jnp.arctan, jnp.arctan2

        @staticmethod
        def where(c, a, b):
            return jnp.where(c, a, b)

        @staticmethod
        def softplus(z):
            return jnp.logaddexp(0.0, z)

        @staticmethod
        def stack(xs):
            return jnp.stack(xs)
    return _J


def casadi_ops():
    import casadi as ca

    class _C:
        sin, cos, tan, tanh, sqrt, arctan, arctan2 = ca.sin, ca.cos, ca.tan, ca.tanh, ca.sqrt, ca.atan, ca.atan2

        @staticmethod
        def where(c, a, b):
            return ca.if_else(c, a, b)

        @staticmethod
        def softplus(z):
            return ca.if_else(z > 30, z, ca.log1p(ca.exp(z)))

        @staticmethod
        def stack(xs):
            return ca.vertcat(*xs)
    return _C


def sat(x, lo, hi, k, o=_NP):
    """Smooth clamp to [lo, hi]: the identity inside (to within ~1/k of the limits), smooth at them."""
    return x - o.softplus(k * (x - hi)) / k + o.softplus(k * (lo - x)) / k


SAT_K = (60.0, 2.0)                 # sharpness of the steering [1/rad] and current [1/A] clamps


def tire_forces(vx, vy, r, w, delta, p, o=_NP):
    """LLA-MPC's Fiala forces (exact=False): Ffx, Ffy (front wheel frame), Frx, Fry, and the combined slips."""
    eps = 0.1
    L = p['lf'] + p['lr']
    vxd = o.sqrt(vx * vx + eps)
    Ffz, Frz = p['m'] * G * p['lr'] / L, p['m'] * G * p['lf'] / L
    Ffmax, Frmax = p['muf'] * Ffz, p['mur'] * Frz
    af = delta - o.arctan2(r * p['lf'] + vy, vxd)
    ar = o.arctan2(r * p['lr'] - vy, vxd)
    vxf = vx * o.cos(delta) + (vy + p['lf'] * r) * o.sin(delta)
    vxfd = o.sqrt(vxf * vxf + eps)
    kr = (p['rw'] * w - vx) / vxd
    kf = (p['rw'] * w - vxf) / vxfd
    sf = o.sqrt(o.tan(af) ** 2 + kf ** 2 + 1e-2)
    sr = o.sqrt(o.tan(ar) ** 2 + kr ** 2 + 1e-2)

    def brush(C, s, F):
        Cs = C * s
        return Cs - Cs ** 2 / (3 * F) + Cs ** 3 / (27 * F ** 2)
    Ft = o.where(sf < 3 * Ffmax / p['Cf'], brush(p['Cf'], sf, Ffmax), Ffmax)
    Rt = o.where(sr < 3 * Frmax / p['Cr'], brush(p['Cr'], sr, Frmax), Frmax)
    Ffy = Ft * o.tan(af) / sf
    Ffx = Ft * kf / sf
    F_roll = p['Cro'] * Frz * o.tanh(vx / 0.5)
    Frx = Rt * kr / sr - F_roll
    Fry = Rt * o.tan(ar) / sr
    return Ffx, Ffy, Frx, Fry


def dynamics(x, a, p=NOMINAL, o=_NP):
    """dx/dt for the set points a = [delta_cmd, I_cmd] (saturated smoothly to the actuator ranges)."""
    psi, vx, vy, r, w, I, delta = x[2], x[3], x[4], x[5], x[6], x[7], x[8]
    Ffx, Ffy, Frx, Fry = tire_forces(vx, vy, r, w, delta, p, o)
    cd, sd = o.cos(delta), o.sin(delta)
    dc = sat(a[0], -p['s_max'], p['s_max'], SAT_K[0], o)
    Ic = sat(a[1], p['I_min'], p['I_max'], SAT_K[1], o)
    tau = p['kt'] * I
    return o.stack([
        vx * o.cos(psi) - vy * o.sin(psi),
        vx * o.sin(psi) + vy * o.cos(psi),
        r,
        (Frx + Ffx * cd - Ffy * sd) / p['m'] + vy * r,
        (Fry + Ffy * cd + Ffx * sd) / p['m'] - vx * r,
        (p['lf'] * (Ffy * cd + Ffx * sd) - p['lr'] * Fry) / p['Iz'],
        (tau - Frx * p['rw'] - Ffx * p['rw']) / (p['Iw'] + p['Im']),
        p['slew'] * o.tanh((Ic - I) / (p['tau_I'] * p['slew'])),
        p['sv_max'] * o.tanh((dc - delta) / (p['tau_s'] * p['sv_max'])),
    ])


def steady_state_residual(z, R, beta, p=NOMINAL, o=_NP):
    """Cornering / drift equilibrium on a circle of radius R at sideslip beta: z = [V, omega_w, delta, I]."""
    V, w, delta, I = (z[i] for i in range(4))
    x = [0, 0, 0, V * o.cos(beta), V * o.sin(beta), V / R, w, I, delta]
    f = dynamics(x, [delta, I], p, o)
    return o.stack([f[3], f[4], f[5], f[6]])
