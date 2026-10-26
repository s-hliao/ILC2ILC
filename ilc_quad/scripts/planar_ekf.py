"""
The planar trunk EKF (2026-10-26): the policy's state estimate from the IMU and the mocap, at a fixed rate
(default 100 Hz). One implementation for the CPU robots and the hardware (dilc_execute.EkfEstimator, numpy) and the
GPU training env (ilc_mjx jump.py, jax.numpy): every function takes the array module `xp` and touches no Python
state, so both run exactly the same filter.

State s (8,), in the jump's planar frame (x forward along the jump, z up; the trunk's base origin, as the mocap
reports it):
    x, z, th, vx, vz, bax, baz, bg
th is the nose-up pitch (ilc_jump_sim.planar_pitch); bax, baz the accelerometer's bias (body x, z), bg the gyro's
(body y). The IMU (body frame, at the base origin): gyro w_y, th_dot = -(w_y - bg); specific force f = (fx, fz),
the world acceleration a = R(th) (f - ba) + g with R(th) = [[c, -s], [s, c]] (a nose-up pitch turns body x
toward +z).

Predict: every 1 / rate s, with the mean of the IMU samples since the last step. Correct: each mocap frame
(x, z, th), taken latency s before it arrived, against the state at its own time (the current step's, moved back
by the velocity times its age: a first-order delayed measurement).
"""
from __future__ import annotations

DEFAULTS = dict(
    rate=100.0,          # Hz: the filter's steps
    sig_p=0.0005,        # m: mocap position noise per frame
    sig_th=0.003,        # rad: mocap pitch noise per frame
    q_acc=0.5,           # m/s^2: velocity process noise (IMU noise + what the model misses), per step as q_acc dt
    q_gyro=0.02,         # rad/s: pitch process noise, per step as q_gyro dt
    q_ba=0.002,          # m/s^2 per step: accelerometer bias random walk
    q_bg=0.0002,         # rad/s per step: gyro bias random walk
    p0_pos=1e-6, p0_th=1e-5, p0_vel=1e-4, p0_ba=0.01, p0_bg=1e-4,   # initial variances (a robot standing still)
)
G = 9.81
N = 8


def params(cfg: dict | None = None) -> dict:
    p = dict(DEFAULTS)
    p.update(cfg or {})
    return p


def init(x, z, th, xp, cfg=None):
    """A robot at rest at (x, z, th): the state and its covariance."""
    p = params(cfg)
    s = xp.stack([x, z, th] + [x * 0.0] * 5) if hasattr(x, "shape") and getattr(x, "ndim", 0) > 0 else \
        xp.asarray([x, z, th, 0.0, 0.0, 0.0, 0.0, 0.0])
    P = xp.diag(xp.asarray([p["p0_pos"], p["p0_pos"], p["p0_th"], p["p0_vel"], p["p0_vel"],
                            p["p0_ba"], p["p0_ba"], p["p0_bg"]]))
    return s, P


def predict(s, P, w_y, f, dt, xp, cfg=None):
    """One step of dt with the IMU's mean gyro w_y and specific force f = (fx, fz) over it."""
    p = params(cfg)
    th, vx, vz, bax, baz, bg = s[2], s[3], s[4], s[5], s[6], s[7]
    c, sn = xp.cos(th), xp.sin(th)
    fx, fz = f[0] - bax, f[1] - baz
    ax, az = c * fx - sn * fz, sn * fx + c * fz - G
    thd = -(w_y - bg)
    s_new = xp.stack([s[0] + vx * dt + 0.5 * ax * dt * dt, s[1] + vz * dt + 0.5 * az * dt * dt, th + thd * dt,
                      vx + ax * dt, vz + az * dt, bax, baz, bg])
    # d a / d th, d a / d (bax, baz)
    dax_th, daz_th = -sn * fx - c * fz, c * fx - sn * fz
    z0 = th * 0.0
    one = z0 + 1.0
    F = xp.stack([
        xp.stack([one, z0, 0.5 * dt * dt * dax_th, dt * one, z0, -0.5 * dt * dt * c, 0.5 * dt * dt * sn, z0]),
        xp.stack([z0, one, 0.5 * dt * dt * daz_th, z0, dt * one, -0.5 * dt * dt * sn, -0.5 * dt * dt * c, z0]),
        xp.stack([z0, z0, one, z0, z0, z0, z0, dt * one]),
        xp.stack([z0, z0, dt * dax_th, one, z0, -dt * c, dt * sn, z0]),
        xp.stack([z0, z0, dt * daz_th, z0, one, -dt * sn, -dt * c, z0]),
        xp.stack([z0, z0, z0, z0, z0, one, z0, z0]),
        xp.stack([z0, z0, z0, z0, z0, z0, one, z0]),
        xp.stack([z0, z0, z0, z0, z0, z0, z0, one]),
    ])
    qv, qt = (p["q_acc"] * dt) ** 2, (p["q_gyro"] * dt) ** 2
    Q = xp.diag(xp.asarray([0.25 * qv * dt * dt, 0.25 * qv * dt * dt, qt, qv, qv,
                            p["q_ba"] ** 2, p["q_ba"] ** 2, p["q_bg"] ** 2]))
    return s_new, F @ P @ F.T + Q


def update(s, P, meas, age, w_y, xp, cfg=None, valid=None):
    """One mocap frame meas = (x, z, th), taken age s before the state's time; w_y the latest gyro. valid (bool,
    jax): False leaves (s, P) as they are (a fixed-size loop over frames that may not exist)."""
    p = params(cfg)
    thd = -(w_y - s[7])
    h = xp.stack([s[0] - s[3] * age, s[1] - s[4] * age, s[2] - thd * age])
    z0 = age * 0.0
    one = z0 + 1.0
    H = xp.stack([
        xp.stack([one, z0, z0, -age * one, z0, z0, z0, z0]),
        xp.stack([z0, one, z0, z0, -age * one, z0, z0, z0]),
        xp.stack([z0, z0, one, z0, z0, z0, z0, -age * one]),
    ])
    R = xp.diag(xp.asarray([p["sig_p"] ** 2, p["sig_p"] ** 2, p["sig_th"] ** 2]))
    S = H @ P @ H.T + R
    K = P @ H.T @ xp.linalg.inv(S)
    s_new = s + K @ (xp.asarray(meas) - h)
    I_KH = xp.eye(N) - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T            # Joseph form
    if valid is None:
        return s_new, P_new
    return xp.where(valid, s_new, s), xp.where(valid, P_new, P)


def extrapolate(s, w_y, dt, xp):
    """The trunk (x, z, th, vx, vz, th_dot) dt after the state's time, at constant velocity (between steps)."""
    thd = -(w_y - s[7])
    return xp.stack([s[0] + s[3] * dt, s[1] + s[4] * dt, s[2] + thd * dt, s[3], s[4], thd])
