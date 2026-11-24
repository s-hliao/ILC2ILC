"""Force-based Go1 trot in MJX, and its single-rigid-body (SRBD) twin for the ILC Jacobians.

Controller (the "policy" for the positivity test; a learned network replaces it later), every 20 ms control step:
  gait      trot, period T_GAIT, duty 0.5, diagonal pairs (FL+RR, FR+RL); a scheduled stance leg is "in contact"
  stance    a desired trunk wrench from PD on the trunk's position / velocity / orientation against a moving reference
            (straight line to the goal, trapezoid speed), distributed over the stance feet by regularized least squares,
            clipped into friction cones -> foot forces u (4, 3, world, ground reaction on the robot), + the ILC offsets du;
            applied every 2 ms tick as joint torques tau = -J^T R^T u (J: the leg's foot Jacobian, base frame)
  swing     Cartesian PD of the foot (base frame) to a Raibert foothold, lifted on a sine, every 2 ms tick
SRBD twin: the whole robot as one rigid body at the base (total mass, the home pose's inertia about the base), state
s = [p (3), euler zyx (3), v (3), omega_world (3)], inputs the stance-foot forces at the MEASURED foot positions (held over
the control step), Euler integration in 4 substeps. The controller's stance-force map is a function of s (with the feet,
contact flags and reference held), so its feedback K = d pi / d s comes by autodiff -- the closed loop A + B K.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from ilc_go1walk import kin

TICK = 0.002
SUB = 10                         # ticks per control step
DT = TICK * SUB                  # 20 ms
T_GAIT = 0.36
DUTY = 0.5
OFFSET = np.array([0.0, 0.5, 0.5, 0.0])     # FL, FR, RL, RR
Z_REF = 0.27
G = np.array([0.0, 0.0, -9.81])
MU_CTRL = 0.5
FZ_MAX = 180.0
SWING_H = 0.07
KP_SW, KD_SW = 800.0, 20.0
M_SWING = 1.2                    # kg: thigh + calf, gravity feedforward on the swing foot
GAINS = dict(kp_xy=40.0, kd_xy=10.0, kp_z=150.0, kd_z=20.0, kp_r=200.0, kd_r=20.0, kp_yaw=60.0, kd_yaw=10.0)
V_MAX, A_MAX = 0.5, 0.8
K_RAIB = 0.03                    # capture-point foothold gain on the velocity error (s)
SETTLE = 0.5                     # s of trotting in place before the reference moves


def quat_to_R(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def euler_to_R(e):
    r, p, y = e[0], e[1], e[2]
    cr, sr, cp, sp, cy, sy = jnp.cos(r), jnp.sin(r), jnp.cos(p), jnp.sin(p), jnp.cos(y), jnp.sin(y)
    return jnp.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]])


def R_to_euler(R):
    return jnp.array([jnp.arctan2(R[2, 1], R[2, 2]), -jnp.arcsin(jnp.clip(R[2, 0], -1, 1)), jnp.arctan2(R[1, 0], R[0, 0])])


def skew(v):
    return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def contact_flags(t):
    ph = jnp.mod(t / T_GAIT + jnp.asarray(OFFSET), 1.0)
    return (ph < DUTY) | (t < SETTLE * 0.4), ph      # all four down for the first 0.2 s


def reference(t, goal, yaw0):
    """Straight line from the origin to goal (2,), trapezoid speed, starting at SETTLE; returns p_ref (3), v_ref (3), yaw."""
    L = jnp.linalg.norm(goal) + 1e-9
    d = goal / L
    tau = jnp.maximum(t - SETTLE, 0.0)
    t_acc = V_MAX / A_MAX
    vpk = jnp.minimum(V_MAX, jnp.sqrt(L * A_MAX))
    ta = vpk / A_MAX
    tc = jnp.maximum(L - vpk * ta, 0.0) / vpk
    s = jnp.where(tau < ta, 0.5 * A_MAX * tau ** 2,
                  jnp.where(tau < ta + tc, 0.5 * vpk * ta + vpk * (tau - ta),
                            L - 0.5 * A_MAX * jnp.maximum(2 * ta + tc - tau, 0.0) ** 2))
    v = jnp.where(tau < ta, A_MAX * tau, jnp.where(tau < ta + tc, vpk, A_MAX * jnp.maximum(2 * ta + tc - tau, 0.0)))
    s = jnp.minimum(s, L)
    del t_acc
    return jnp.array([d[0] * s, d[1] * s, Z_REF]), jnp.array([d[0] * v, d[1] * v, 0.0]), yaw0


class Robot:
    """Mass / inertia of the nominal robot (home pose, about the base origin), for the controller and the SRBD."""

    def __init__(self, qm):
        m, d = qm.model, mujoco.MjData(qm.model)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[:3] = [0, 0, Z_REF]
        mujoco.mj_forward(m, d)
        b0 = qm.base_body_id
        pb, Rb = d.xpos[b0].copy(), d.xmat[b0].reshape(3, 3).copy()
        M, I = 0.0, np.zeros((3, 3))
        for b in range(1, m.nbody):
            mb = m.body_mass[b]
            if mb <= 0:
                continue
            Rib = d.ximat[b].reshape(3, 3)
            Ib = Rib @ np.diag(m.body_inertia[b]) @ Rib.T
            r = d.xipos[b] - pb
            I += Ib + mb * (r @ r * np.eye(3) - np.outer(r, r))
            M += mb
        self.mass = float(M)
        self.inertia = Rb.T @ I @ Rb                      # base frame, about the base origin


def controller(s, feet_w, c, t, goal, yaw0, mass, inertia, gains=GAINS):
    """Stance forces u (4, 3) (world, on the robot) from the SRBD state s, the feet (4, 3, world), contact flags c (4,)."""
    p, e, v, w = s[0:3], s[3:6], s[6:9], s[9:12]
    p_ref, v_ref, yaw_ref = reference(t, goal, yaw0)
    a = jnp.array([gains['kp_xy'] * (p_ref[0] - p[0]) + gains['kd_xy'] * (v_ref[0] - v[0]),
                   gains['kp_xy'] * (p_ref[1] - p[1]) + gains['kd_xy'] * (v_ref[1] - v[1]),
                   gains['kp_z'] * (p_ref[2] - p[2]) + gains['kd_z'] * (0.0 - v[2])])
    dyaw = jnp.arctan2(jnp.sin(yaw_ref - e[2]), jnp.cos(yaw_ref - e[2]))
    al = jnp.array([gains['kp_r'] * (0.0 - e[0]) - gains['kd_r'] * w[0],
                    gains['kp_r'] * (0.0 - e[1]) - gains['kd_r'] * w[1],
                    gains['kp_yaw'] * dyaw - gains['kd_yaw'] * w[2]])
    Rw = euler_to_R(e)
    Iw = Rw @ jnp.asarray(inertia) @ Rw.T
    W = jnp.concatenate([mass * (a - jnp.asarray(G)), Iw @ al])
    r = feet_w - p[None]
    A = jnp.concatenate([jnp.concatenate([jnp.eye(3)] * 4, 1),
                         jnp.concatenate([skew(r[i]) for i in range(4)], 1)], 0)       # (6, 12)
    mask = jnp.repeat(c.astype(jnp.float32), 3)
    Am = A * mask[None]
    f = Am.T @ jnp.linalg.solve(Am @ Am.T + 1e-3 * jnp.diag(jnp.array([1., 1., 1., .1, .1, .1])), W)
    f = f.reshape(4, 3)
    fz = jnp.clip(f[:, 2], 0.0, FZ_MAX)
    ft = f[:, :2]
    nt = jnp.sqrt(jnp.sum(ft * ft, axis=1) + 1e-6)          # smooth: the norm's gradient at 0 is NaN (swing legs)
    ft = ft * jnp.minimum(1.0, MU_CTRL * fz / nt)[:, None]
    return jnp.concatenate([ft, fz[:, None]], 1) * c[:, None]


def srbd_step(s, u, feet_w, mass, inertia, n_sub=4):
    """One control step of the SRBD under foot forces u (4, 3) at feet_w (held)."""
    h = DT / n_sub

    def f(s):
        p, e, v, w = s[0:3], s[3:6], s[6:9], s[9:12]
        Rw = euler_to_R(e)
        Iw = Rw @ jnp.asarray(inertia) @ Rw.T
        F = jnp.sum(u, 0)
        Tq = jnp.sum(jnp.cross(feet_w - p[None], u), 0)
        dv = F / mass + jnp.asarray(G)
        dw = jnp.linalg.solve(Iw, Tq - jnp.cross(w, Iw @ w))
        r, pi = e[0], e[1]
        Einv = jnp.array([[1.0, jnp.sin(r) * jnp.tan(pi), jnp.cos(r) * jnp.tan(pi)],
                          [0.0, jnp.cos(r), -jnp.sin(r)],
                          [0.0, jnp.sin(r) / jnp.cos(pi), jnp.cos(r) / jnp.cos(pi)]])
        wb = Rw.T @ w
        de = Einv @ wb
        return jnp.concatenate([v, de, dv, dw])

    for _ in range(n_sub):
        s = s + h * f(s)
    return s


class Sim:
    """The MJX Go1 under the controller; nominal or perturbed (payload kg on the trunk, torque scale, foot friction)."""

    def __init__(self, payload=0.0, torque_scale=1.0, friction=None, iterations=4, ls_iterations=8):
        import os
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        for p in ('../../ilc_mjx', '../../ilc_quad'):
            q = os.path.abspath(os.path.join(here, p))
            if q not in sys.path:
                sys.path.insert(0, q)
        from ilc_quad.sim_quad_model import QuadModel
        from ilc_mjx import MENAGERIE
        from ilc_mjx.model import mjx_model
        qm = QuadModel('go1', MENAGERIE, payload=dict(mass=payload) if payload > 0 else None)
        if friction is not None:
            for g in list(qm.foot_geom_ids) + list(qm.terrain_geom_ids):
                qm.model.geom_friction[g, 0] = friction
        self.qm = qm
        self.m, self.mx = mjx_model(qm, feet_only_contacts=True, iterations=iterations, ls_iterations=ls_iterations)
        self.qpos_adr = jnp.asarray(qm.qpos_adr)
        self.qvel_adr = jnp.asarray(qm.qvel_adr)
        self.act_idx = jnp.asarray(qm.act_idx)
        self.tau_lim = jnp.asarray(qm.torque_limit) * torque_scale
        self.torque_scale = torque_scale
        d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, d, 0)
        d.qpos[:3] = [0, 0, Z_REF + 0.01]
        mujoco.mj_forward(self.m, d)
        self.d0 = mjx.put_data(self.m, d)

    def measure(self, dx):
        """SRBD state s (12), base rotation R, feet (4, 3) world, joint q, qd."""
        q = dx.qpos[self.qpos_adr]
        qd = dx.qvel[self.qvel_adr]
        R = quat_to_R(dx.qpos[3:7])
        p = dx.qpos[0:3]
        s = jnp.concatenate([p, R_to_euler(R), dx.qvel[0:3], R @ dx.qvel[3:6]])
        feet_w = p[None] + (R @ kin.feet(q).T).T
        return s, R, feet_w, q, qd

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def rollout(self, goal, du, robot_mi, n_steps):
        """One walk to goal (2,) for n_steps control steps; du (n_steps, 4, 3) force offsets on the stance legs.
        Returns per control step: s (at the step's start), feet_w, c, u (applied), t; and the final s."""
        mass, inertia = robot_mi

        def ctrl_step(carry, k):
            dx, lift = carry
            t = k * DT
            s, R, feet_w, q, qd = self.measure(dx)
            c, ph = contact_flags(t + DT / 2)                 # the gait sampled mid-step, held over the step
            u = controller(s, feet_w, c, t, goal, 0.0, mass, inertia) + du[k] * c[:, None]
            # swing targets (Raibert) for this control step
            p_ref, v_ref, _ = reference(t, goal, 0.0)
            stance_b = jnp.asarray(kin.HIP) + jnp.stack([jnp.zeros(4), jnp.asarray(kin.ABD), jnp.zeros(4)], 1)   # under the thigh
            hips_w = s[0:3][None] + (R @ stance_b.T).T
            t_left = (1.0 - jnp.clip((ph - DUTY) / (1 - DUTY), 0, 1)) * (1 - DUTY) * T_GAIT
            land = hips_w[:, :2] + s[6:8][None] * (t_left + DUTY * T_GAIT / 2)[:, None] \
                + K_RAIB * (s[6:8] - v_ref[:2])[None]
            land = jnp.concatenate([land, jnp.full((4, 1), kin.FOOT_R)], 1)

            def tick(i, st):
                dx, lift = st
                ti = t + i * TICK
                _, phi = contact_flags(ti)
                ci = c                                            # stance / swing switch only at control steps
                _, Ri, fw, qi, qdi = self.measure(dx)
                J = kin.jacs(qi)
                sp = jnp.clip((phi - DUTY) / (1 - DUTY), 0.0, 1.0)
                lift = jnp.where(ci[:, None], fw, lift)          # remembers the foot at lift-off
                ss = sp * sp * (3 - 2 * sp)
                tgt = lift + (land - lift) * ss[:, None] + jnp.stack(
                    [jnp.zeros(4), jnp.zeros(4), SWING_H * jnp.sin(jnp.pi * sp)], 1)
                e_b = (Ri.T @ (tgt - fw).T).T
                vf = jnp.einsum('lij,lj->li', J, qdi.reshape(4, 3))
                F_sw = KP_SW * e_b - KD_SW * vf + Ri.T @ jnp.array([0.0, 0.0, M_SWING * 9.81])
                F_st = -(Ri.T @ u.T).T                           # the foot pushes the ground with -u
                F = jnp.where(ci[:, None], F_st, F_sw)
                tau = jnp.einsum('lji,lj->li', J, F).reshape(12)
                tau = jnp.clip(tau * self.torque_scale, -self.tau_lim, self.tau_lim)
                ctrl = jnp.zeros(self.m.nu).at[self.act_idx].set(tau)
                dx = dx.replace(ctrl=ctrl)
                dx = mjx.step(self.mx, dx)
                return dx, lift

            dx, lift = jax.lax.fori_loop(0, SUB, tick, (dx, lift))
            return (dx, lift), dict(s=s, feet=feet_w, c=c, u=u, t=t)

        _, R0, fw0, _, _ = self.measure(self.d0)
        (dx, _), out = jax.lax.scan(ctrl_step, (self.d0, fw0), jnp.arange(n_steps))
        s_end, _, _, _, _ = self.measure(dx)
        return out, s_end
