#!/usr/bin/env python3
"""feasibility.py: can MJX fly our jump? (step 0 of the GPU port)

  1. MJX takes ilc_quad's Go1 (all collision geoms, and feet-only contacts)
  2. one jump with the same simple synchronous controller on CPU MuJoCo and on MJX: stand up and settle,
     the TO plan open loop (its joint torque feedforward, joint PD on swing legs, damping on stance legs),
     the landing a joint PD to the home pose (Nguyen et al.'s) -- the base's x, z, pitch side by side
  3. throughput: batched jumps per second on one GPU, at several batch sizes

    ~/miniconda3/envs/ilcmjx/bin/python scripts/feasibility.py [--goal 0.5] [--gpu 0]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="/home/henry/ilc_ws/src/ilc_quad/experiments/go1/plans/ref_s8_go1_f50_m85.npz")
ap.add_argument("--gpu", default="0")
ap.add_argument("--batches", type=int, nargs="*", default=[64, 512, 2048])
ap.add_argument("--feet-only", action="store_true")
ap.add_argument("--iters", type=int, default=0)
ap.add_argument("--ls-iters", type=int, default=0)
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402

from ilc_mjx.model import quad_model, mjx_model  # noqa: E402
from ilc_quad.ilc_gen import PlanarQuadModel  # noqa: E402
from ilc_mjx.robot import expand, planar_pitch, HIP_IDX, PLANAR_TO_CANONICAL  # noqa: E402

qm = quad_model()
fb = PlanarQuadModel(qm)
m, mx = mjx_model(qm, feet_only_contacts=a.feet_only, iterations=a.iters, ls_iterations=a.ls_iters)
print(f"MJX model: {m.ngeom} geoms, contacts {'feet only' if a.feet_only else 'all'}, solver iters {m.opt.iterations}/{m.opt.ls_iterations}; jax {jax.__version__} "
      f"on {jax.devices()[0]}")

# the controller's numbers (ilc_jump_sim defaults, as dilc_execute flies them)
plan = np.load(a.plan)
Ndc, Nsc, Nfl = 30, 30, 30
Nc, N, dt_to, tick = Ndc + Nsc, 90, 0.01, 0.002
STAND, SETTLE, LAND = 1.0, 0.5, 1.0
g_stand, g_jump, g_contact, g_land = (60.0, 5.0), (30.0, 1.0), (0.0, 1.0), (60.0, 5.0)
q_stand = expand(fb.q_home)
f_static = np.array([0.0, 0.5, 0.0, 0.5]) * fb.total_mass * fb.g
tau_stand = expand(fb.torque_map(fb.q_home, 0.0) @ f_static)
q_ref = np.array([expand(v) for v in plan["info_q_ref"]])          # (N+1, 12)
qd_ref = np.array([expand(v) for v in plan["info_qd_ref"]])
tau_to = np.array([expand(v / 2.0) for v in plan["info_tau"]])     # (N, 12), per motor
swing = np.zeros((Nc, 4), bool)
swing[Ndc:, 0:2] = True
stance_front = np.array([not swing[k, 0] for k in range(Nc)] + [False] * (N - Nc))
F_IDX = PLANAR_TO_CANONICAL[0] + PLANAR_TO_CANONICAL[1]
R_IDX = PLANAR_TO_CANONICAL[2] + PLANAR_TO_CANONICAL[3]
qadr, vadr, act = qm.qpos_adr, qm.qvel_adr, qm.act_idx
lim = qm.torque_limit
n_stand, n_jump, n_land = int(round((STAND + SETTLE) / tick)), int(round(N * dt_to / tick)), int(round(LAND / tick))
T = n_stand + n_jump + n_land

# one table per tick: q_des, dq_des, kp, kd, tau_ff (the controller is open loop in time)
Qd, DQd, KP, KD, TFF = (np.zeros((T, 12)) for _ in range(5))
q0 = qm.home_qpos
for i in range(T):
    if i < n_stand:
        t = i * tick
        al = min(t / STAND, 1.0)
        Qd[i], KP[i], KD[i], TFF[i] = (1 - al) * q0 + al * q_stand, al * g_stand[0], g_stand[1], al * tau_stand
    elif i < n_stand + n_jump:
        t = (i - n_stand) * tick
        k = min(int(t / dt_to + 1e-6), N - 1)
        fr = t / dt_to - k
        Qd[i] = (1 - fr) * q_ref[k] + fr * q_ref[k + 1]
        DQd[i] = (1 - fr) * qd_ref[k] + fr * qd_ref[k + 1]
        kp, kd = np.full(12, g_jump[0]), np.full(12, g_jump[1])
        if k < Nc:
            if not swing[k, 0]:
                kp[F_IDX], kd[F_IDX] = g_contact
            kp[R_IDX], kd[R_IDX] = g_contact                      # rear pair in contact to takeoff
        kp[HIP_IDX], kd[HIP_IDX] = g_stand
        KP[i], KD[i], TFF[i] = kp, kd, np.clip(tau_to[k], -lim, lim)
    else:
        Qd[i], KP[i], KD[i], TFF[i] = q_stand, g_land[0], g_land[1], tau_stand


def pitch(quat):
    return planar_pitch(quat)


# CPU MuJoCo
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
cpu = np.zeros((T, 3))
t0 = time.time()
for i in range(T):
    q, dq = d.qpos[qadr], d.qvel[vadr]
    tau = np.clip(KP[i] * (Qd[i] - q) + KD[i] * (DQd[i] - dq) + TFF[i], -lim, lim)
    d.ctrl[act] = tau
    mujoco.mj_step(m, d)
    cpu[i] = d.qpos[0], d.qpos[2], pitch(d.qpos[3:7])
t_cpu = time.time() - t0

# MJX: the same, the tables on the device, one scan over the ticks
tabs = tuple(jnp.asarray(x, jnp.float32) for x in (Qd, DQd, KP, KD, TFF))
jl = jnp.asarray(lim, jnp.float32)
jq, jv, ja = jnp.asarray(qadr), jnp.asarray(vadr), jnp.asarray(act)


def jpitch(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return -jnp.arcsin(jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0))


def tick_fn(dx, tab):
    qd_, dqd_, kp, kd, tff = tab
    q, dq = dx.qpos[jq], dx.qvel[jv]
    tau = jnp.clip(kp * (qd_ - q) + kd * (dqd_ - dq) + tff, -jl, jl)
    dx = dx.replace(ctrl=dx.ctrl.at[ja].set(tau))
    dx = mjx.step(mx, dx)
    return dx, jnp.stack([dx.qpos[0], dx.qpos[2], jpitch(dx.qpos[3:7])])


def rollout(dx0):
    return jax.lax.scan(tick_fn, dx0, tabs)[1]


d0 = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d0, 0)
mujoco.mj_forward(m, d0)
dx0 = mjx.put_data(m, d0)
one = jax.jit(rollout)
t0 = time.time()
gpu = np.asarray(one(dx0))
t_c1 = time.time() - t0
t0 = time.time()
gpu = np.asarray(jax.block_until_ready(one(dx0)))
t_g1 = time.time() - t0

print(f"\none jump ({T} ticks): CPU {t_cpu:.2f} s; MJX compile+run {t_c1:.1f} s, run {t_g1:.2f} s")
for name, i in (("settled", n_stand - 1), ("takeoff k=Nc", n_stand + int(Nc * dt_to / tick)),
                ("landing k=N", n_stand + n_jump - 1), ("end", T - 1)):
    c, g = cpu[i], gpu[i]
    print(f"  {name:13s} CPU x {c[0]:+.3f} z {c[1]:.3f} pitch {np.degrees(c[2]):+6.1f} | "
          f"MJX x {g[0]:+.3f} z {g[1]:.3f} pitch {np.degrees(g[2]):+6.1f}")
dev = np.abs(cpu - gpu)
print(f"  max |CPU - MJX| through the jump: x {dev[:n_stand + n_jump, 0].max():.3f} m, "
      f"z {dev[:n_stand + n_jump, 1].max():.3f} m, pitch {np.degrees(dev[:n_stand + n_jump, 2].max()):.1f} deg")

# throughput
for B in a.batches:
    dxb = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), dx0)
    f = jax.jit(jax.vmap(rollout))
    t0 = time.time()
    jax.block_until_ready(f(dxb))
    tc = time.time() - t0
    t0 = time.time()
    jax.block_until_ready(f(dxb))
    tr = time.time() - t0
    print(f"batch {B:5d}: {B / tr:8.1f} jumps/s ({B * T / tr / 1e3:8.1f} k steps/s; compile {tc:.0f} s); "
          f"CPU one core {1 / t_cpu:.2f} jumps/s")
