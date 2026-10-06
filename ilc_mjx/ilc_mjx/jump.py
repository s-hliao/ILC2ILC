"""
The jump on the GPU: B Go1 jumps at once in MJX, each its own goal, flown by the force policy -- ilc_quad's
controller (dilc_execute.Driver + ilc_jump_sim._tick_jump), synchronous and nominal:

  stand     every jump starts from one settled stance (the stand-up flown once, then copied)
  contact   samples k < Nc: at each sample's first tick the policy reads the SRB state and sets the four
            contact forces, u = u_ref(g)[k] + a_scale pi(obs), clipped into the friction cones and force
            limits; every tick the joints get the TO's torque plus J^T R^T (u - u_ref), damping only on
            legs in contact, joint PD on swing legs, the hips held (the controller's feedforward:=to_torque)
  flight    the TO's joint profile under joint PD, the legs levelled onto the landing late in flight
            (level_feet_samples, front lever)
  landing   a joint PD to the home pose (Nguyen et al.'s; the controller's landing_controller:=pd);
            a jump that tilts past max_tilt (or sinks: only the feet collide) fell

Box banks (plans whose config has a box: jumps onto a box, Nguyen et al.'s (x, z) targets): the model carries
QuadModel's static box geom, and every jump has its own -- the plans' boxes (front face x_front ahead of the
standing CoM, height) interpolated with the plans, set into the model's geom_pos / geom_size at every tick; the
thigh and calf capsules and the trunk meet the box too (not the floor), as on the CPU robots. A bank
may mix flat plans and box plans (one network for both): a flat plan is a box of height 0 (front face at
FLAT_X_FRONT before its goal), and a jump whose interpolated box is under BOX_MIN high has none (the geom moved
under the floor).

The SRB state is the true one, from the planar model: [CoM x, z (shifted to start on the plan), pitch, and
their rates]. With fd=True each contact sample also flies, alongside the jump, ten copies of it to the next
sample -- one per action channel moved by eps_a, one per SRB coordinate moved by eps_s x sx (the joints and
base changed by the smallest step in the planar mass metric that leaves the feet in contact where they
are) -- and the jump's one-step Jacobians A, B come from their differences (ilc_quad's FDJac, as batch
entries instead of forked processes).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import host_path
from .model import mjx_model, quad_model
from .planar import Planar
from .robot import HIP_IDX, PLANAR_TO_CANONICAL, expand

TICK = 0.002
FLAT_X_FRONT = 0.25        # a flat plan in a mixed bank: a box of height 0, front face this far before its goal
BOX_MIN = 0.005            # an interpolated box lower than this is none (dilc_execute.GoalBank.reference: the same)


@dataclass
class Config:
    stand_time: float = 1.0
    settle_time: float = 0.5
    land_time: float = 1.0
    gains: tuple = ((60.0, 5.0), (30.0, 1.0), (0.0, 1.0), (60.0, 5.0))   # stand, jump, contact, land
    mu: float = 0.6
    fmin: float = 5.0
    fmax: float = 500.0
    level_feet_samples: int = 10
    max_tilt: float = 0.6
    min_height: float = 0.10
    solver_iters: int = 4
    ls_iters: int = 8
    eps_a: float = 0.05
    eps_s: float = 0.1


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rotate(q, v):
    w, u = q[0], q[1:]
    return v + 2.0 * jnp.cross(u, jnp.cross(u, v) + w * v)


def _pitch(q):
    w, x, y, z = q
    return -jnp.arcsin(jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0))


class JumpEnv:
    def __init__(self, bank: dict, cfg: Config | None = None):
        from mujoco import mjx
        self.mjx = mjx
        self.cfg = cfg = cfg or Config()
        self.bank = bank
        self.has_box = any(p.get("box") is not None for p in bank["plans"])
        self.boxes = [p.get("box") or dict(x_front=p["goal"][0] - FLAT_X_FRONT, height=0.0) for p in bank["plans"]]
        self.BOX_L, self.BOX_W = 1.0, 1.0                       # QuadModel's box defaults (the CPU robots')
        # the model's own box: the tallest, far ahead (the stand-up never meets it; each jump sets its own)
        self.qm = qm = quad_model(bank.get("robot", "go1"),
                                  box=dict(x_front=5.0, height=max(b["height"] for b in self.boxes))
                                  if self.has_box else None)
        from ilc_quad.ilc_gen import PlanarQuadModel
        self.fb = fb = PlanarQuadModel(qm)
        self.pl = Planar(fb)
        self.m, self.mx = mjx_model(qm, feet_only_contacts=True, iterations=cfg.solver_iters,
                                    ls_iterations=cfg.ls_iters, legs_hit_box=self.has_box)
        refs = [dict(np.load(host_path(p["path"]))) for p in bank["plans"]]
        c0 = json.loads(str(refs[0]["config"]))
        self.Ndc, self.Nsc, self.Nfl = c0["phases"]
        self.Nc, self.N = self.Ndc + self.Nsc, self.Ndc + self.Nsc + self.Nfl
        self.dt = float(c0["dt"])
        self.tps = int(round(self.dt / TICK))                       # ticks per TO sample
        self.goals = np.array([p["goal"] for p in bank["plans"]], float)
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        self.plans = {k: f32(np.stack([r[src] for r in refs]))
                      for k, src in (("x_ref", "x_ref"), ("u_ref", "u_ref"), ("q_ref", "info_q_ref"),
                                     ("qd_ref", "info_qd_ref"), ("tau", "info_tau"), ("s", "info_s"))}
        if self.has_box:                                         # per plan its box: front face x, height
            import mujoco
            self.box_geom = mujoco.mj_name2id(qm.model, mujoco.mjtObj.mjOBJ_GEOM, "box")
            self.plans["box_xh"] = f32([[b["x_front"], b["height"]] for b in self.boxes])
        self.sx, self.a_scale = f32(bank["sx"]), f32(bank["a_scale"])
        self.g_center, self.g_scale = f32(bank["g_center"]), f32(bank["g_scale"])
        self.H, self.PA = int(bank.get("hist", 0)), bool(bank.get("prev_action", False))
        swing = np.zeros((self.Nc, 4), bool)
        swing[self.Ndc:, 0:2] = True
        self.swing = jnp.asarray(swing)
        E = np.zeros((12, 4))
        for i, idx in enumerate(PLANAR_TO_CANONICAL):
            E[idx, i] = 1.0
        self.E = f32(E)
        self.qadr, self.vadr, self.act = (jnp.asarray(x) for x in (qm.qpos_adr, qm.qvel_adr, qm.act_idx))
        self.lim = f32(qm.torque_limit)
        self.q_stand = f32(expand(fb.q_home))
        f_static = np.array([0.0, 0.5, 0.0, 0.5]) * fb.total_mass * fb.g
        self.tau_stand = f32(expand(fb.torque_map(fb.q_home, 0.0) @ f_static))
        self.hips = jnp.asarray(HIP_IDX)
        self.front = jnp.asarray(PLANAR_TO_CANONICAL[0] + PLANAR_TO_CANONICAL[1])
        self.rear = jnp.asarray(PLANAR_TO_CANONICAL[2] + PLANAR_TO_CANONICAL[3])
        self.base = int(qm.base_body_id)
        m = self.m
        self.robot_bodies = np.array([b for b in range(m.nbody) if m.body_rootid[b] == m.body_rootid[self.base]])
        self.nom = {k: np.asarray(getattr(m, k), np.float64).copy()
                    for k in ("body_mass", "body_inertia", "body_ipos", "geom_friction", "dof_frictionloss")}
        self.w_noload = f32(np.tile([30.1, 30.1, 20.06], 4))       # ilc_jump_lockstep.GO1_NO_LOAD_SPEED
        self.priv_dim = 0                                          # > 0: the actor also reads dyn["z"] (privileged)
        self.feet_geoms = jnp.asarray(qm.foot_geom_ids)
        self._stand = None
        self._jit = {}

    # -- goals ------------------------------------------------------------------------------
    def weights(self, goals):
        """Each goal's weights over the bank's plans (dilc_execute.GoalBank.weights)."""
        from dilc_execute import GoalBank
        gb = getattr(self, "_gb", None)
        if gb is None:
            gb = self._gb = GoalBank.__new__(GoalBank)
            gb.goals = self.goals
        return np.stack([GoalBank.weights(gb, g)[0] for g in goals])

    def references(self, goals):
        """Per jump the interpolated plan: x_ref, u_ref, q_ref, qd_ref, tau and the plan's feet off its base."""
        w = jnp.asarray(self.weights(goals), jnp.float32)
        ref = {k: jnp.einsum("bk,k...->b...", w, v) for k, v in self.plans.items()}
        feet = jax.vmap(jax.vmap(self.pl.feet))(ref["s"])                        # (B, N+1, 4)
        ref["feet_off"] = feet.reshape(feet.shape[:2] + (2, 2)) - ref["s"][..., None, :2]
        ref["goal_obs"] = (jnp.asarray(goals, jnp.float32) - self.g_center) / self.g_scale
        return ref

    # -- state ------------------------------------------------------------------------------
    def planar_state(self, dx):
        q, dq = dx.qpos[self.qadr], dx.qvel[self.vadr]
        quat = dx.qpos[3:7]
        th = _pitch(quat)
        w_world = _rotate(quat, dx.qvel[3:6])
        s = jnp.concatenate([dx.qpos[jnp.array([0, 2])], th[None], 0.5 * self.E.T @ q])
        sd = jnp.concatenate([dx.qvel[jnp.array([0, 1 + 1])], -w_world[1][None], 0.5 * self.E.T @ dq])
        return s, sd

    def srb(self, dx, shift):
        s, sd = self.planar_state(dx)
        com = self.pl.com(s)
        comd = self.pl.com_jac(s) @ sd
        return jnp.concatenate([com + shift, s[2:3], comd, sd[2:3]])

    # -- the controller ---------------------------------------------------------------------
    def _forces(self, a, k, ref):
        """The policy's action -> contact forces, clipped (Driver._clip), and the action as stored."""
        u = ref["u_ref"][k] + self.a_scale * a
        fz = jnp.clip(u[1::2], self.cfg.fmin, self.cfg.fmax)
        fx = jnp.clip(u[0::2], -self.cfg.mu * fz, self.cfg.mu * fz)
        u = jnp.stack([fx[0], fz[0], fx[1], fz[1]])
        u = jnp.where(self.swing[k], 0.0, u)
        a_st = jnp.clip(jnp.where(self.swing[k], 0.0, (u - ref["u_ref"][k]) / self.a_scale), -0.999, 0.999)
        return u, a_st

    def _command(self, dx, k, i, u, ref, contact: bool):
        """ilc_jump_sim._tick_jump at tick i of sample k: joint targets, gains, feedforward."""
        (gs, gj, gc, _) = self.cfg.gains
        fr = i / self.tps
        q, dq = dx.qpos[self.qadr], dx.qvel[self.vadr]
        q_des = self.E @ ((1 - fr) * ref["q_ref"][k] + fr * ref["q_ref"][k + 1])
        dq_des = self.E @ ((1 - fr) * ref["qd_ref"][k] + fr * ref["qd_ref"][k + 1])
        s, _ = self.planar_state(dx)
        if not contact:                         # level touchdown, the front feet's lever held
            off = (1 - fr) * ref["feet_off"][k] + fr * ref["feet_off"][k + 1]
            tgt = s[:2] + off
            tgt = tgt.at[0, 0].set(jnp.maximum(tgt[0, 0], s[0] + ref["feet_off"][self.N][0, 0]))
            q_lvl = self.E @ self.pl.ik(s, tgt.ravel(), 0.5 * self.E.T @ q_des)
            q_des = jnp.where(k >= self.N - self.cfg.level_feet_samples, q_lvl, q_des)
        tau_to = self.E @ (ref["tau"][k] / 2.0)
        kp, kd = jnp.full(12, gj[0]), jnp.full(12, gj[1])
        if contact:
            T = self.pl.torque_map(0.5 * self.E.T @ q, s[2])
            tau_to = tau_to + self.E @ (T @ (u - ref["u_ref"][k]))
            front_c = ~self.swing[k, 0]
            kp = kp.at[self.front].set(jnp.where(front_c, gc[0], gj[0]))
            kd = kd.at[self.front].set(jnp.where(front_c, gc[1], gj[1]))
            kp, kd = kp.at[self.rear].set(gc[0]), kd.at[self.rear].set(gc[1])
        kp, kd = kp.at[self.hips].set(gs[0]), kd.at[self.hips].set(gs[1])
        tau_ff = jnp.clip(tau_to, -self.lim, self.lim)
        return q_des, dq_des, kp, kd, tau_ff

    def _model(self, r=None):
        """The device model for one jump: with a box bank, its own box (r: the jump's reference) moved in."""
        if not self.has_box or r is None:
            return self.mx
        xf, h = r["box_xh"][0], r["box_xh"][1]
        hh = jnp.maximum(h, BOX_MIN) / 2
        pos = jnp.stack([xf + self.BOX_L / 2, 0.0, jnp.where(h >= BOX_MIN, hh, -1.0)])     # none: under the floor
        size = jnp.stack([jnp.float32(self.BOX_L / 2), jnp.float32(self.BOX_W / 2), hh])
        return self.mx.replace(geom_pos=self.mx.geom_pos.at[self.box_geom].set(pos),
                               geom_size=self.mx.geom_size.at[self.box_geom].set(size))

    def _step(self, dx, cmd, p=None, r=None):
        """One tick; p: this jump's dynamics (domain randomization, from sample_dyn), None for nominal; r: the
        jump's reference (its box, with a box bank)."""
        q_des, dq_des, kp, kd, tau_ff = cmd
        q, dq = dx.qpos[self.qadr], dx.qvel[self.vadr]
        tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
        mx = self._model(r)
        if p is None:
            tau = jnp.clip(tau, -self.lim, self.lim)
            dx = dx.replace(ctrl=dx.ctrl.at[self.act].set(tau))
            return self.mjx.step(mx, dx)
        # the reality model's motors (ilc_jump_lockstep.Reality.motor_torque): scaled limits, and the
        # torque-speed envelope while motoring (full torque to half the no-load speed, none at it)
        lim = self.lim * p["motor_scale"]
        w_max = self.w_noload * p["speed_scale"]
        drop = jnp.clip((w_max - jnp.abs(dq)) / (0.5 * w_max), 0.0, 1.0)
        lim = jnp.where((p["curve"] > 0.5) & (tau * dq > 0), lim * drop, lim)
        tau = jnp.clip(tau, -lim, lim)
        dx = dx.replace(ctrl=dx.ctrl.at[self.act].set(tau))
        mx = mx.replace(body_mass=p["body_mass"], body_inertia=p["body_inertia"], body_ipos=p["body_ipos"],
                        geom_friction=p["geom_friction"], dof_frictionloss=p["dof_frictionloss"])
        return self.mjx.step(mx, dx)

    def _fly_sample(self, dx, k, u, ref, contact, p=None):
        def tick(dx, i):
            return self._step(dx, self._command(dx, k, i, u, ref, contact), p, ref), None
        return jax.lax.scan(tick, dx, jnp.arange(self.tps))[0]

    # -- domain randomization (baselines only: our method trains on the nominal dynamics) ----------
    DYN_KEYS = ("mass_scale", "com_x", "motor_scale", "curve", "speed_scale", "friction", "joint_friction", "payload")
    DYN_CENTER = np.array([1.025, 0.0, 0.925, 0.5, 0.95, 0.85, 0.15, 0.3])
    DYN_SCALE = np.array([0.0725, 0.0115, 0.045, 0.5, 0.05, 0.2, 0.09, 0.6])

    def sample_dyn(self, B, rng):
        """B robots from the dynamics part of dilc_execute.sample_condition (the family the real-like robots are
        drawn from): mass x U(0.9, 1.15), CoM x U(-2, 2) cm, motors x U(0.85, 1), half with the torque-speed curve
        (speed x U(0.85, 1)), half with ground friction U(0.5, 0.8), joint friction + U(0, 0.3) N m, 30 % with a
        payload U(0, 2) kg on the trunk. NOT in the family (the GPU sim cannot): delays, jitter, dropped commands,
        mocap, joint/foot sensing, ground stiffness. Returns raw params (B, 8) in DYN_KEYS order."""
        curve = rng.random(B) < 0.5
        fr = rng.random(B) < 0.5
        pay = rng.random(B) < 0.3
        return np.stack([rng.uniform(0.9, 1.15, B), rng.uniform(-0.02, 0.02, B), rng.uniform(0.85, 1.0, B),
                         curve.astype(float), np.where(curve, rng.uniform(0.85, 1.0, B), 1.0),
                         np.where(fr, rng.uniform(0.5, 0.8, B), 1.0), rng.uniform(0.0, 0.3, B),
                         np.where(pay, rng.uniform(0.0, 2.0, B), 0.0)], 1)

    def dyn_from_cond(self, args: str):
        """The raw params (8,) of one ilc_quad condition string (its dynamics part)."""
        import shlex
        t = shlex.split(args)
        val = lambda flag, d, n=1: (float(t[t.index(flag) + 1]) if n == 1 else [float(x) for x in t[t.index(flag) + 1:t.index(flag) + 1 + n]]) if flag in t else d
        curve = "--motor-curve" in t
        com = val("--com-offset", [0.0, 0.0], 2)
        return np.array([val("--mass-scale", 1.0), com[0], val("--motor-scale", 1.0), float(curve),
                         val("--motor-speed-scale", 1.0) if curve else 1.0, val("--friction", 1.0),
                         val("--joint-friction", 0.0), val("--payload", 0.0)])

    def dyn_arrays(self, params):
        """Raw params (B, 8) -> the per-jump model arrays for _step (and z, the normalized params)."""
        P = np.asarray(params, float)
        B = len(P)
        nom = self.nom
        mass = np.repeat(nom["body_mass"][None], B, 0)
        inertia = np.repeat(nom["body_inertia"][None], B, 0)
        ipos = np.repeat(nom["body_ipos"][None], B, 0)
        rb, b0 = self.robot_bodies, self.base
        mass[:, rb] *= P[:, 0:1]
        inertia[:, rb] *= P[:, 0:1, None]
        ipos[:, b0, 0] += P[:, 1]
        # the payload (a 0.15 x 0.10 x 0.05 box at (0, 0, 0.08) on the trunk) merged into the trunk
        pm, M = P[:, 7], mass[:, b0].copy()
        pos = np.array([0.0, 0.0, 0.08])
        c_new = (M[:, None] * ipos[:, b0] + pm[:, None] * pos) / (M + pm)[:, None]
        d_old, d_pay = ipos[:, b0] - c_new, pos[None] - c_new
        size = np.array([0.15, 0.10, 0.05])
        box = (size[[1, 0, 0]] ** 2 + size[[2, 2, 1]] ** 2) / 12.0
        par = lambda d: np.stack([d[:, 1] ** 2 + d[:, 2] ** 2, d[:, 0] ** 2 + d[:, 2] ** 2, d[:, 0] ** 2 + d[:, 1] ** 2], 1)
        inertia[:, b0] += M[:, None] * par(d_old) + pm[:, None] * (box[None] + par(d_pay))
        ipos[:, b0], mass[:, b0] = c_new, M + pm
        fric = np.repeat(nom["geom_friction"][None], B, 0)
        fric[:, :, 0] = np.where(P[:, 5:6] < 0.999, P[:, 5:6], fric[:, :, 0])
        floss = np.repeat(nom["dof_frictionloss"][None], B, 0)
        floss[:, np.asarray(self.vadr)] += P[:, 6:7]
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        return dict(body_mass=f32(mass), body_inertia=f32(inertia), body_ipos=f32(ipos), geom_friction=f32(fric),
                    dof_frictionloss=f32(floss), motor_scale=f32(P[:, 2]), curve=f32(P[:, 3]),
                    speed_scale=f32(P[:, 4]), z=f32((P - self.DYN_CENTER) / self.DYN_SCALE))

    # -- the policy -------------------------------------------------------------------------
    @staticmethod
    def planner(w, obs):
        """FADA-style planner (a baseline): the next sample's normalized SRB error the policy should reach."""
        h = obs
        for i in range(2):
            h = jax.nn.relu(w[f"P_W{i}"] @ h + w[f"P_b{i}"])
        return w["P_Wo"] @ h + w["P_bo"]

    @staticmethod
    def actor(w, obs, key=None):
        h = obs
        if "P_W0" in w:                 # planner-IDM: the IDM reads the observation and the planned next error
            h = jnp.concatenate([obs, JumpEnv.planner(w, obs)])
        for i in range(2):
            h = jax.nn.relu(w[f"W{i}"] @ h + w[f"b{i}"])
        mu = w["W_mu"] @ h + w["b_mu"]
        if key is None:
            return jnp.tanh(mu)
        ls = jnp.clip(w["W_ls"] @ h + w["b_ls"], -5.0, 1.0)
        return jnp.tanh(mu + jnp.exp(ls) * jax.random.normal(key, mu.shape))

    def obs(self, err_hist, prev_a, k, goal_obs):
        """dilc_execute.GoalBank.obs_at: [error, time, goal, the previous H errors, the previous action]."""
        parts = [err_hist[0], jnp.array([2.0 * k / self.Nc - 1.0]), goal_obs]
        parts += [err_hist[j] for j in range(1, self.H + 1)]
        if self.PA:
            parts.append(prev_a)
        return jnp.concatenate(parts)

    # -- finite differences -------------------------------------------------------------------
    def _perturb(self, dx, i, k):
        """SRB coordinate i (0-2 position, 3-5 rate) moved by eps_s sx[i], feet in contact held."""
        s, _ = self.planar_state(dx)
        H = self.pl.mass_matrix(s)
        Jcom = self.pl.com_jac(s)
        Jf = self.pl.feet_jac(s)
        c_front = (k < self.Ndc).astype(jnp.float32)
        Jf = Jf * jnp.array([c_front, c_front, 1.0, 1.0])[:, None]
        C = jnp.concatenate([Jcom, jnp.eye(7)[2:3], Jf])
        rhs = jnp.zeros(7).at[i % 3].set(self.cfg.eps_s * self.sx[i])
        HiCt = jnp.linalg.solve(H, C.T)
        G = C @ HiCt
        ds = HiCt @ jnp.linalg.solve(G + 1e-9 * jnp.trace(G) / 7 * jnp.eye(7), rhs)
        quat = dx.qpos[3:7]
        is_pos = i < 3
        half = -0.5 * ds[2]                                          # nose-up = about -y (world)
        dq_rot = _quat_mul(jnp.array([jnp.cos(half), 0.0, jnp.sin(half), 0.0]), quat)
        qpos = dx.qpos.at[0].add(ds[0]).at[2].add(ds[1]).at[3:7].set(dq_rot / jnp.linalg.norm(dq_rot))
        qpos = qpos.at[self.qadr].add(self.E @ ds[3:])
        w_body = _rotate(quat * jnp.array([1.0, -1.0, -1.0, -1.0]), jnp.array([0.0, -ds[2], 0.0]))
        qvel = dx.qvel.at[0].add(ds[0]).at[2].add(ds[1]).at[3:6].add(w_body).at[self.vadr].add(self.E @ ds[3:])
        return dx.replace(qpos=jnp.where(is_pos, qpos, dx.qpos), qvel=jnp.where(is_pos, dx.qvel, qvel))

    # -- one batch of jumps ---------------------------------------------------------------------
    def stand(self):
        """One settled stance (stand-up ramp and settle under the stand PD), for every jump to start from."""
        if self._stand is None:
            import mujoco
            d = mujoco.MjData(self.m)
            mujoco.mj_resetDataKeyframe(self.m, d, 0)
            mujoco.mj_forward(self.m, d)
            dx = self.mjx.put_data(self.m, d)
            q0 = dx.qpos[self.qadr]
            gs = self.cfg.gains[0]
            n = int(round((self.cfg.stand_time + self.cfg.settle_time) / TICK))

            def tick(dx, j):
                al = jnp.minimum(j * TICK / self.cfg.stand_time, 1.0)
                cmd = ((1 - al) * q0 + al * self.q_stand, jnp.zeros(12), jnp.full(12, al * gs[0]),
                       jnp.full(12, gs[1]), al * self.tau_stand)
                return self._step(dx, cmd), None
            self._stand = jax.jit(lambda dx: jax.lax.scan(tick, dx, jnp.arange(n))[0])(dx)
        return self._stand

    def stand_batch(self, q_off, dyn=None):
        """Per jump its own settled stance: the stand-up ramp to the home pose plus q_off (B, 12) (exploration:
        the jumps start from different stances, so the policy meets states off the nominal one)."""
        if "stand_b" not in self._jit:
            import mujoco
            d = mujoco.MjData(self.m)
            mujoco.mj_resetDataKeyframe(self.m, d, 0)
            mujoco.mj_forward(self.m, d)
            dx = self.mjx.put_data(self.m, d)
            q0 = dx.qpos[self.qadr]
            gs = self.cfg.gains[0]
            n = int(round((self.cfg.stand_time + self.cfg.settle_time) / TICK))

            def one(off, p):
                def tick(dx_, j):
                    al = jnp.minimum(j * TICK / self.cfg.stand_time, 1.0)
                    cmd = ((1 - al) * q0 + al * (self.q_stand + off), jnp.zeros(12), jnp.full(12, al * gs[0]),
                           jnp.full(12, gs[1]), al * self.tau_stand)
                    return self._step(dx_, cmd, p), None
                return jax.lax.scan(tick, dx, jnp.arange(n))[0]
            self._jit["stand_b"] = jax.jit(jax.vmap(one))
        return self._jit["stand_b"](jnp.asarray(q_off, jnp.float32), dyn)

    def explore_noise(self, B, rng, sig_q, sig_a):
        """Exploration (not domain randomization: the dynamics stay nominal): per jump a stance offset of the planar
        joints ~ N(0, sig_q) rad, and smooth action offsets (a Gaussian kernel of 4 samples) of rms sig_a over the
        contact samples, the swing legs' entries zero."""
        E = np.zeros((12, 4))
        for i, idx in enumerate(PLANAR_TO_CANONICAL):
            E[idx, i] = 1.0
        q_off = (rng.normal(0.0, sig_q, (B, 4)) @ E.T) if sig_q > 0 else np.zeros((B, 12))
        a_off = np.zeros((B, self.Nc, 4))
        if sig_a > 0:
            ker = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
            z = np.stack([[np.convolve(rng.standard_normal(self.Nc + 24), ker, "valid")[:self.Nc] for _ in range(4)]
                          for _ in range(B)], 0).transpose(0, 2, 1)
            z = z * ~np.asarray(self.swing)[None]
            a_off = z * sig_a / np.sqrt((z ** 2).sum((1, 2), keepdims=True) / (~np.asarray(self.swing)).sum())
        return q_off, a_off

    def restore(self, rec, ks, contact_front, foot_z=None, window=0.025):
        """MuJoCo (qpos, qvel) at contact samples ks of a recorded jump (rec_* of a saved episode: the robot's own
        measurements -- mocap pose, joint encoders), for linearizing the nominal model along a real trial: base
        velocities by a causal least-squares slope over `window` (the controller estimator's), then the stance feet
        put on the ground (the base shifted down/up) and their velocities projected to zero (the smallest change in
        the mass metric), so measurement noise does not start the copies with a contact pop. foot_z (Nc, 4): the
        feet's heights in the nominal sim at each sample (rollout's foot_z; the soft contact's depth under load):
        the stance feet are put at their mean (else exactly on the floor)."""
        import mujoco
        m = self.m
        d = mujoco.MjData(m)
        t, pos, quat, q, dq = (np.asarray(rec[k_], float) for k_ in ("t", "pos", "quat", "q", "dq"))
        qadr, vadr = np.asarray(self.qm.qpos_adr), np.asarray(self.qm.qvel_adr)
        feet = np.asarray(self.qm.foot_geom_ids)
        r_foot = float(m.geom_size[feet[0], 0])
        Q, V = [], []
        for k in ks:
            i = min(int(k) * self.tps, len(t) - 1)
            sel = (t <= t[i] + 1e-9) & (t >= t[i] - window)
            tt = t[sel] - t[i]
            slope = lambda y: np.polyfit(tt, y, 1)[0] if len(tt) > 2 else np.zeros(y.shape[1:])
            v = slope(pos[sel])
            # body angular velocity from the quaternion's slope: w_body = 2 conj(q) dq/dt
            qs = quat[sel] * np.sign((quat[sel] * quat[i]).sum(1, keepdims=True))
            dqt = slope(qs)
            w0, xv = quat[i][0], quat[i][1:]
            conj = np.array([w0, -xv[0], -xv[1], -xv[2]])
            prod = np.array(_quat_mul(jnp.asarray(conj), jnp.asarray(dqt)))
            d.qpos[:] = m.qpos0
            d.qpos[0:3], d.qpos[3:7] = pos[i], quat[i] / np.linalg.norm(quat[i])
            d.qpos[qadr] = q[i]
            d.qvel[:] = 0.0
            d.qvel[0:3], d.qvel[3:6] = v, 2.0 * prod[1:]
            d.qvel[vadr] = dq[i]
            st = feet if (contact_front[k] if hasattr(contact_front, "__len__") else contact_front) else feet[2:]
            mujoco.mj_kinematics(m, d)
            st_i = np.searchsorted(feet, st) if foot_z is not None else None
            z_tgt = float(np.mean(np.asarray(foot_z)[int(k), st_i])) if foot_z is not None else r_foot
            d.qpos[2] -= np.mean(d.geom_xpos[st, 2]) - z_tgt               # the stance feet on the floor
            mujoco.mj_forward(m, d)
            J = []
            for g in st:
                jp = np.zeros((3, m.nv))
                mujoco.mj_jacGeom(m, d, jp, None, g)
                J.append(jp)
            J = np.concatenate(J)
            Mi = np.zeros((m.nv, m.nv))
            mujoco.mj_solveM(m, d, Mi, np.eye(m.nv))
            G = J @ Mi @ J.T
            d.qvel[:] = d.qvel - Mi @ J.T @ np.linalg.solve(G + 1e-9 * np.trace(G) * np.eye(len(G)), J @ d.qvel)
            Q.append(d.qpos.copy())
            V.append(d.qvel.copy())
        return np.array(Q), np.array(V)

    def fd_at(self, qpos, qvel, ks, acts, ref):
        """One-step FD Jacobians A (B, 6, 6), B (B, 6, 4) of the nominal model from given states (restore), entry b at
        contact sample ks[b] with action acts[b] (normalized, as flown), its goal's reference ref (per entry)."""
        if "fd_at" not in self._jit:
            self._jit["fd_at"] = jax.jit(self._fd_at)
        out = self._jit["fd_at"](jnp.asarray(qpos, jnp.float32), jnp.asarray(qvel, jnp.float32),
                                 jnp.asarray(ks), jnp.asarray(acts, jnp.float32), ref, self.stand())
        return tuple(np.asarray(x) for x in out)

    def _fd_at(self, qpos, qvel, ks, acts, ref, tmpl):
        nc = 11
        B = qpos.shape[0]
        dx = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), tmpl)
        dx = dx.replace(qpos=qpos, qvel=qvel)
        dx = jax.vmap(lambda d, r: self.mjx.forward(self._model(r), d))(dx, ref)
        da = jnp.concatenate([jnp.zeros((1, 4)), self.cfg.eps_a * jnp.eye(4), jnp.zeros((6, 4))])

        def one(d, k, a, r):
            uc, _ = jax.vmap(lambda d_: self._forces(a + d_, k, r))(da)
            dc = jax.tree_util.tree_map(lambda v: jnp.repeat(v[None], nc, 0), d)
            dc = jax.vmap(lambda d_, j: jax.lax.cond(j >= 5, lambda e: self._perturb(e, j - 5, k), lambda e: e, d_))(
                dc, jnp.arange(nc))
            s0 = jax.vmap(lambda d_: self.srb(d_, jnp.zeros(2)))(dc)
            dc = jax.vmap(lambda d_, u_: self._fly_sample(d_, k, u_, r, True))(dc, uc)
            s1 = jax.vmap(lambda d_: self.srb(d_, jnp.zeros(2)))(dc)
            Bj = ((s1[1:5] - s1[0][None]) / self.cfg.eps_a).T
            Bj = jnp.where(self.swing[k][None, :], 0.0, Bj)
            dS0 = (s0[5:] - s0[:1]).T
            dS1 = (s1[5:] - s1[0][None]).T
            return dS1 @ jnp.linalg.inv(dS0), Bj
        return jax.vmap(one)(dx, ks, acts, ref)

    def rollout(self, weights, ref, key, stochastic=True, fd=False, a_offset=None, q_offset=None, dyn=None):
        """B jumps (ref from references(goals)); weights: the actor's arrays. Returns per jump X (N+1, 6)
        the true SRB state at every sample, U (Nc, 4) forces, act (Nc, 4), obs (Nc+1, od), fell, and with
        fd the one-step Jacobians A (Nc, 6, 6), B (Nc, 6, 4) (SRB units; B per unit of normalized action).
        a_offset (B, Nc, 4): added to the policy's action at every contact sample (exact perturbations: the
        sim is deterministic, so a jump flown twice is the same jump). q_offset (B, 12): each jump's own stance
        (stand_batch). Also returns mu (Nc, 4): the policy's mean action at each sample (the ILC targets' base).
        dyn: per-jump dynamics, dyn_arrays(params) (domain randomization; None: nominal); with priv_dim > 0 the
        actor also reads dyn["z"] (a privileged teacher), and obs carry it."""
        tag = (bool(stochastic), bool(fd), dyn is not None, self.priv_dim)
        if tag not in self._jit:
            self._jit[tag] = jax.jit(lambda w, r, k_, d0, off, p: self._rollout(w, r, k_, d0, off, *tag[:2], p))
        B = ref["x_ref"].shape[0]
        off = jnp.zeros((B, self.Nc, 4), jnp.float32) if a_offset is None else jnp.asarray(a_offset, jnp.float32)
        if q_offset is None and dyn is None:
            d0 = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), self.stand())
        else:
            d0 = self.stand_batch(np.zeros((B, 12)) if q_offset is None else q_offset, dyn)
        out = self._jit[tag](weights, ref, key, d0, off, dyn)
        return jax.tree_util.tree_map(np.asarray, out)

    def _rollout(self, w, ref, key, d_stand, a_off, stochastic, fd, dyn=None):
        B = ref["x_ref"].shape[0]
        zp = dyn["z"][:, :self.priv_dim] if (dyn is not None and self.priv_dim) else None
        dx0 = d_stand
        shift = jax.vmap(lambda r, d: r["x_ref"][0, :2] - self.pl.com(self.planar_state(d)[0]))(ref, dx0)
        H, od = self.H, 9 + 6 * self.H + (4 if self.PA else 0)
        srb = jax.vmap(self.srb)
        nc = 1 + 4 + 6 if fd else 1

        def contact_sample(carry, k):
            dx, hist, prev_a, key = carry
            x = srb(dx, shift)
            err = jnp.clip((x - ref["x_ref"][:, k]) / self.sx, -20.0, 20.0)
            hist = jnp.concatenate([err[:, None], hist[:, :-1]], 1) if H else err[:, None]
            o = jax.vmap(lambda h, pa, g: self.obs(h, pa, k, g))(hist, prev_a, ref["goal_obs"])
            if zp is not None:
                o = jnp.concatenate([o, zp], 1)
            key, sub = jax.random.split(key)
            keys = jax.random.split(sub, B)
            mu = jax.vmap(lambda o_: self.actor(w, o_))(o)
            a = jax.vmap(lambda o_, kk: self.actor(w, o_, kk if stochastic else None))(o, keys) + a_off[:, k]
            u, a_st = jax.vmap(lambda a_, r: self._forces(a_, k, r))(a, ref)
            fz = dx.geom_xpos[:, self.feet_geoms, 2]                          # (B, 4) the feet's height
            if not fd:
                dx = jax.vmap(lambda d, u_, r, p: self._fly_sample(d, k, u_, r, True, p))(dx, u, ref, dyn)
                return (dx, hist, a_st, key), (x, u, a_st, o, mu, jnp.zeros((B, 0, 6)), jnp.zeros((B, 0, 6)), fz)
            # the jump and its ten copies, side by side: copy 0 the jump itself
            da = jnp.concatenate([jnp.zeros((1, 4)), self.cfg.eps_a * jnp.eye(4), jnp.zeros((6, 4))])
            uc, _ = jax.vmap(lambda a_, r: jax.vmap(lambda d_: self._forces(a_ + d_, k, r))(da))(a, ref)
            dxc = jax.tree_util.tree_map(lambda v: jnp.repeat(v[:, None], nc, 1), dx)
            idx = jnp.arange(nc)

            def pert(d, j):
                return jax.lax.cond(j >= 5, lambda d_: self._perturb(d_, j - 5, k), lambda d_: d_, d)
            dxc = jax.vmap(jax.vmap(pert, (0, 0)), (0, None))(dxc, idx)
            s0 = jax.vmap(jax.vmap(lambda d: self.srb(d, jnp.zeros(2))))(dxc)
            refc = jax.tree_util.tree_map(lambda v: jnp.repeat(v[:, None], nc, 1), ref)
            dxc = jax.vmap(jax.vmap(lambda d, u_, r, p: self._fly_sample(d, k, u_, r, True, p), (0, 0, 0, None)))(
                dxc, uc, refc, dyn)
            s1 = jax.vmap(jax.vmap(lambda d: self.srb(d, jnp.zeros(2))))(dxc)
            dx = jax.tree_util.tree_map(lambda v: v[:, 0], dxc)
            return (dx, hist, a_st, key), (x, u, a_st, o, mu, s0, s1, fz)

        def flight_sample(dx, k):
            x = srb(dx, shift)
            dx = jax.vmap(lambda d, r, p: self._fly_sample(d, k, jnp.zeros(4), r, False, p))(dx, ref, dyn)
            return dx, x

        hist0 = jnp.zeros((B, H + 1, 6))
        (dx, hist, prev_a, _), (Xc, U, A_st, O, MU, S0, S1, FZ) = jax.lax.scan(
            contact_sample, (dx0, hist0, jnp.zeros((B, 4)), key), jnp.arange(self.Nc))
        x_nc = srb(dx, shift)
        err = jnp.clip((x_nc - ref["x_ref"][:, self.Nc]) / self.sx, -20.0, 20.0)
        hist = jnp.concatenate([err[:, None], hist[:, :-1]], 1) if H else err[:, None]
        o_nc = jax.vmap(lambda h, pa, g: self.obs(h, pa, self.Nc, g))(hist, prev_a, ref["goal_obs"])
        if zp is not None:
            o_nc = jnp.concatenate([o_nc, zp], 1)
        dx, Xf = jax.lax.scan(flight_sample, dx, jnp.arange(self.Nc, self.N))
        x_n = srb(dx, shift)

        # the landing: joint PD to the home pose; a fall is a trunk past max_tilt or sunk
        gl = self.cfg.gains[3]
        cmd = (self.q_stand, jnp.zeros(12), jnp.full(12, gl[0]), jnp.full(12, gl[1]), self.tau_stand)

        def land_tick(carry, _):
            dx, fell = carry
            dx = jax.vmap(lambda d, p, r: self._step(d, cmd, p, r))(dx, dyn, ref)
            quat = dx.qpos[:, 3:7]
            tilt = jnp.arccos(jnp.clip(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1.0, 1.0))
            fell = fell | (tilt > self.cfg.max_tilt) | (dx.qpos[:, 2] < self.cfg.min_height)
            return (dx, fell), None
        n_land = int(round(self.cfg.land_time / TICK))
        (dx, fell), _ = jax.lax.scan(land_tick, (dx, jnp.zeros(B, bool)), None, length=n_land)

        sw = lambda v: jnp.swapaxes(v, 0, 1)
        X = jnp.concatenate([sw(Xc), x_nc[:, None], sw(Xf)[:, 1:], x_n[:, None]], 1)   # (B, N+1, 6)
        out = dict(X=X, U=sw(U), act=sw(A_st), obs=jnp.concatenate([sw(O), o_nc[:, None]], 1), fell=fell,
                   mu=sw(MU) * ~self.swing[None], foot_z=sw(FZ))
        if fd:
            S0, S1 = sw(S0), sw(S1)                                    # (B, Nc, nc, 6)
            s1 = S1[:, :, 0]                                           # copy 0: the jump itself, at k+1
            Bj = jnp.swapaxes((S1[:, :, 1:5] - s1[:, :, None]) / self.cfg.eps_a, -1, -2)
            Bj = jnp.where(self.swing[None, :, None, :], 0.0, Bj)
            dS0 = jnp.swapaxes(S0[:, :, 5:] - S0[:, :, :1], -1, -2)    # (B, Nc, 6 state, 6 copies)
            dS1 = jnp.swapaxes(S1[:, :, 5:] - s1[:, :, None], -1, -2)
            out.update(A=dS1 @ jnp.linalg.inv(dS0), B=Bj)
        return out
