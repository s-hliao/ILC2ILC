#!/usr/bin/env python3
"""Goal-conditioned Deep-ILC force policy on the Go1 lockstep sim: execute it, serve
rollouts to its trainer, and turn ILC runs into its training episodes.

    dilc_execute.py run --policy OUT/policy --jump 0.45 0 --jump 0.55 0 --conds challenging
    dilc_execute.py run --policy OUT/policy --jump 0.45 0 --conds dr:16 --compare
    dilc_execute.py dataset --bank OUT/bank.json --runs OUT/ilc/*/ --out OUT/ilc_episodes.npz
    dilc_execute.py serve --bank OUT/bank.json      # stdin/stdout, spawned by dilc_train.py

The policy (trained by dilc_train.py: SAC with Gurumurthy et al.'s value-gradient loss,
the SRB model's A_t, B_t as the approximate simulator) replaces the ILC's feedforward
forces with state feedback: at the start of every TO sample k < Nc it reads the SRB state
x_k from the mocap frames and joints (a causal fit, `Estimator` -- the same function
replays the ILC's recorded trials into training data, so both see identical states) and
sets that sample's contact forces

    U[k] = u_ref(g)[k] + a_scale * pi(obs_k),   obs_k = [(x_k - x_ref(g)[k]) / sx, k, g]

held over the sample like the ILC's. Everything else is the ILC controller, untouched:
TO joint torques and joint PD, level-feet legs, the balance landing controller.

The reference of a goal g the ILC never flew is interpolated from the plans of the goals
it did (`GoalBank`: piecewise-linear along a line of goals, barycentric in a triangle of
them) -- x_ref, u_ref, the joint profile, TO torque and lever arms. Nothing is re-planned
and no trial is learned: that is the zero-shot execution.

Every rollout runs against the asynchronous reality model (actuation delay and jitter,
dropped commands, stale joint readings, 240 Hz delayed noisy mocap, raw foot sensors)
and, as named conditions or randomized (`sample_condition`), the model errors that were
hard for the ILC: heavier/lighter robot, CoM offset, weak motors, the torque-speed curve
with battery sag, low friction, soft ground, payload, joint friction, long delays.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shlex
import struct
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# conditions ------------------------------------------------------------------------------
ASYNC = ("--act-delay 0.002 --act-jitter 0.003 --cmd-drop 0.02 --state-jitter 0.004 "
         "--pose-rate 240 --pose-delay 0.006 --pose-noise 0.0005 0.003 "
         "--param pose_latency:=0.006 --joint-noise 0.002 0.3 --foot-sensor 40 0.3 5")
# experiments/go1/robust/conds_async.txt, less the seed (given per rollout)
NAMED = {
    "nominal": ASYNC,
    "heavy15": ASYNC + " --mass-scale 1.15",
    "light10": ASYNC + " --mass-scale 0.9",
    "comfwd": ASYNC + " --com-offset 0.02 0",
    "comback": ASYNC + " --com-offset -0.02 0",
    "weak85": ASYNC + " --motor-scale 0.85",
    "curve": ASYNC + " --motor-curve",
    "curvesag": ASYNC + " --motor-curve --motor-speed-scale 0.85",
    "mu05": ASYNC + " --friction 0.5",
    "jfric": ASYNC + " --joint-friction 0.3",
    "hardgnd": ASYNC + " --ground 2e4 3e3",
    "softgnd": ASYNC + " --ground 2e3 5e2",
    "payload2": ASYNC + " --payload 2.0",
    "delay10": ASYNC.replace("--act-delay 0.002", "--act-delay 0.010"),
    "mocapbad": ("--act-delay 0.002 --act-jitter 0.003 --cmd-drop 0.02 --state-jitter 0.004 "
                 "--joint-noise 0.002 0.3 --foot-sensor 40 0.3 5 --pose-rate 120 "
                 "--pose-delay 0.015 --pose-noise 0.002 0.01 --param pose_latency:=0.015"),
    "real_s1": ASYNC + " --mass-scale 1.05 --com-offset 0.01 0 --motor-curve --joint-friction 0.2",
    # perturbed starts on the nominal robot (the plan and the policy are not told): a block
    # under the front feet, and the stance's planar joints offset [F thigh, F calf, R thigh,
    # R calf] (crouched, taller, rear crouched = nose up, front crouched = nose down)
    "blk15": ASYNC + " --step 0.13 0.225 0.015",
    "blk2": ASYNC + " --step 0.13 0.225 0.02",
    "crouch": ASYNC + " --param stand_offset:=[0.1,-0.2,0.1,-0.2]",
    "tall": ASYNC + " --param stand_offset:=[-0.1,0.2,-0.1,0.2]",
    "noseup": ASYNC + " --param stand_offset:=[0.0,0.0,0.1,-0.2]",
    "nosedn": ASYNC + " --param stand_offset:=[0.1,-0.2,0.0,0.0]",
    "real_s2": ASYNC + (" --mass-scale 0.95 --com-offset -0.01 0 --motor-curve "
                        "--motor-speed-scale 0.9 --joint-friction 0.2"),
}
# the conditions the ILC struggled with in the async grid (runs still falling at the end,
# boxes left short), with the nominal one
CHALLENGING = ["nominal", "heavy15", "weak85", "curvesag", "payload2", "softgnd", "mu05",
               "real_s1"]


def sample_condition(rng) -> tuple[str, str]:
    """A randomized reality: the async model with its delays and mocap drawn too, and every
    model error the ILC found hard drawn at once, each over the named conditions' range."""
    mocap_bad = rng.random() < 0.15
    delay = rng.uniform(0.001, 0.010)
    parts = [f"--act-delay {delay:.4f} --act-jitter 0.003 --cmd-drop {rng.uniform(0, 0.04):.3f}",
             "--state-jitter 0.004 --joint-noise 0.002 0.3 --foot-sensor 40 0.3 5"]
    if mocap_bad:
        parts.append("--pose-rate 120 --pose-delay 0.015 --pose-noise 0.002 0.01 "
                     "--param pose_latency:=0.015")
    else:
        pd = rng.uniform(0.004, 0.010)
        parts.append(f"--pose-rate 240 --pose-delay {pd:.4f} --pose-noise 0.0005 0.003 "
                     f"--param pose_latency:={pd:.4f}")
    parts.append(f"--mass-scale {rng.uniform(0.9, 1.15):.3f}")
    parts.append(f"--com-offset {rng.uniform(-0.02, 0.02):.4f} 0")
    parts.append(f"--motor-scale {rng.uniform(0.85, 1.0):.3f}")
    if rng.random() < 0.5:
        parts.append(f"--motor-curve --motor-speed-scale {rng.uniform(0.85, 1.0):.3f}")
    if rng.random() < 0.5:
        parts.append(f"--friction {rng.uniform(0.5, 0.8):.2f}")
    parts.append(f"--joint-friction {rng.uniform(0.0, 0.3):.2f}")
    r = rng.random()
    if r < 0.25:
        parts.append(f"--ground {10 ** rng.uniform(np.log10(2e3), np.log10(2e4)):.0f} "
                     f"{10 ** rng.uniform(np.log10(5e2), np.log10(3e3)):.0f}")
    if rng.random() < 0.3:
        parts.append(f"--payload {rng.uniform(0.0, 2.0):.2f}")
    return "dr", " ".join(parts)



# randomized realities drawn once and frozen (seed 2026): "real-like" robots that keep their
# name across exploration and evaluation (dr:N draws anew with every seed)
_rng_real = np.random.default_rng(2026)
for _i in range(6):
    NAMED[f"real_r{_i}"] = sample_condition(_rng_real)[1]
del _rng_real, _i

def conditions(spec: str, seed: int = 0) -> list[tuple[str, str]]:
    """'challenging', 'all', 'dr:N' (N randomized), a comma list of names, or a file of
    name|args lines (conds_async.txt)."""
    if spec.startswith("dr:"):
        rng = np.random.default_rng(seed + 7919)
        return [(f"dr{i}", sample_condition(rng)[1]) for i in range(int(spec[3:]))]
    if os.path.isfile(spec):
        out = []
        for line in open(spec):
            if line.strip() and "|" in line:
                name, _, a = line.strip().partition("|")
                out.append((name, " ".join(t for t in a.split()
                                           if not t.startswith("--seed")).replace("  ", " ")))
        return out
    names = CHALLENGING if spec == "challenging" else list(NAMED) if spec == "all" \
        else spec.split(",")
    return [(n, NAMED[n]) for n in names]


def strip_seed(args: str) -> str:
    toks = shlex.split(args)
    out, skip = [], False
    for t in toks:
        if skip:
            skip = False
            continue
        if t == "--seed":
            skip = True
            continue
        out.append(t)
    return " ".join(out)


# goals and references --------------------------------------------------------------------
class GoalBank:
    """The ILC'd goals' TO plans, the policy's normalization, and references for any goal
    by interpolating the plans. `bank` is dilc_train.py's bank.json."""

    SKIP = ("config",)

    def __init__(self, bank: dict):
        self.cfg = bank
        self.plans = bank["plans"]
        self.refs = [dict(np.load(p["path"])) for p in self.plans]
        self.goals = np.array([p["goal"] for p in self.plans], float)
        c0 = json.loads(str(self.refs[0]["config"]))
        self.base_config = c0
        self.Ndc, self.Nsc, self.Nfl = c0["phases"]
        self.Nc, self.N = self.Ndc + self.Nsc, self.Ndc + self.Nsc + self.Nfl
        self.dt = float(c0["dt"])
        for r in self.refs[1:]:
            c = json.loads(str(r["config"]))
            if c["phases"] != c0["phases"] or c["dt"] != c0["dt"]:
                raise ValueError("every plan in the bank needs the same phases and dt")
        self.boxes = [p.get("box") for p in self.plans]
        if any(b is None for b in self.boxes) and not all(b is None for b in self.boxes):
            raise ValueError("the bank mixes flat and box plans")
        self.sx = np.asarray(bank["sx"], float)
        self.a_scale = np.asarray(bank["a_scale"], float)
        self.g_center = np.asarray(bank["g_center"], float)
        self.g_scale = np.asarray(bank["g_scale"], float)
        self.swing_mask = np.zeros((self.Nc, 4), bool)
        self.swing_mask[self.Ndc:, 0:2] = True

    # interpolation weights over the plans
    def weights(self, g) -> tuple[np.ndarray, bool]:
        """Weights of the plans for goal g (sum 1), and whether g lies outside their hull."""
        g = np.asarray(g, float)
        G = self.goals
        K = len(G)
        if K == 1:
            return np.ones(1), bool(np.abs(g - G[0]).max() > 1e-9)
        D = G - G[0]
        rank = np.linalg.matrix_rank(D, tol=1e-6)
        w = np.zeros(K)
        if rank == 1:                     # goals on a line: piecewise linear along it
            d = D[np.argmax(np.linalg.norm(D, axis=1))]
            d = d / np.linalg.norm(d)
            s = D @ d
            sg = float((g - G[0]) @ d)
            order = np.argsort(s)
            ss = s[order]
            j = int(np.clip(np.searchsorted(ss, sg) - 1, 0, K - 2))
            lam = (sg - ss[j]) / (ss[j + 1] - ss[j])
            w[order[j]], w[order[j + 1]] = 1 - lam, lam
            off = np.linalg.norm((g - G[0]) - sg * d) > 1e-6
            return w, bool(lam < -1e-9 or lam > 1 + 1e-9 or off)
        best = None                       # a triangle of goals: barycentric
        import itertools
        for tri in itertools.combinations(range(K), 3):
            P = G[list(tri)]
            M = np.column_stack([P[1] - P[0], P[2] - P[0]])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            l12 = np.linalg.solve(M, g - P[0])
            lam = np.array([1 - l12.sum(), *l12])
            key = (-min(lam.min(), 0.0), abs(np.linalg.det(M)))
            if best is None or key < best[0]:
                best = (key, tri, lam)
        _, tri, lam = best
        w[list(tri)] = lam
        return w, bool(lam.min() < -1e-9)

    def weights_jac(self, g) -> np.ndarray:
        """d weights / d g (K, 2). Between goals on a line, the slope of the segment g is on
        (at a goal between two segments, their mean); in a triangle, the barycentric map."""
        g = np.asarray(g, float)
        G = self.goals
        K = len(G)
        J = np.zeros((K, 2))
        if K == 1:
            return J
        D = G - G[0]
        if np.linalg.matrix_rank(D, tol=1e-6) == 1:
            d = D[np.argmax(np.linalg.norm(D, axis=1))]
            d = d / np.linalg.norm(d)
            s = D @ d
            sg = float((g - G[0]) @ d)
            order = np.argsort(s)
            ss = s[order]
            segs = []
            for j in range(K - 1):
                if ss[j] - 1e-9 <= sg <= ss[j + 1] + 1e-9:
                    segs.append(j)
            if not segs:
                segs = [0 if sg < ss[0] else K - 2]
            for j in segs:
                h = ss[j + 1] - ss[j]
                J[order[j]] -= d / h / len(segs)
                J[order[j + 1]] += d / h / len(segs)
            return J
        w, _ = self.weights(g)
        tri = [i for i in range(K) if abs(w[i]) > 1e-12][:3]
        if len(tri) < 3:                   # on an edge or a vertex: any triangle holding it
            tri = sorted(set(tri) | set(range(K)))[:3]
        P = G[tri]
        Minv = np.linalg.inv(np.column_stack([P[1] - P[0], P[2] - P[0]]))
        J[tri[1]], J[tri[2]] = Minv[0], Minv[1]
        J[tri[0]] = -Minv[0] - Minv[1]
        return J

    def reference_jac(self, g) -> tuple[np.ndarray, np.ndarray]:
        """d x_ref / d g (N+1, 6, 2) and d u_ref / d g (Nc, 4, 2) of the interpolated plan."""
        J = self.weights_jac(g)
        dx = sum(np.asarray(r["x_ref"], float)[..., None] * J[i] for i, r in enumerate(self.refs))
        du = sum(np.asarray(r["u_ref"], float)[..., None] * J[i] for i, r in enumerate(self.refs))
        return dx, du

    def project(self, g, margin=0.02):
        """g moved onto the bank's goals line (goals in a line), at most `margin` m beyond
        its ends (the labels' u* = U + S (g_label - g_achieved) carries what the move off
        the line changes, e.g. a landing a little low). Goals off a line: g itself."""
        g = np.asarray(g, float)
        G = self.goals
        D = G - G[0]
        if len(G) == 1 or np.linalg.matrix_rank(D, tol=1e-6) > 1:
            return g
        d = D[np.argmax(np.linalg.norm(D, axis=1))]
        d = d / np.linalg.norm(d)
        s = D @ d
        t = float(np.clip((g - G[0]) @ d, s.min() - margin, s.max() + margin))
        return G[0] + t * d

    @property
    def n_hist(self) -> int:
        return int(self.cfg.get("hist", 0))

    @property
    def prev_action(self) -> bool:
        return bool(self.cfg.get("prev_action", False))

    def norm_action(self, u, k, u_ref):
        """Forces -> the policy's action at sample k, as stored for training."""
        a = (np.asarray(u, float) - u_ref[k]) / self.a_scale
        a[self.swing_mask[k]] = 0.0
        return np.clip(a, -0.999, 0.999)

    def obs_at(self, X, acts, k, g, x_ref) -> np.ndarray:
        """Observation at sample k from the states X[0..k] and actions acts[0..k-1]:
        [state error (6), time, goal (2), then the n_hist previous state errors and the
        previous action if the bank asks for them (zeros before the jump)]."""
        err = lambda j: np.clip((np.asarray(X[j]) - x_ref[j]) / self.sx, -20.0, 20.0) \
            if j >= 0 else np.zeros(6)
        parts = [err(k), [2.0 * k / self.Nc - 1.0],
                 (np.asarray(g, float) - self.g_center) / self.g_scale]
        parts += [err(k - j) for j in range(1, self.n_hist + 1)]
        if self.prev_action:
            parts.append(np.asarray(acts[k - 1], float) if k >= 1 else np.zeros(4))
        return np.concatenate(parts)

    def reference(self, g) -> tuple[dict, dict | None, bool]:
        """(reference arrays as JumpILC.save_reference writes them, box, extrapolated)."""
        w, extrap = self.weights(g)
        out = {}
        for key in self.refs[0]:
            if key in self.SKIP:
                continue
            vals = [np.asarray(r[key]) for r in self.refs]
            if key == "info_success":
                out[key] = np.array(all(bool(v) for v in vals))
            elif key == "info_min_clearance":
                out[key] = np.array(min(float(v) for v in vals))
            elif vals[0].dtype.kind in "fi":
                out[key] = sum(wi * v.astype(float) for wi, v in zip(w, vals))
            else:
                out[key] = vals[0]
        box = None
        if self.boxes[0] is not None:
            box = {k: float(sum(wi * b[k] for wi, b in zip(w, self.boxes)))
                   for k in ("x_front", "height")}
        cfg = dict(self.base_config, jump=[float(v) for v in g], box=box)
        out["config"] = json.dumps(cfg)
        return out, box, extrap



# state estimate --------------------------------------------------------------------------
class Estimator:
    """
    The SRB state [c_x, c_z, theta, cd_x, cd_z, theta_d] at the start of a TO sample, from
    what the controller has seen by then: each mocap frame (dated its latency back, with
    the joints interpolated to that time) gives the whole-body CoM and pitch in the TO's
    frame, as JumpILC's trial log does; a line through the frames of the last `window` s
    gives the state now (extrapolated over the latency) and its rate. Causal, and fed tick
    by tick, so a recorded trial replays into exactly what the live policy saw.
    """

    def __init__(self, fb, x_ref0, origin, yaw, latency, window):
        from ilc_jump_sim import collapse, planar_pitch
        self._collapse, self._pitch = collapse, planar_pitch
        self.fb, self.x0 = fb, np.asarray(x_ref0[:2], float)
        self.origin, self.c, self.s = np.asarray(origin, float), math.cos(yaw), math.sin(yaw)
        self.lat, self.w = float(latency), float(window)
        self.tt, self.qq, self.frames, self.shift = [], [], [], None

    def push(self, t, pos, quat, q12, new):
        self.tt.append(float(t))
        self.qq.append(self._collapse(np.asarray(q12)))
        if not (new or len(self.tt) == 1):
            return
        tf = t - self.lat
        ta, qa = np.asarray(self.tt), np.asarray(self.qq)
        qf = np.array([np.interp(tf, ta, qa[:, i]) for i in range(4)])
        rel = np.asarray(pos) - self.origin
        th = self._pitch(quat)
        com = self.fb.com(np.concatenate([[self.c * rel[0] + self.s * rel[1], rel[2], th],
                                          qf])).full().ravel()
        if self.shift is None:            # start where the reference starts
            self.shift = self.x0 - com
        com = com + self.shift
        self.frames.append((tf, com[0], com[1], th))

    def estimate(self, t_now) -> np.ndarray:
        F = np.asarray(self.frames)
        m = F[:, 0] >= t_now - self.w - 1e-9
        sel = F[m] if m.sum() >= 2 else F[-2:]
        if len(sel) < 2:
            return np.array([sel[-1, 1], sel[-1, 2], sel[-1, 3], 0.0, 0.0, 0.0])
        h = sel[:, 0] - t_now
        fit = [np.polyfit(h, sel[:, i], 1) for i in (1, 2, 3)]
        return np.array([fit[0][1], fit[1][1], fit[2][1], fit[0][0], fit[1][0], fit[2][0]])


def replay_states(fb, x_ref, rec, dt, Nc, latency, window) -> np.ndarray:
    """A recorded trial's (rec_*) states at samples 0..Nc, as Estimator saw them live."""
    from ilc_jump_sim import yaw_of
    t, pos, quat, q, new = rec["t"], rec["pos"], rec["quat"], rec["q"], rec["pose_new"]
    est = Estimator(fb, x_ref[0], pos[0], yaw_of(quat[0]), latency, window)
    X, last = [], -1
    for i in range(len(t)):
        est.push(t[i], pos[i], quat[i], q[i], bool(new[i]))
        k = int(t[i] / dt + 1e-6)
        if last < k <= Nc:
            last = k
            X.append(est.estimate(t[i]))
            if k == Nc:
                break
    return np.asarray(X)


# actor (numpy: no torch in the robot's container) ----------------------------------------
class NumpyActor:
    """dilc_train.py's actor exported as npz: MLP (relu) -> tanh-Gaussian mean, log std."""

    def __init__(self, weights: dict):
        self.W = [np.asarray(weights[f"W{i}"], np.float64) for i in range(weights["n_hidden"] + 0)]
        self.b = [np.asarray(weights[f"b{i}"], np.float64) for i in range(weights["n_hidden"] + 0)]
        self.W_mu, self.b_mu = np.asarray(weights["W_mu"], float), np.asarray(weights["b_mu"], float)
        self.W_ls, self.b_ls = np.asarray(weights["W_ls"], float), np.asarray(weights["b_ls"], float)
        # a FADA-style planner-IDM (baseline, ilc_mjx fada_train.py): the planner's predicted next error
        # (P_*) joins the IDM's (the MLP above) input
        self.P = None
        if "P_W0" in weights:
            self.P = ([np.asarray(weights[f"P_W{i}"], float) for i in range(2)],
                      [np.asarray(weights[f"P_b{i}"], float) for i in range(2)],
                      np.asarray(weights["P_Wo"], float), np.asarray(weights["P_bo"], float))

    @classmethod
    def load(cls, path):
        d = dict(np.load(path))
        d["n_hidden"] = int(d["n_hidden"])
        return cls(d)

    def __call__(self, obs, rng=None):
        h = np.asarray(obs, float)
        if self.P is not None:
            p = h
            for W, b in zip(*self.P[:2]):
                p = np.maximum(W @ p + b, 0.0)
            h = np.concatenate([h, self.P[2] @ p + self.P[3]])
        for W, b in zip(self.W, self.b):
            h = np.maximum(W @ h + b, 0.0)
        mu = self.W_mu @ h + self.b_mu
        if rng is None:
            return np.tanh(mu)
        ls = np.clip(self.W_ls @ h + self.b_ls, -5.0, 1.0)
        return np.tanh(mu + np.exp(ls) * rng.standard_normal(mu.shape))


class Ensemble:
    def __init__(self, actors):
        self.actors = actors

    def __call__(self, obs, rng=None):
        return np.mean([a(obs, rng) for a in self.actors], axis=0)


# the simulator's own Jacobians, by finite differences ----------------------------------------
class FDJac:
    """
    The value-gradient update's Jacobians from the simulator itself, by finite differences
    (Gurumurthy et al.'s VG-SAC-FD): nothing about the task is assumed beyond which state
    the policy's observation measures. At the first tick of every contact sample k the
    jump's process forks once per perturbation; each copy flies on to sample k+1 and reports
    the state there, and the jump itself, unperturbed, is the nominal. A fork carries
    everything over as it is -- MuJoCo, the asynchronous reality (commands in flight with
    their delay and jitter, dropped ones, the mocap and joint-reading buffers, and its random
    generator: the copies see the same drops, jitter and noise, common random numbers), the
    controller and the estimator -- so the differences are those of the asynchronous
    simulator, not of a synchronous stand-in. The copies run one after another (one core
    per jump).

      B[:, j]  the policy's action j moved by eps_a (normalized units; the swing legs'
               entries are skipped, their forces are zero), everything else as flown
      A        the state moved by eps_s x sx along each SRB coordinate, the action held: the
               generalized positions (or velocities) changed by the smallest step in the
               mass metric that moves the CoM x, z and the trunk pitch (or their rates) that
               much while every foot in contact stays put; A = dS_{k+1} dS_k^-1, dS_k the
               change measured after the step

    The state is the robot's true SRB state in the jump's frame (whole-body CoM x, z, trunk
    pitch and their rates, from MuJoCo), not the policy's estimate of it: the estimator's
    latency would hide most of a one-sample change. The true state's Jacobians stand in for
    the observation's (approximate Jacobians, their Sec. 4.3).
    """

    def __init__(self, eps_a=0.05, eps_s=0.1, timeout=30.0, branch=True):
        self.eps_a, self.eps_s, self.timeout = float(eps_a), float(eps_s), float(timeout)
        self.branch = branch                  # False: only the true state per sample (S)
        self.S, self.res = {}, {}             # nominal state per sample; copies' reports per sample
        self.child = None                     # in a copy: (pipe, its state at k)
        self.sd = None

    def _data(self, sim):
        import mujoco
        if self.sd is None:
            self.sd = mujoco.MjData(sim.model)
        self.sd.qpos[:], self.sd.qvel[:] = sim.data.qpos, sim.data.qvel
        return self.sd

    def state(self, node) -> np.ndarray:
        """[c_x, c_z, theta, cd_x, cd_z, theta_d], true, in the jump's frame (scratch data:
        the simulation's own is left untouched)."""
        import mujoco
        from ilc_jump_sim import planar_pitch
        sim, fr = node.lockstep_sim, node.frame
        m, d = sim.model, self._data(sim)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_subtreeVel(m, d)
        b = sim.base_body_id
        c, s = math.cos(fr["yaw"]), math.sin(fr["yaw"])
        p, v = d.subtree_com[b] - fr["origin"], d.subtree_linvel[b]
        th = planar_pitch(d.qpos[3:7])
        q2, h = d.qpos.copy(), 1e-5
        mujoco.mj_integratePos(m, q2, d.qvel, h)
        return np.array([c * p[0] + s * p[1], p[2], th, c * v[0] + s * v[1], v[2],
                         (planar_pitch(q2[3:7]) - th) / h])

    def _perturb(self, node, i, sx, contacts):
        """State coordinate i moved by eps_s sx[i], the feet in contact held."""
        import mujoco
        sim, fr = node.lockstep_sim, node.frame
        m, d = sim.model, self._data(sim)
        mujoco.mj_fwdPosition(m, d)
        nv, b = m.nv, sim.base_body_id
        c, s = math.cos(fr["yaw"]), math.sin(fr["yaw"])
        jc, jp, jr = np.zeros((3, nv)), np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(m, d, jc, b)
        mujoco.mj_jacBody(m, d, jp, jr, b)
        # theta is nose-up: minus the rotation about the jump frame's y axis
        rows = [c * jc[0] + s * jc[1], jc[2], s * jr[0] - c * jr[1]]
        for leg, f in enumerate(contacts):
            if f > 1.0:
                g = sim.foot_geom_ids[leg]
                jf, jfr = np.zeros((3, nv)), np.zeros((3, nv))
                mujoco.mj_jac(m, d, jf, jfr, d.geom_xpos[g], m.geom_bodyid[g])
                rows += list(jf)
        C = np.array(rows)
        rhs = np.zeros(len(C))
        rhs[i % 3] = self.eps_s * sx[i]
        MiC = np.zeros_like(C)                       # M^-1 C' (rows), M factored by mj_fwdPosition
        mujoco.mj_solveM(m, d, MiC, C)
        MiCt = MiC.T
        G = C @ MiCt
        dv = MiCt @ np.linalg.solve(G + 1e-10 * np.trace(G) / len(G) * np.eye(len(G)), rhs)
        if i < 3:
            mujoco.mj_integratePos(m, sim.data.qpos, dv, 1.0)
        else:
            sim.data.qvel[:] += dv

    def sample(self, node, k, bank, a, fly):
        """Sample k's first tick in the nominal jump: its state, then (k < Nc) a copy per
        perturbation, each run to sample k+1 before the next starts. Returns the forces this
        process flies -- the nominal's, or in a copy its perturbed ones."""
        import select
        import signal
        s = self.state(node)
        self.S[k] = s
        if a is None:
            return None
        u = fly(a)
        if not self.branch:
            return u
        jobs = [("a", j) for j in range(len(a)) if not bank.swing_mask[k, j]] + \
               [("s", i) for i in range(6)]
        contacts = node.lockstep_sim.foot_normal_forces()
        res = {}
        for kind, j in jobs:
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:                             # the copy
                os.close(r)
                try:
                    if kind == "a":
                        da = np.zeros(len(a))
                        da[j] = self.eps_a
                        self.child = (w, s)
                        return fly(a + da)
                    self._perturb(node, j, bank.sx, contacts)
                    self.child = (w, self.state(node))
                    return u
                except BaseException as err:
                    print(f"FDJac copy ({kind}{j}, k={k}) failed: {err!r}", file=sys.stderr, flush=True)
                    os._exit(1)
            os.close(w)
            buf = b""
            if select.select([r], [], [], self.timeout)[0]:
                buf = os.read(r, 1024)
            if len(buf) != 96:
                os.kill(pid, signal.SIGKILL)
            os.close(r)
            os.waitpid(pid, 0)
            res[(kind, j)] = np.frombuffer(buf, np.float64).reshape(2, 6) if len(buf) == 96 else None
        self.res[k] = res
        return u

    def report(self, node):
        """In a copy, at sample k+1: the states at k (after its change) and now; then it ends."""
        w, s0 = self.child
        os.write(w, np.concatenate([s0, self.state(node)]).astype(np.float64).tobytes())
        os._exit(0)

    def jacobians(self, bank):
        """A (Nc, 6, 6) in the SRB state's units, B (Nc, 6, 4) per unit of normalized action;
        NaN where a sample's copies did not all report."""
        Nc = bank.Nc
        A, B = np.full((Nc, 6, 6), np.nan), np.full((Nc, 6, 4), np.nan)
        for k, res in self.res.items():
            if k + 1 not in self.S or any(v is None for v in res.values()):
                continue
            s0, s1 = self.S[k], self.S[k + 1]
            B[k] = 0.0
            for (kind, j), v in res.items():
                if kind == "a":
                    B[k][:, j] = (v[1] - s1) / self.eps_a
            dS0 = np.stack([res[("s", i)][0] - s0 for i in range(6)], 1)
            dS1 = np.stack([res[("s", i)][1] - s1 for i in range(6)], 1)
            if np.linalg.cond(dS0 / (self.eps_s * bank.sx)[:, None]) < 1e3:
                A[k] = dS1 @ np.linalg.inv(dS0)
        return A, B


# the force policy inside the controller --------------------------------------------------
class Driver:
    """IlcJumpNode.force_policy: the state estimate at every sample, then the forces --
    from the actor (mode "policy") or a fixed force sequence (mode "fixed": baselines and
    ILC-style feedforward), clipped into the friction cones and force limits."""

    def __init__(self, bank: GoalBank, g, ref, actor=None, U_fixed=None, rng=None,
                 window=0.025, lqr=None, fd=None):
        self.bank, self.g, self.ref = bank, np.asarray(g, float), ref
        self.actor, self.U_fixed, self.rng, self.window = actor, U_fixed, rng, window
        self.lqr = lqr                       # (U_ff, X_nom, K): time-varying LQR (baseline)
        self.fd = fd                         # FDJac: the simulator's Jacobians along the jump
        self.est = None
        self.X, self.U, self.A, self.n_pushed = [], [], [], 0

    def _clip(self, u, k, p):
        u = u.copy()
        for pair in range(2):
            fx, fz = u[2 * pair], u[2 * pair + 1]
            fz = float(np.clip(fz, p["fmin"], p["fmax"]))
            u[2 * pair], u[2 * pair + 1] = float(np.clip(fx, -p["mu"] * fz, p["mu"] * fz)), fz
        u[self.bank.swing_mask[k]] = 0.0
        return u

    def __call__(self, node, k, q, dq):
        ilc = node.ilc
        if self.fd is not None and self.fd.child is not None:
            self.fd.report(node)                     # a perturbed copy: its state, and it ends
        if self.est is None:
            self.est = Estimator(node.fb, ilc.x_ref[0], node.frame["origin"], node.frame["yaw"],
                                 node.pose_latency, self.window)
        rec = node.rec
        n = len(rec["t"])
        for i in range(self.n_pushed, n):            # recorded ticks not yet seen
            self.est.push(rec["t"][i], rec["pos"][i], rec["quat"][i], rec["q"][i],
                          bool(rec["pose_new"][i]))
        t_now = round((k * ilc.dt) / node.tick) * node.tick
        self.est.push(t_now, node.pose[1], node.pose[2], q, bool(node.pose_fresh))
        self.n_pushed = n + 1                        # this tick is recorded next, as pushed
        x = self.est.estimate(t_now)
        self.X.append(x)
        if k >= ilc.Nc:
            if self.fd is not None:
                self.fd.sample(node, k, self.bank, None, None)
            return None
        u_ref = self.ref["u_ref"][k]
        if self.actor is not None:
            a = self.actor(self.bank.obs_at(self.X, self.A, k, self.g, self.ref["x_ref"]),
                           self.rng)
            u = u_ref + self.bank.a_scale * a
        else:
            u = np.asarray(self.U_fixed[k], float)
            if self.lqr is not None:          # feedback, with the policy's authority
                U_ff, X_nom, K = self.lqr[:3]
                du = -K[k] @ (x - X_nom[k])
                u = U_ff[k] + np.clip(du, -self.bank.a_scale, self.bank.a_scale)
        if self.fd is not None:               # forks here; a copy returns its perturbed forces
            a0 = (u - u_ref) / self.bank.a_scale
            u = self.fd.sample(node, k, self.bank, a0,
                               lambda a_: self._clip(u_ref + self.bank.a_scale * a_, k,
                                                     ilc.mdc_params))
        else:
            u = self._clip(u, k, ilc.mdc_params)
        self.U.append(u)
        self.A.append(self.bank.norm_action(u, k, self.ref["u_ref"]))
        return u


# one jump ------------------------------------------------------------------------------
_REF_DIR = tempfile.mkdtemp(prefix="dilc_ref_")


def run_jump(bank: GoalBank, g, cond: str, seed: int, driver_kw: dict, land_time=1.5,
             reference=None, menagerie_root="", reference_file=None):
    """One jump of goal g against reality `cond`, with Driver(**driver_kw) flying the
    contact forces. Returns (driver, trial log, rec, fell, reference, extrapolated)."""
    import ilc_jump_lockstep as lockstep
    from ilc_jump_sim import IlcJumpNode

    if reference_file:
        ref = dict(np.load(reference_file))
        c = json.loads(str(ref["config"]))
        box, extrap, ref_path = c.get("box"), False, reference_file
    else:
        ref, box, extrap = reference if reference is not None else bank.reference(g)
        ref_path = os.path.join(_REF_DIR, f"ref_{os.getpid()}.npz")
        np.savez(ref_path, **ref)
    cfg = json.loads(str(ref["config"]))
    argv = ["--robot", cfg["robot"], "--jump", str(g[0]), str(g[1]), "--max-trials", "1",
            "--reference-file", ref_path,
            "--param", "phases:=[" + ",".join(str(int(v)) for v in cfg["phases"]) + "]",
            "--param", f"margin:={cfg['margin']}", "--param", f"land_time:={land_time}"]
    if menagerie_root:
        argv += ["--menagerie-root", menagerie_root]
    if box is not None:
        argv += ["--box", str(box["x_front"]), str(box["height"])]
    argv += shlex.split(strip_seed(cond)) + ["--seed", str(int(seed))]
    args = lockstep.build_parser().parse_args(argv)

    out = {}

    class PolicyNode(IlcJumpNode):
        def __init__(self):
            super().__init__()
            if not np.allclose(self.ilc.x_ref, ref["x_ref"]):
                raise RuntimeError("the node did not load the interpolated reference")
            out["driver"] = Driver(bank, g, ref, **driver_kw)
            self.force_policy = out["driver"]

        def _update_worker(self):                # no learning: keep the trial and stop
            try:
                log, rec, _ = self._trial_log()
                out.update(log=log, rec={k: np.asarray(v) for k, v in rec.items()},
                           fell=bool(self.fell))
                self.last_result = dict(converged=True)
            except Exception as err:
                self.get_logger().error(f"trial log failed: {err!r}")
                self.last_result = dict(converged=False, failed=True)

    lockstep.IlcJumpNode = PolicyNode
    lockstep.run(args)
    fd = driver_kw.get("fd")
    if fd is not None and fd.child is not None:   # a copy that never reached its next sample
        os._exit(1)
    if "log" not in out:
        raise RuntimeError("the jump produced no trial log")
    return out["driver"], out["log"], out["rec"], out["fell"], ref, extrap


# episodes: what the trainer learns from ----------------------------------------------------
class EpisodeMaker:
    """Turns a flown jump into an MDP episode on the TO grid, samples 0..Nc-1: normalized
    observations and actions, the SRB model's Jacobians there (the value-gradient update's
    approximate simulator, eq. 12 of Gurumurthy et al.), and the ILC's stage costs split
    by stage, with their gradients in the next state -- the trainer weighs the stages."""

    def __init__(self, bank: GoalBank, menagerie_root=""):
        from ilc_quad.ilc_gen import PlanarQuadModel, SRBModel
        from ilc_quad.sim_quad_model import QuadModel, default_menagerie_root
        self.bank = bank
        qm = QuadModel(bank.base_config["robot"], menagerie_root or default_menagerie_root())
        self.fb = PlanarQuadModel(qm)
        self.srb = SRBModel(mass=self.fb.total_mass,
                            inertia=float(self.fb.pitch_inertia(self.fb.standing_state())))
        self.qe = np.asarray(bank.cfg["qe"], float)
        self.r_scale = float(bank.cfg["r_scale"])

    def make(self, g, ref, X, U, logX, fell, rec=None, jac=None) -> dict:
        """jac: (A, B) of the simulator along the jump (FDJac.jacobians), in place of the
        SRB model's wherever a sample has them."""
        b = self.bank
        Nc, Ndc, N, dt = b.Nc, b.Ndc, b.N, b.dt
        x_ref, u_ref = ref["x_ref"], ref["u_ref"]
        R1, R2 = ref["info_R1"], ref["info_R2"]
        D, asc = b.sx, b.a_scale
        X, U = np.asarray(X, float), np.asarray(U, float)
        act = np.array([b.norm_action(U[k], k, u_ref) for k in range(Nc)])
        obs = np.array([b.obs_at(X, act, k, g, x_ref) for k in range(Nc + 1)])
        gs = b.g_scale
        dxr, dur = b.reference_jac(g)
        A = np.zeros((Nc, 6, 6))
        B = np.zeros((Nc, 6, 4))
        Cg = np.zeros((Nc, 6, 2))          # d (next state error) / d goal, state error and action held
        for k in range(Nc):
            Ak, Bk = self.srb.f_lin(X[k], U[k], R1[k], R2[k], dt)
            Ak, Bk = Ak.full(), Bk.full() * asc[None, :]       # B per unit of normalized action
            Bk[:, b.swing_mask[k]] = 0.0
            if jac is not None and np.isfinite(jac[0][k]).all() and np.isfinite(jac[1][k]).all():
                Ak, Bk = jac[0][k], jac[1][k]
            Ck = Ak @ dxr[k] + (Bk / asc[None, :]) @ dur[k] - dxr[k + 1]
            A[k] = Ak * D[None, :] / D[:, None]
            B[k] = Bk / D[:, None]
            Cg[k] = Ck / D[:, None] * gs[None, :]
        qe, rs = self.qe, self.r_scale
        rc, gn = np.zeros((Nc, 3)), np.zeros((Nc, 3, 6))
        # scored on the trial log's states (a centered fit, as the ILC scores a trial), not
        # the policy's causal estimate, which lags through the push-off
        for k in range(Nc):                          # Stage I rows 1..Nc, Stage II from Ndc
            e = logX[k + 1] - x_ref[k + 1]
            c, gx = rs * e @ (qe * e), 2 * rs * qe * e * D
            rc[k, 0], gn[k, 0] = c, gx
            if k + 1 >= Ndc:
                rc[k, 1], gn[k, 1] = c, gx
        # flight rows (Stage II) and the landing (Stage III), on the last transition through
        # the SRB model's ballistic flight map from the takeoff state
        Af = self.srb.f_lin(np.zeros(6), np.zeros(4), np.zeros(2), np.zeros(2), dt)[0].full()
        Phi = np.eye(6)
        target = x_ref[N].copy()
        target[:2] = x_ref[0, :2] + np.asarray(g, float)
        target[2] = 0.0
        dtarget = dxr[N].copy()
        dtarget[:2] = np.eye(2)
        dtarget[2] = 0.0
        gg = np.zeros((Nc, 3, 2))          # d stage cost / d goal, the next state error held
        for m in range(Nc + 1, N + 1):
            Phi = Af @ Phi
            e = logX[m] - x_ref[m]
            gx = 2 * rs * qe * e
            rc[-1, 1] += rs * e @ (qe * e)
            gn[-1, 1] += (Phi.T @ gx) * D
            gg[-1, 1] += (gx @ (Phi @ dxr[Nc] - dxr[m])) * gs
        e3 = logX[N] - target
        gx = 2 * rs * qe * e3
        rc[-1, 2] = rs * e3 @ (qe * e3)
        gn[-1, 2] = (Phi.T @ gx) * D
        gg[-1, 2] = (gx @ (Phi @ dxr[Nc] - dtarget)) * gs
        info = np.array([e3[0], e3[1], e3[2], float(fell), *self.stick(rec)])
        return dict(obs=obs.astype(np.float32), act=act.astype(np.float32),
                    A=A.astype(np.float32), B=B.astype(np.float32),
                    rc=rc.astype(np.float32), gn=gn.astype(np.float32),
                    Cg=Cg.astype(np.float32), gg=gg.astype(np.float32),
                    fell=np.float32(fell), goal=np.asarray(g, np.float32),
                    info=info.astype(np.float32), X=X.astype(np.float32),
                    U=U.astype(np.float32), logX=np.asarray(logX, np.float32))

    def ilc_labels(self, plan_ref, U, logX, g_label, label_ref) -> dict:
        """
        The ILC's own answer for this trial, as Sobolev labels for the policy (the framework:
        each trial's A, B regularize the network). Along the trial exactly as JumpILC._step
        linearizes it (its logged states, the flown forces, the plan's lever arms):

          G_N     the landing rows of the lifted G (d x_N / d U), the outcome metric
          S       (G_N' Qe G_N + Qu)^-1 G_N' Qe E: the Stage III QP's force change per goal
                  change (Stage III Qe, Qu -- the deployment objective, whatever stage the
                  trial was flown in)
          K_t     the time-varying LQR along the trial (Riccati from Stage III Qe at the
                  landing, the stage tracking rows Qe x k_crun before it, R = k_rscale x Qu):
                  the state feedback the ILC's model implies at each sample

        and, for the label goal g_label (the commanded goal, or in hindsight the achieved one):
          target  u* = U + S (g_label - g_achieved): the forces for g_label (U itself when
                  g_label is where the trial landed; the ILC's next Stage III step when it is
                  the goal it was aiming at)
          Kx      d pi / d e the network should have: -K_t, in its normalized units
          Kg      d pi / d g at fixed e: with pi = u*(g) - K (x - x*(g)) along the trial,
                  S_t + K_t dx*_t/dg - K_t dx_ref_t/dg - du_ref_t/dg (x* moving with the forces
                  through G, the reference with the interpolated plan), normalized
        """
        from ilc_quad.ilc_gen import build_lifted_G
        b = self.bank
        Nc, N, dt, nu = b.Nc, b.N, b.dt, 4
        D, asc, gs = b.sx, b.a_scale, b.g_scale
        qe = self.qe
        qu = float(b.cfg.get("qu3", 1e-5))
        U = np.asarray(U, float)
        U_full = np.vstack([U, np.zeros((N - Nc, nu))])
        A_l, B_l = self.srb.linearize_along_trial(np.asarray(logX, float), U_full,
                                                  plan_ref["info_R1"], plan_ref["info_R2"], dt)
        for k in range(Nc):
            B_l[k][:, b.swing_mask[k]] = 0.0
        G = build_lifted_G(A_l, B_l, N, Nc, 6, nu)              # (N, Nc, 6, 4): x_{t+1} / u_j
        GN = G[N - 1].transpose(1, 0, 2).reshape(6, Nc * nu)     # (6, Nc*4), time-major
        if b.cfg.get("secant_C"):
            # the landing sensitivity as the ILC's own trial pairs measured it (secant_fit):
            # the SRB model's G_N plus the fitted correction, for the step S and the metric
            if not hasattr(self, "_secant_C"):
                self._secant_C = np.load(b.cfg["secant_C"])["C"]
            C = self._secant_C * np.repeat(~b.swing_mask.reshape(-1)[None, :], 6, 0)
            GN = GN + C
            if b.cfg.get("secant_full"):
                # and the rest of the labels (the feedback K, the co-states, d x*/d g) with
                # each sample's B corrected so its landing sensitivity moves to G_N + C: the
                # correction in the velocity rows (a force acts on the next state through
                # them), least squares in Qe through Phi_k = d x_N / d x_{k+1}
                W = np.diag(qe)
                Phi = np.eye(6)
                for k in range(N - 1, -1, -1):
                    if k < Nc:
                        P_ = Phi[:, 3:]
                        d = np.linalg.solve(P_.T @ W @ P_ + 1e-9 * np.eye(3),
                                            P_.T @ W @ C[:, nu * k:nu * k + nu])
                        B_l[k] = B_l[k].copy()
                        B_l[k][3:] += d
                    Phi = Phi @ A_l[k]
                G = build_lifted_G(A_l, B_l, N, Nc, 6, nu)
        Q = np.diag(qe)
        E = np.zeros((6, 2))
        E[0, 0] = E[1, 1] = 1.0
        S = np.linalg.solve(GN.T @ Q @ GN + qu * np.eye(Nc * nu), GN.T @ Q @ E)   # (Nc*4, 2)
        S[np.repeat(b.swing_mask.reshape(-1)[:, None], 2, 1)] = 0.0
        # the trial's feedback: the time-varying LQR of the ILC's weights along it -- Stage III
        # Qe on the landing, the stages' tracking rows (Qe x k_crun) on the samples before it,
        # R = k_rscale x Qu (bank: 0.1 and 0.01, the weights that serve the LQR best)
        c_run = float(b.cfg.get("k_crun", 0.01))
        R = float(b.cfg.get("k_rscale", 0.1)) * qu * np.eye(nu)
        P = Q.copy()
        for t in range(N - 1, Nc - 1, -1):
            P = c_run * Q + A_l[t].T @ P @ A_l[t]
        K = np.zeros((Nc, nu, 6))
        Hu = np.zeros((Nc, nu, nu))                  # the QP's curvature in u_t: R + B'P B
        for t in range(Nc - 1, -1, -1):
            A, B = A_l[t], B_l[t]
            Hu[t] = R + B.T @ P @ B
            K[t] = np.linalg.solve(Hu[t], B.T @ P @ A)
            P = c_run * Q + A.T @ P @ (A - B @ K[t])
            P = 0.5 * (P + P.T)
        x0 = plan_ref["x_ref"][0, :2]
        g_ach = np.asarray(logX[N, :2], float) - x0
        dU = (S @ (np.asarray(g_label, float) - g_ach)).reshape(Nc, nu)
        u_star = U + dU
        dxr, dur = b.reference_jac(g_label)
        SU = S.reshape(Nc, nu, 2)
        tgt = np.array([b.norm_action(u_star[k], k, label_ref["u_ref"]) for k in range(Nc)])
        Kx = np.zeros((Nc, nu, 6))
        Kg = np.zeros((Nc, nu, 2))
        for k in range(Nc):
            dxs = np.zeros((6, 2)) if k == 0 else \
                np.einsum("jab,jbc->ac", G[k - 1], SU)          # d x*_k / d g
            dpi = SU[k] + K[k] @ dxs - K[k] @ dxr[k] - dur[k]
            m = ~b.swing_mask[k]
            Kx[k][m] = (-K[k] * D[None, :] / asc[:, None])[m]
            Kg[k][m] = (dpi * gs[None, :] / asc[:, None])[m]
        GNn = GN.reshape(6, Nc, nu) * asc[None, None, :]       # per unit of normalized action
        # the trial's co-states (Gurumurthy et al. Sec. 4.2: the value gradient is the co-state),
        # stage by stage, from its own linearization and its measured errors against the label
        # goal's plan: q_k = d(cost of transitions k..)/d x_{k+1} with the forces held,
        # q_k = grad c(x_{k+1}) + gamma A_{k+1}' q_{k+1}; the flight and landing rows belong
        # to the last transition (as the episode's rewards do). Targets for the critic:
        # dQ_k/da_k = B_k' q_k, dQ_k/de_k = A_k' q_k (normalized)
        gamma = float(b.cfg.get("gamma", 0.99))
        xr = label_ref["x_ref"]
        X = np.asarray(logX, float)
        rs = self.r_scale
        target = xr[N].copy()
        target[:2] = xr[0, :2] + np.asarray(g_label, float)
        target[2] = 0.0
        grad_c = lambda e: 2 * rs * qe * e
        mca = np.zeros((Nc, 3, nu))
        mcs = np.zeros((Nc, 3, 6))
        mca_cl = np.zeros((Nc, 3, nu))      # the same, under the trial's feedback K (Q^pi of a
        mcs_cl = np.zeros((Nc, 3, 6))       # feedback policy: iLQR/DDP's V_x recursion)
        for s in range(3):
            # the flight/landing part, carried back to x_Nc through the flight's own A's
            q = np.zeros(6)
            for m in range(N, Nc, -1):
                c_m = np.zeros(6)
                if s == 1:
                    c_m = grad_c(X[m] - xr[m])
                if s == 2 and m == N:
                    c_m = grad_c(X[N] - target)
                q = A_l[m - 1].T @ (q + c_m) if m - 1 >= Nc else q + c_m
            # q is now d(flight + landing cost)/d x_Nc; the contact samples back to 0
            q_cl = q.copy()
            for k in range(Nc - 1, -1, -1):
                m = k + 1                                 # this transition's next state
                c_m = np.zeros(6)
                if s == 0 or (s == 1 and m >= b.Ndc):
                    c_m = grad_c(X[m] - xr[m])
                qk = c_m + (q if k == Nc - 1 else gamma * (A_l[m].T @ q))
                qk_cl = c_m + (q_cl if k == Nc - 1 else
                               gamma * ((A_l[m] - B_l[m] @ K[m]).T @ q_cl))
                mca[k, s] = (B_l[k].T @ qk) * asc
                mcs[k, s] = (A_l[k].T @ qk) * D
                mca_cl[k, s] = (B_l[k].T @ qk_cl) * asc
                mcs_cl[k, s] = (A_l[k].T @ qk_cl) * D
                q, q_cl = qk, qk_cl
        return dict(tgt=tgt.astype(np.float32), Kx=Kx.astype(np.float32),
                    Kg=Kg.astype(np.float32), GN=GNn.astype(np.float32),
                    g_ach=g_ach.astype(np.float32), mca=mca.astype(np.float32),
                    mcs=mcs.astype(np.float32), mca_cl=mca_cl.astype(np.float32),
                    mcs_cl=mcs_cl.astype(np.float32),
                    gcN=grad_c(X[N] - target).astype(np.float32),     # d landing cost / d x_N
                    Hu=(rs * Hu * asc[None, :, None] * asc[None, None, :]).astype(np.float32))

    def stick(self, rec):
        """(unstick ms, rear-unloaded ms) after touchdown from the true contacts
        (experiments/go1/sweep/stick.py)."""
        if rec is None or "t" not in rec:
            return (np.nan, np.nan)
        k = "true_contacts" if "true_contacts" in rec else "contacts"
        t, con = np.asarray(rec["t"]), np.asarray(rec[k])
        Nc, dt = self.bank.Nc, self.bank.dt
        air = np.where((t > Nc * dt + 0.03) & (con.max(1) < 1))[0]
        un = rear = np.nan
        if air.size:
            after = np.arange(air[0], len(t))
            alld = after[con[after].min(1) >= 1]
            if alld.size:
                tail = np.arange(alld[0], len(t))
                ddt = np.diff(t[tail], append=t[tail][-1])
                un = 1000 * ddt[con[tail].min(1) < 1].sum()
                rear = 1000 * ddt[con[tail][:, 2:].min(1) < 1].sum()
        return (un, rear)


INFO_COLS = ["ex", "ez", "eth", "fell", "unstick_ms", "rear_ms"]


def stack_episodes(eps: list[dict], extra: dict | None = None) -> dict:
    out = {k: np.stack([e[k] for e in eps]) for k in eps[0]}
    out.update(extra or {})
    return out


def to_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def from_bytes(b: bytes) -> dict:
    with np.load(io.BytesIO(b), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


# serve: rollouts for dilc_train.py over stdin/stdout ---------------------------------------
def _read_msg(f):
    h = f.read(4)
    if len(h) < 4:
        return None, None
    (n,) = struct.unpack("<I", h)
    header = json.loads(f.read(n))
    (m,) = struct.unpack("<Q", f.read(8))
    payload = from_bytes(f.read(m)) if m else {}
    return header, payload


def _write_msg(f, header, payload=None):
    hb = json.dumps(header).encode()
    pb = to_bytes(payload) if payload else b""
    f.write(struct.pack("<I", len(hb)) + hb + struct.pack("<Q", len(pb)) + pb)
    f.flush()


def _serve_one(bank, maker, header, payload, root, out):
    """One rollout of serve, its message written to out."""
    t0 = time.time()
    try:
        g = header["goal"]
        rng = np.random.default_rng(header["seed"])
        if header.get("cond") == "dr":
            cname, cond = sample_condition(rng)
        else:
            cname, cond = header["cond"], NAMED.get(header["cond"], header["cond"])
        kw = dict(window=bank.cfg["est_window"])
        if payload:
            payload["n_hidden"] = int(payload["n_hidden"])
            kw.update(actor=NumpyActor(payload),
                      rng=rng if header.get("stochastic") else None)
        else:
            kw.update(U_fixed=bank.reference(g)[0]["u_ref"])
        if header.get("fd"):              # the simulator's Jacobians: dict(eps_a, eps_s)
            kw.update(fd=FDJac(**header["fd"]))
        drv, log, rec, fell, ref, _ = run_jump(bank, g, cond, header["seed"], kw,
                                                land_time=header.get("land_time", 1.0),
                                                menagerie_root=root)
        jac = drv.fd.jacobians(bank) if drv.fd is not None else None
        ep = maker.make(g, ref, drv.X, drv.U, log["X"], fell, rec, jac=jac)
        fd_ok = float(np.isfinite(jac[0]).all((1, 2)).mean()) if jac is not None else 0.0
        _write_msg(out, dict(ok=True, id=header.get("id"), cond=cname, cond_args=cond,
                             wall=time.time() - t0, fd_ok=fd_ok), ep)
    except Exception as err:          # report and carry on: one bad rollout is not fatal
        import traceback
        traceback.print_exc()
        _write_msg(out, dict(ok=False, id=header.get("id"), error=repr(err)))


def serve(args):
    # the protocol owns stdout; everything the controller prints goes to stderr
    proto = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    inp = sys.stdin.buffer
    bank = GoalBank(json.load(open(args.bank)))
    maker = EpisodeMaker(bank, args.menagerie_root)
    import ilc_jump_lockstep  # noqa: F401  (imported once here, not in every jump's process)
    while True:
        header, payload = _read_msg(inp)
        if header is None or header.get("cmd") == "quit":
            break
        # each rollout in its own short-lived process: whatever a jump leaves behind (the ROS
        # context, the controller's models) goes with it, so the server stays at its starting
        # size -- and FDJac's copies fork from a small process, not one grown over many jumps
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            with os.fdopen(w, "wb") as out:
                _serve_one(bank, maker, header, payload, args.menagerie_root, out)
            os._exit(0)
        os.close(w)
        with os.fdopen(r, "rb") as f:
            msg = f.read()
        _, status = os.waitpid(pid, 0)
        if msg:
            proto.write(msg)
            proto.flush()
        else:
            _write_msg(proto, dict(ok=False, id=header.get("id"),
                                   error=f"rollout process died (status {status})"))


# dataset: ILC runs -> episodes --------------------------------------------------------------
def labeled_episodes(bank, maker, g, ref, X, U, logX, fell, rec, trial=0):
    """A flown jump as the trainer's episodes: for the goal it aimed at, and (unless it
    fell) in hindsight for where it landed, each with the ILC labels of its own A, B."""
    ep = maker.make(g, ref, X, U, logX, fell, rec)
    ep.update(maker.ilc_labels(ref, U, logX, g, ref))
    ep.update(trial=np.int32(trial), sup=np.float32(not fell), relabel=np.float32(0.0),
              anchor=np.float32(0.0))
    out = [ep]
    g_h = bank.project(ep["g_ach"])
    if not fell and g_h is not None and np.linalg.norm(g_h - np.asarray(g)) > 2e-3:
        ref_h = bank.reference(g_h)[0]
        eh = maker.make(g_h, ref_h, X, U, logX, fell, rec)
        eh.update(maker.ilc_labels(ref, U, logX, g_h, ref_h))
        eh.update(trial=np.int32(trial), sup=np.float32(1.0), relabel=np.float32(1.0),
                  anchor=np.float32(0.0))
        out.append(eh)
    return out


def _rollout_one(job):
    policy_dir, member, g, cname, cond, seed, root = job
    os.environ["ROS_DOMAIN_ID"] = str(1 + os.getpid() % 98)   # < 100: no ephemeral ports
    bank, actor, _, _ = _cached(("policy", policy_dir, member),
                                lambda: load_policy(policy_dir, member))
    maker = _cached(("maker", policy_dir), lambda: EpisodeMaker(bank, root))
    ref, box, extrap = bank.reference(g)
    drv, log, rec, fell, ref, _ = run_jump(bank, g, cond, seed,
                                            dict(actor=actor, window=bank.cfg["est_window"]),
                                            land_time=1.5, reference=(ref, box, extrap),
                                            menagerie_root=root)
    eps = labeled_episodes(bank, maker, g, ref, drv.X, drv.U, log["X"], fell, rec, seed)
    sc, e = landing_score(log["X"], ref["x_ref"], g, fell, np.asarray(bank.cfg["qe"]),
                          bank.cfg["r_scale"])
    return eps, dict(goal=list(map(float, g)), cond=cname, seed=seed, score=sc,
                     ex=float(e[0]), ez=float(e[1]), eth=float(e[2]), fell=bool(fell))


def rollouts(args):
    """The policy's own jumps on the robot (real rollouts, the paper's fine-tuning setting):
    goals spread over the bank's hull, each labeled from its own A, B like an ILC trial --
    for one policy across every goal, instead of an ILC run per goal."""
    # forkserver, not fork: a worker forked from the threaded parent (the pool's own threads) can deadlock on an
    # inherited lock, past the per-jump alarm
    from multiprocessing import get_context
    bank = GoalBank(json.load(open(os.path.join(args.policy, "bank.json"))))
    G = bank.goals
    D = G - G[0]
    d = D[np.argmax(np.linalg.norm(D, axis=1))]
    t = np.linspace(0.0, 1.0, args.n_goals)
    rng = np.random.default_rng(args.seed)
    t = np.clip(t + rng.uniform(-0.5, 0.5, args.n_goals) / max(args.n_goals - 1, 1), 0, 1)
    goals = [G[0] + ti * d for ti in t]
    cname, cond = conditions(args.cond, args.seed)[0]
    jobs = [(args.policy, args.member, g, cname, cond, args.seed + 97 * i,
             args.menagerie_root) for i, g in enumerate(goals)]
    eps, rows = [], []
    with get_context("forkserver").Pool(min(args.jobs, len(jobs)), maxtasksperchild=8) as pool:
        for e, r in pool.imap_unordered(_rollout_one, jobs):
            eps += e
            rows.append(r)
    out = stack_episodes(eps, dict(run=np.array([f"rollout_{args.tag}"] * len(eps))))
    np.savez(args.out, **out)
    sc = [r["score"] for r in rows]
    print(f"rollouts {args.tag}: {len(rows)} jumps ({cname}), mean score {np.mean(sc):.2f}, "
          f"median {np.median(sc):.2f}, falls {sum(r['fell'] for r in rows)} -> {len(eps)} "
          f"labeled episodes in {args.out}")
    with open(args.out.replace(".npz", ".json"), "w") as f:
        json.dump(rows, f, indent=1)


def _dataset_run(job):
    run_dir, bank_path, root = job
    import glob
    bank = GoalBank(json.load(open(bank_path)))
    maker = EpisodeMaker(bank, root)
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    latency = float(meta.get("params", {}).get("pose_latency", 0.0))
    cond = os.path.basename(os.path.normpath(run_dir)).partition("__")[2]
    eps, costs, hind = [], [], []
    for p in sorted(glob.glob(os.path.join(run_dir, "trial_*.npz"))):
        d = dict(np.load(p))
        cfg = json.loads(str(d["config"]))
        g = np.array(cfg["jump"], float)
        ref = {k: d[k] for k in d if k in ("x_ref", "u_ref") or k.startswith("info_")}
        rec = {k[4:]: d[k] for k in d if k.startswith("rec_")}
        X = replay_states(maker.fb, ref["x_ref"], rec, bank.dt, bank.Nc, latency,
                          bank.cfg["est_window"])
        fell = bool(d["log_fell"])
        U, logX = d["U_flown"], d["log_X"]
        # the trial for the goal it aimed at, and in hindsight for the goal it reached (on
        # the bank's goals line): the same states and forces, against that goal's plan
        pair = labeled_episodes(bank, maker, g, ref, X, U, logX, fell, rec, int(d["trial"]))
        eps.append(pair[0])
        hind += pair[1:]
        e = pair[0]["info"][:3]
        qe = maker.qe[:3]
        costs.append(np.inf if fell else float(e @ (qe * e)))
    best = int(np.argmin(costs)) if np.isfinite(costs).any() else -1
    for i, ep in enumerate(eps):
        ep["anchor"] = np.float32(i == best)
    for eh in hind:
        eh["anchor"] = np.float32(0.0)
    return run_dir, cond, eps + hind


def dataset(args):
    from multiprocessing import get_context
    jobs = [(r, args.bank, args.menagerie_root) for r in args.runs
            if os.path.exists(os.path.join(r, "meta.json"))]
    eps, names = [], []
    with get_context("forkserver").Pool(args.jobs) as pool:
        for run_dir, cond, run_eps in pool.imap_unordered(_dataset_run, jobs):
            for ep in run_eps:
                eps.append(ep)
                names.append(f"{os.path.basename(os.path.normpath(run_dir))}")
            print(f"{run_dir}: {len(run_eps)} trials", file=sys.stderr)
    out = stack_episodes(eps, dict(run=np.array(names)))
    np.savez(args.out, **out)
    print(f"{len(eps)} episodes -> {args.out}", file=sys.stderr)


# run: execute a trained policy ---------------------------------------------------------------
def load_policy(policy_dir, member="best"):
    man = json.load(open(os.path.join(policy_dir, "manifest.json")))
    bank = GoalBank(json.load(open(os.path.join(policy_dir, "bank.json"))))
    if member == "all":
        members = [m["name"] for m in man["members"]]
    elif member.startswith("mean:"):           # mean:v1,v2@N: the ensemble mean of those members
        spec, _, upd = member[5:].partition("@")
        members = [m["name"] for m in man["members"] if m["variant"] in spec.split(",")
                   and (not upd or int(m["updates"]) == int(upd))]
    elif member == "best":
        members = [man["best"]]
    else:
        members = [member if not member.isdigit() else f"member{int(member)}"]
    actors = [NumpyActor.load(os.path.join(policy_dir, f"{m}.npz")) for m in members]
    return bank, actors[0] if len(actors) == 1 else Ensemble(actors), members, man


def build_lqr(bank: GoalBank, policy_dir, g, ref, rscale=1.0, root="", c_run=0.01,
              c_flight=0.01):
    """
    The feedback baseline one gets from the ILC with no learning: its best nominal trial of
    each bank goal (policy/ilc_best.json) -- forces, and states as the live estimator saw
    them -- interpolated to goal g, the SRB model linearized along that, and a backward
    Riccati pass with the ILC's own weights: Qe x c_run on the contact samples, Qe on the
    landing (and x c_flight through the flight, the ILC's Stage II rows) carried back by
    the ballistic flight map, R = rscale x the Stage III Qu. u_k = U_ff[k] - K_k (x - X_nom[k]).
    Returns (U_ff, X_nom, K, P), P[k] the cost-to-go's Hessian at sample k (k = 0..Nc).
    """
    maker = _cached(("maker", policy_dir), lambda: EpisodeMaker(bank, root))
    paths = json.load(open(os.path.join(policy_dir, "ilc_best.json")))
    w, _ = bank.weights(g)
    U_ff = np.array(ref["u_ref"], float)
    X_nom = 0.0
    for wi, p, r in zip(w, bank.plans, bank.refs):
        d = dict(np.load(paths[p["name"]]))
        meta = json.load(open(os.path.join(os.path.dirname(paths[p["name"]]), "meta.json")))
        lat = float(meta.get("params", {}).get("pose_latency", 0.0))
        rec = {k[4:]: d[k] for k in d if k.startswith("rec_")}
        Xh = replay_states(maker.fb, d["x_ref"], rec, bank.dt, bank.Nc, lat, bank.cfg["est_window"])
        U_ff = U_ff + wi * (d["U_flown"] - r["u_ref"])
        X_nom = X_nom + wi * Xh
    Nc, N, dt = bank.Nc, bank.N, bank.dt
    qe = np.diag(np.asarray(bank.cfg["qe"], float))
    R = rscale * 1e-5 * np.eye(4)
    Af = maker.srb.f_lin(np.zeros(6), np.zeros(4), np.zeros(2), np.zeros(2), dt)[0].full()
    P, Phi = np.zeros((6, 6)), np.eye(6)
    for m in range(Nc + 1, N + 1):                  # the flight and the landing, back to x_Nc
        Phi = Af @ Phi
        P += (1.0 if m == N else c_flight) * Phi.T @ qe @ Phi
    K = np.zeros((Nc, 4, 6))
    Ps = np.zeros((Nc + 1, 6, 6))
    Ps[Nc] = P
    for k in range(Nc - 1, -1, -1):
        A, B = maker.srb.f_lin(X_nom[k], U_ff[k], ref["info_R1"][k], ref["info_R2"][k], dt)
        A, B = A.full(), B.full()
        B[:, bank.swing_mask[k]] = 0.0
        K[k] = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = c_run * qe + A.T @ P @ (A - B @ K[k])
        P = 0.5 * (P + P.T)
        Ps[k] = P
    return U_ff, X_nom, K, Ps


_CACHE = {}


def _cached(key, make):
    """Per worker process: the policy and the models are built once, not per jump."""
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def _run_one(job):
    """One evaluation jump; a jump that hangs (a stuck ROS context) is reported, not waited on."""
    import signal

    def _timeout(*_):
        raise TimeoutError("jump timed out")
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(300)
    try:
        return _run_one_inner(job)
    except Exception as err:
        g = job[2]
        print(f"jump failed ({job[3]}, g={g[0]:.3f}, {job[1]}): {err!r}", file=sys.stderr)
        return None
    finally:
        signal.alarm(0)


def _run_one_inner(job):
    (policy_dir, member, g, cname, cond, seed, mode, land_time, root, ilc_U_path,
     ref_file, save_dir) = job
    os.environ["ROS_DOMAIN_ID"] = str(1 + os.getpid() % 98)   # < 100: no ephemeral ports
    if mode != "policy":
        member = "best"
    if mode == "policy":
        bank, actor, _, _ = _cached(("policy", policy_dir, member),
                                    lambda: load_policy(policy_dir, member))
    else:                                  # baselines need the bank only, no trained policy
        bank, actor = _cached(("bank", policy_dir), lambda: GoalBank(
            json.load(open(os.path.join(policy_dir, "bank.json"))))), None
    ref, box, extrap = bank.reference(g)
    kw = dict(window=bank.cfg["est_window"])
    if ref_file:
        ref = dict(np.load(ref_file))
    if mode == "policy":
        kw["actor"] = actor
    elif mode == "ilc_interp":
        # the ILC's converged forces of each bank goal, interpolated like the reference:
        # what zero-shot gives without a network
        U = dict(np.load(ilc_U_path))
        w, _ = bank.weights(g)
        dU = sum(wi * (U[p["name"]] - r["u_ref"]) for wi, p, r in zip(w, bank.plans, bank.refs))
        kw["U_fixed"] = ref["u_ref"] + dU
    elif mode == "to":
        kw["U_fixed"] = ref["u_ref"]
    elif mode.startswith("lqr"):                # lqr[:R scale[:running-cost weight]]
        opts = (mode.split(":")[1:] + ["", ""])[:2]
        rscale = float(opts[0] or 1.0)
        c_run = float(opts[1]) if opts[1] else 0.01
        lq = _cached(("lqr", policy_dir, tuple(np.round(g, 6)), rscale, c_run),
                     lambda: build_lqr(bank, policy_dir, g, ref, rscale, root, c_run, c_run))
        kw["U_fixed"], kw["lqr"] = lq[0], lq
    else:
        raise ValueError(mode)
    t0 = time.time()
    drv, log, rec, fell, ref, _ = run_jump(bank, g, cond, seed, kw, land_time=land_time,
                                            reference=(ref, box, extrap), menagerie_root=root,
                                            reference_file=ref_file)
    maker = _cached(("maker", policy_dir), lambda: EpisodeMaker(bank, root))
    ep = maker.make(g, ref, drv.X, drv.U, log["X"], fell, rec)
    info = dict(zip(INFO_COLS, (float(v) for v in ep["info"])))
    cost = float(-(ep["rc"][:, 2].sum()))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, f"{mode}_g{g[0]:.3f}_{g[1]:.3f}_{cname}_s{seed}.npz"),
                 **ep, **{f"rec_{k}": v for k, v in rec.items()})
    return dict(mode=mode, member=member if mode == "policy" else "", goal=list(map(float, g)),
                cond=cname, seed=seed,
                extrapolated=extrap, landing_cost=-cost, wall=time.time() - t0, **info,
                passed=bool(abs(info["ex"]) <= 0.01 and abs(info["ez"]) <= 0.02
                            and abs(info["eth"]) <= math.radians(2.0) and not info["fell"]))


def summarize(rows, key=("mode",)):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in key), []).append(r)
    lines = []
    for k, rs in sorted(groups.items()):
        ex = np.array([abs(r["ex"]) for r in rs]) * 100
        ez = np.array([abs(r["ez"]) for r in rs]) * 100
        th = np.degrees([abs(r["eth"]) for r in rs])
        rear = np.array([r["rear_ms"] for r in rs if not r["fell"]])
        lines.append(f"{'/'.join(map(str, k)):28s} n={len(rs):3d} pass={sum(r['passed'] for r in rs):3d} "
                     f"falls={sum(bool(r['fell']) for r in rs):3d} |ex| med {np.median(ex):4.1f} "
                     f"p90 {np.percentile(ex, 90):4.1f} cm  |ez| med {np.median(ez):4.1f} cm  "
                     f"|eth| med {np.median(th):4.1f} p90 {np.percentile(th, 90):4.1f} deg  "
                     f"rear med {np.nanmedian(rear) if rear.size else np.nan:4.0f} ms")
    return "\n".join(lines)


def run(args):
    from multiprocessing import get_context
    conds = conditions(args.conds, args.seed)
    modes = ["policy", "ilc_interp", "to"] if args.compare else args.mode.split(",")
    ilc_U = os.path.join(args.policy, "ilc_U.npz")
    if args.member.startswith("each"):   # every member on its own (policy mode only);
        man = json.load(open(os.path.join(args.policy, "manifest.json")))   # each:v1,v2
        spec, _, upd = args.member.partition("@")                # each:v1,v2@N: snapshot N only
        keep = spec.partition(":")[2].split(",") if ":" in spec else None
        members = [m["name"] for m in man["members"] if (keep is None or m["variant"] in keep)
                   and (not upd or int(m["updates"]) == int(upd))]
    else:
        members = [args.member]
    jobs = []
    for g in args.jump:
        for cname, cond in conds:
            for seed in cell_seeds(args.seed, args.episodes):
                for mode, member in [(m, mb) for m in modes
                                     for mb in (members if m == "policy" else ["best"])]:
                    jobs.append((args.policy, member, np.array(g, float), cname, cond,
                                 seed, mode, args.land_time,
                                 args.menagerie_root, ilc_U, args.reference_file,
                                 args.save_dir))
    rows = []
    with get_context("forkserver").Pool(min(args.jobs, len(jobs)), maxtasksperchild=8) as pool:
        for r in pool.imap_unordered(_run_one, jobs):
            if r is None:
                continue
            rows.append(r)
            if args.member.startswith("each"):
                continue
            print(f"{r['mode']:10s} g=({r['goal'][0]:.3f},{r['goal'][1]:.3f}) {r['cond']:9s} "
                  f"ex {r['ex'] * 100:+5.1f} cm ez {r['ez'] * 100:+5.1f} cm "
                  f"eth {math.degrees(r['eth']):+5.1f} deg {'FELL' if r['fell'] else ''} "
                  f"rear {r['rear_ms']:4.0f} ms {'PASS' if r['passed'] else ''}", flush=True)
    print("\n" + summarize(rows, ("mode", "member")))
    if not args.member.startswith("each"):
        print("\n" + summarize(rows, ("mode", "cond")))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1, default=float)


# ilc: how many trials the ILC itself takes, from scratch or warm-started --------------------
def cell_seeds(base, episodes):
    """The reality seeds of a goal/condition cell: `run` and `ilc` fly the same ones, so a
    policy's zero-shot jump and the ILC's first trial meet the same robot."""
    return [base + 1000 * s + 17 for s in range(episodes)]


def landing_score(logX, x_ref, g, fell, qe, r_scale, reward_scale=0.01, fall_penalty=20.0):
    """The trainer's eval score of one jump: the ILC's Stage III cost of the landing (x_N
    against the goal, pitch 0, the plan's landing velocities), scaled, plus a fall penalty."""
    target = np.array(x_ref[-1], float)
    target[:2] = x_ref[0, :2] + np.asarray(g, float)
    target[2] = 0.0
    e = np.asarray(logX[-1], float) - target
    return reward_scale * r_scale * float(e @ (qe * e)) + fall_penalty * float(fell), e


def _ilc_one(job):
    import glob
    import subprocess
    (policy_dir, member, g, cname, cond, seed, start, out, max_trials, root, jid, sec_prior) = job
    os.environ["ROS_DOMAIN_ID"] = str(1 + jid % 98)
    bank, actor, _, _ = load_policy(policy_dir, member) if start == "policy" else \
        (GoalBank(json.load(open(os.path.join(policy_dir, "bank.json")))), None, None, None)
    ref, box, _ = bank.reference(g)
    name = f"{start}_g{g[0]:.3f}_{g[1]:.3f}_{cname}_s{seed}"
    ref_path = os.path.join(out, "refs", f"{name}.npz")
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    np.savez(ref_path, **ref)
    cfg = json.loads(str(ref["config"]))
    extra, pre = [], None
    if start in ("interp", "policy", "nearest"):
        if start == "nearest":
            # the nearest ILC'd goal's converged forces, its correction on top of its plan
            # carried onto this goal's plan (JumpILC.transfer_from's retarget)
            U = dict(np.load(os.path.join(policy_dir, "ilc_U.npz")))
            i = int(np.argmin(np.linalg.norm(bank.goals - np.asarray(g), axis=1)))
            U0 = ref["u_ref"] + (U[bank.plans[i]["name"]] - bank.refs[i]["u_ref"])
        elif start == "interp":
            U = dict(np.load(os.path.join(policy_dir, "ilc_U.npz")))
            w, _ = bank.weights(g)
            U0 = ref["u_ref"] + sum(wi * (U[p["name"]] - r["u_ref"])
                                    for wi, p, r in zip(w, bank.plans, bank.refs))
        else:
            # the policy's zero-shot jump on this robot, then the ILC from the forces it flew
            drv, log, rec, fell, _, _ = run_jump(bank, g, cond, seed,
                                                  dict(actor=actor, window=bank.cfg["est_window"]),
                                                  land_time=1.5, reference=(ref, box, False),
                                                  menagerie_root=root)
            U0 = np.asarray(drv.U)
            sc, e = landing_score(log["X"], ref["x_ref"], g, fell, np.asarray(bank.cfg["qe"]),
                                  bank.cfg["r_scale"])
            pre = dict(score=sc, ex=float(e[0]), ez=float(e[1]), eth=float(e[2]), fell=bool(fell))
        u_path = os.path.join(out, "refs", f"{name}_U.npz")
        np.savez(u_path, U=U0)
        extra = ["--param", f"initial_forces:={u_path}", "--param", "n_stage1:=0",
                 "--param", "n_stage2:=0"]
    lockstep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ilc_jump_lockstep.py")
    argv = [sys.executable, lockstep, "--robot", cfg["robot"], "--max-trials", str(max_trials),
            "--jump", str(g[0]), str(g[1]), "--reference-file", ref_path,
            "--param", "phases:=[" + ",".join(str(int(v)) for v in cfg["phases"]) + "]",
            "--param", f"margin:={cfg['margin']}", *extra,
            *shlex.split(strip_seed(cond)), "--seed", str(seed),
            "--log-dir", out, "--run-name", name,
            "--summary", os.path.join(out, name + ".json")]
    if box is not None:
        argv += ["--box", str(box["x_front"]), str(box["height"])]
    if root:
        argv += ["--menagerie-root", root]
    if sec_prior:
        argv += ["--param", f"stage3_secant_prior:={sec_prior}"]
    with open(os.path.join(out, name + ".log"), "w") as f:
        subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)
    scores, conv, fells, errs = [], None, [], []
    qe = np.asarray(bank.cfg["qe"])
    for p in sorted(glob.glob(os.path.join(out, name, "trial_*.npz"))):
        d = np.load(p)
        r = json.loads(str(d["result"]))
        sc, e = landing_score(d["log_X"], d["x_ref"], g, bool(d["log_fell"]), qe,
                              bank.cfg["r_scale"])
        scores.append(sc)
        fells.append(bool(d["log_fell"]))
        errs.append([float(e[0]), float(e[1]), float(e[2])])
        if conv is None and r.get("converged"):
            conv = int(r["trial"])
    return dict(start=start, goal=list(map(float, g)), cond=cname, seed=seed, run=name,
                converged_at=conv, scores=scores, fell=fells, err=errs, policy_jump=pre)


def ilc_bench(args):
    from multiprocessing import get_context
    os.makedirs(args.out, exist_ok=True)
    conds = conditions(args.conds, args.seed)
    jobs, jid = [], 0
    for start in args.start.split(","):
        for g in args.jump:
            for cname, cond in conds:
                for seed in cell_seeds(args.seed, args.episodes):
                    jid += 1
                    jobs.append((args.policy, args.member, np.array(g, float), cname, cond, seed,
                                 start, args.out, args.max_trials, args.menagerie_root, jid,
                                 args.secant_prior))
    rows = []
    with get_context("forkserver").Pool(min(args.jobs, len(jobs)), maxtasksperchild=8) as pool:
        for r in pool.imap_unordered(_ilc_one, jobs):
            rows.append(r)
            pj = r["policy_jump"]
            print(f"{r['start']:8s} g={r['goal'][0]:.3f} {r['cond']:9s} s{r['seed']}: converged at "
                  f"{r['converged_at'] if r['converged_at'] else '>' + str(len(r['scores']))} "
                  f"| scores {' '.join(f'{s:.1f}' for s in r['scores'][:8])}"
                  + (f" | policy jump {pj['score']:.1f}" if pj else ""), flush=True)
    for start in args.start.split(","):
        rs = [r for r in rows if r["start"] == start]
        conv = [r["converged_at"] for r in rs]
        n_ok = [c for c in conv if c]
        print(f"{start:8s}: converged {len(n_ok)}/{len(rs)}, trials to converge median "
              f"{np.median(n_ok) if n_ok else float('nan'):.1f} (converged runs), "
              f"mean score trial 1 {np.mean([r['scores'][0] for r in rs]):.2f}")
    with open(os.path.join(args.out, "ilc_bench.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)


def lqr_bank(args):
    """The ILC's LQR problems over a grid of goals across the bank's hull, for the trainer's
    LQR regularization: per goal the interpolated ILC forces and states, the Riccati gains
    and cost-to-go Hessians, and the plan's forces and states they are measured against."""
    bank = GoalBank(json.load(open(os.path.join(args.policy, "bank.json"))))
    G = bank.goals
    D = G - G[0]
    if np.linalg.matrix_rank(D, tol=1e-6) > 1:
        raise SystemExit("lqrbank: goals off a line are not supported yet")
    d = D[np.argmax(np.linalg.norm(D, axis=1))]
    s = D @ d / (d @ d)
    goals = np.array([G[0] + t * d for t in np.linspace(s.min(), s.max(), args.n_goals)])
    out = {k: [] for k in ("U_ff", "X_nom", "K", "P", "u_ref", "x_ref")}
    for g in goals:
        ref, _, _ = bank.reference(g)
        U_ff, X_nom, K, P = build_lqr(bank, args.policy, g, ref, args.rscale,
                                      args.menagerie_root, args.c_run, args.c_run)
        for k, v in (("U_ff", U_ff), ("X_nom", X_nom), ("K", K), ("P", P),
                     ("u_ref", ref["u_ref"]), ("x_ref", ref["x_ref"][:bank.Nc + 1])):
            out[k].append(np.asarray(v, float))
    np.savez(args.out, goals=goals, **{k: np.stack(v) for k, v in out.items()},
             rscale=args.rscale, c_run=args.c_run)
    print(f"lqrbank: {len(goals)} goals {goals[0]} .. {goals[-1]} -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("rollouts", help="the policy's own jumps across the goals, labeled "
                                        "like ILC trials (for fine-tuning)")
    o.add_argument("--policy", required=True)
    o.add_argument("--member", default="best")
    o.add_argument("--n-goals", type=int, default=32)
    o.add_argument("--cond", default="nominal")
    o.add_argument("--seed", type=int, default=5000)
    o.add_argument("--tag", default="r1")
    o.add_argument("--out", required=True)
    o.add_argument("--jobs", type=int, default=os.cpu_count())
    o.add_argument("--menagerie-root", default="")
    q = sub.add_parser("lqrbank", help="the ILC's LQR problems over a grid of goals")
    q.add_argument("--policy", required=True, help="folder with bank.json and ilc_best.json")
    q.add_argument("--out", required=True)
    q.add_argument("--n-goals", type=int, default=41)
    q.add_argument("--rscale", type=float, default=0.1)
    q.add_argument("--c-run", type=float, default=0.01)
    q.add_argument("--menagerie-root", default="")
    b = sub.add_parser("ilc", help="the ILC on the same goals/robots: trials to converge, from "
                                   "scratch, the interpolated ILC forces, or the policy's jump")
    b.add_argument("--policy", required=True, help="policy folder (bank.json, ilc_U.npz)")
    b.add_argument("--member", default="best")
    b.add_argument("--jump", type=float, nargs=2, action="append", required=True)
    b.add_argument("--conds", default="challenging")
    b.add_argument("--episodes", type=int, default=1)
    b.add_argument("--seed", type=int, default=11)
    b.add_argument("--start", default="scratch",
                   help="scratch | nearest | interp | policy (comma list)")
    b.add_argument("--max-trials", type=int, default=20)
    b.add_argument("--secant-prior", default="",
                   help="npz with C: Stage III plans with G_N + C (fixed, fitted offline)")
    b.add_argument("--out", required=True)
    b.add_argument("--jobs", type=int, default=os.cpu_count())
    b.add_argument("--menagerie-root", default="")
    r = sub.add_parser("run", help="execute a trained policy (zero-shot, no learning)")
    r.add_argument("--policy", required=True, help="dilc_train.py's OUT/policy")
    r.add_argument("--member", default="best",
                   help="best | all (ensemble mean) | mean:VARIANT@UPDATES | each:VARIANT@UPDATES | NAME")
    r.add_argument("--jump", type=float, nargs=2, action="append", required=True,
                   metavar=("DX", "DZ"), help="goal(s): CoM displacement, m")
    r.add_argument("--conds", default="challenging",
                   help="challenging | all | dr:N | name,name | conds file")
    r.add_argument("--episodes", type=int, default=1, help="seeds per goal and condition")
    r.add_argument("--mode", default="policy",
                   help="policy | ilc_interp | to | lqr[:R scale] (comma list)")
    r.add_argument("--compare", action="store_true", help="all three modes")
    r.add_argument("--reference-file", default="",
                   help="fly this plan instead of the interpolated reference")
    r.add_argument("--land-time", type=float, default=1.5)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--jobs", type=int, default=os.cpu_count())
    r.add_argument("--json", default="")
    r.add_argument("--save-dir", default="", help="keep each jump's episode + recording")
    r.add_argument("--menagerie-root", default="")
    d = sub.add_parser("dataset", help="ILC run folders -> training episodes")
    d.add_argument("--bank", required=True)
    d.add_argument("--runs", nargs="+", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--jobs", type=int, default=os.cpu_count())
    d.add_argument("--menagerie-root", default="")
    s = sub.add_parser("serve", help="rollout worker for dilc_train.py (stdin/stdout)")
    s.add_argument("--bank", required=True)
    s.add_argument("--menagerie-root", default="")
    args = ap.parse_args()
    {"run": run, "dataset": dataset, "serve": serve, "ilc": ilc_bench,
     "lqrbank": lqr_bank, "rollouts": rollouts}[args.cmd](args)


if __name__ == "__main__":
    main()
