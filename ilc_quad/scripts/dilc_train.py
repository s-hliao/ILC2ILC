#!/usr/bin/env python3
"""Train a goal-conditioned state-feedback force policy from a few ILC'd goals: SAC with
the value-gradient update of Gurumurthy, Kolter & Manchester, "Deep Off-Policy Iterative
Learning Control" (L4DC 2023), many variants and seeds at once on every GPU.

The recommended setup, from the ILC's trials alone (the "explore with ILC, then
synthesize" setting: 5 goals x 2 seeds of nominal ILC, ~90 jumps, no rollouts):

    dilc_train.py OUT --goals f40_m85,f45_m85,f50_m85,f55_m85,f60_m85 --ilc-conds nominal \\
        --ilc-seeds 1 2 --online-episodes 0 --pretrain-updates 10000 --batch 512 \\
        --variants "sobaug3:rl=0,bc_scale=0,zeta_s=0,zeta_a=0,sob_v=1,sob_x=1,sob_g=1,sob_aug=3;\\
dilc:zeta_s=0,zeta_a=0,zeta_c=5,rl_h=1,sob_v=5,sob_x=5,sob_g=5,sob_aug=5"

What each trial contributes (dilc_execute.EpisodeMaker.ilc_labels), all from its own
linearization (its A_t, B_t, as JumpILC._step builds them) and the ILC's Q's:
  - hindsight: the trial twice, for the goal it aimed at and for where it landed
  - Sobolev labels for the actor: the forces for the label goal (u* = U + S dg, S the
    Stage III QP's step), d pi/d e = -K_t (the trial's Riccati gains: Stage III Qe on the
    landing, the stages' tracking rows on the way, R from Qu), d pi/d g (via S and K),
    and the outcome metric G_N for the value term; held over a neighbourhood of the trial
    by its own gain (sob_aug)
  - co-states for the critic (zeta_c): the trial's exact value gradients, the targets the
    one-step value-gradient bootstrap (zeta_s, zeta_a) is meant to approximate -- offline it
    drifts until the early-stance action gradients point against the ILC's
  - the QP's curvature R + B'PB (rl_h): the SAC actor's step on the critic is taken
    against it, an ILC-sized step

    dilc_train.py OUT                                   # everything, defaults below
    dilc_train.py OUT --goals f40_m85,f50_m85,f60_m85 --ilc-conds challenging
    dilc_train.py OUT --stages train,eval --variants "goal:zeta_g=0.5;sac:zeta_s=0,zeta_a=0"

Runs on the host (torch, CUDA); every jump flies in the ilc_quad container through
dilc_execute.py, which also executes the result (see there). OUT must lie inside the
colcon workspace mounted into the container. Stages (--stages, comma list):

  bank     OUT/bank.json: the goals' TO plans (experiments/go1/plans/ref_s8_go1_<goal>.npz),
           the observation/action scales, the ILC's Qe.
  ilc      the ILC itself, on each goal under each --ilc-conds reality (the asynchronous
           model, and the model errors it found hard): OUT/ilc/<goal>__<cond>/, 20 trials.
  dataset  every ILC trial -> an MDP episode (dilc_execute.py dataset): OUT/ilc_episodes.npz.
  train    one learner process per GPU ("group"), each training every --variants member
           x --seeds-per-group side by side as one batched network (a small MLP alone
           leaves the GPU idle on kernel launches; 12 members at batch 1024 cost about what
           one did), from one replay shared by the group: the ILC's trials and the group's
           own rollouts (--workers in the container, flown round-robin by its members).
  eval     every member snapshot zero-shot on the held-out goals (the midpoints between
           the ILC'd ones) under every challenging condition; the manifest's best by it.
           dilc_execute.py ilc then counts the ILC's own trials on the same robots.

Variants (per member): zeta_s, zeta_a (the value-gradient weights, 0: plain SAC), zeta_g
(goal gradient, below), secant (the Jacobians' secant correction, below), hist (the state
history input), bc_scale (x the Qu trust region).

The MDP. One step per TO contact sample k < Nc (10 ms; the forces are held over it as the
ILC's are): observation [(x_k - x_ref(g)[k]) / sx, k, g], x the SRB state measured causally
from the mocap and joints, g the CoM displacement goal; action the four contact forces'
offset from the (interpolated) plan's, u = u_ref(g) + a_scale * a, |a| < 1. The flight
and landing follow the last step. Reward: minus the ILC's own stage costs (x - x_ref)' Qe
(x - x_ref) -- Stage I on the contact samples, Stage II from rear-leg contact through the
flight, Stage III on the landing sample against the goal -- weighed as the ILC schedules
its stages: Stage I first, then II, then III, with a small share of the other two kept
(--stage-schedule, --stage-floor), minus a fall penalty.

The value-gradient update (their eqs. 13-16). Next to the TD loss, each critic's gradient
in the state and in the action is fit to the Bellman equation differentiated through the
approximate simulator -- here the ILC's own SRB model, linearized along the recorded
trial exactly where the ILC linearizes it (its A_t, B_t, stored with every transition):

    d_s Q(s,a) <- d_s r + gamma A_t' d_s' [Q'(s',a') - alpha log pi(a'|s')]
    d_a Q(s,a) <- d_a r + gamma B_t' d_s' [ ... ]

as a Huber loss on the residual with its threshold at the running median (their hinge),
weighted zeta * |grad Lq| / |grad Ls| (their eq. 18, refreshed every 20 updates). The
rewards' gradients come from the same A_t, B_t (the stage costs are costs of the next
state, the landing's through the ballistic flight).

Goal gradient (zeta_g). The Bellman equation differentiated in the goal too: with the
state error and action held, a goal change moves the next state error by
C_t = A_t dx_ref/dg + B_t du_ref/dg - dx_ref'/dg (the interpolated plan's slopes; the ILC's
own model), and the landing cost by its target; d_g Q is fit to d_g r + gamma (C_t' d_e' V'
+ d_g' V'), along the goal directions the ILC'd goals span. This is what the interpolation
between goals rests on; the plain update supervises nothing across goals.

Secant correction (secant). The SRB model's landing sensitivity to the forces is off
(pitch 5-10x, per JumpILC's secant note); the ILC's trial pairs measure how. A ridge fit of
the landing residuals over all its runs (secant_fit: e.g. 11.8 -> 3.9 deg rms pitch on
held-out runs) is folded back into each step's B_t's velocity rows (secant_dB), so the
value gradients use the corrected sensitivities.

History (hist). The last --hist state errors and the previous action in the observation
(zeros for members without), so the policy can tell e.g. a heavy robot from weak motors
by how the error has grown; the value-gradient targets follow the history's shift.

The Qs of the learning stage. Qe is the reward, stage by stage as above. Qu -- in the ILC
the price of a force step between trials, its trust region -- becomes the actor's trust
region around what the ILC learned: a penalty r_scale * Qu * |u_pi(s) - u_ILC(s)|^2 on the
states of each ILC run's best trial (behaviour cloning weighted by the ILC's own Qu), so
the policy leaves the ILC's forces only where the critic finds it worth that price.

Data. The offline buffer holds every ILC trial (all stages, falls included -- off-policy),
and stays; online rollouts add domain-randomized jumps (dilc_execute.sample_condition)
at goals drawn over the ILC'd goals' hull. Workers run the jumps while the learner
updates (asynchronous: a rollout flies the newest weights when it starts).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import selectors
import shlex
import struct
import subprocess
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONTAINER = "ilc_quad"
CWS = "/ilc_ws"                      # the workspace inside the container
CHALLENGING = ["nominal", "heavy15", "weak85", "curvesag", "payload2", "softgnd", "mu05",
               "real_s1"]           # dilc_execute.CHALLENGING
ASYNC = ("--act-delay 0.002 --act-jitter 0.003 --cmd-drop 0.02 --state-jitter 0.004 "
         "--pose-rate 240 --pose-delay 0.006 --pose-noise 0.0005 0.003 "
         "--param pose_latency:=0.006 --joint-noise 0.002 0.3 --foot-sensor 40 0.3 5")
COND_EXTRA = {"nominal": "", "heavy15": "--mass-scale 1.15", "weak85": "--motor-scale 0.85",
              "curvesag": "--motor-curve --motor-speed-scale 0.85", "payload2": "--payload 2.0",
              "softgnd": "--ground 2e3 5e2", "mu05": "--friction 0.5",
              "real_s1": "--mass-scale 1.05 --com-offset 0.01 0 --motor-curve --joint-friction 0.2",
              "light10": "--mass-scale 0.9", "comfwd": "--com-offset 0.02 0",
              "comback": "--com-offset -0.02 0", "curve": "--motor-curve",
              "jfric": "--joint-friction 0.3", "hardgnd": "--ground 2e4 3e3"}


# -- plumbing to the container ----------------------------------------------------------------
class Container:
    def __init__(self, ws_host: str, name=CONTAINER):
        self.ws_host, self.name = os.path.abspath(ws_host), name
        self.direct = subprocess.run(["docker", "ps"], capture_output=True).returncode == 0

    def path(self, p: str) -> str:
        p = os.path.abspath(p)
        if not p.startswith(self.ws_host + os.sep):
            raise ValueError(f"{p} is outside the workspace {self.ws_host} (not in the container)")
        return CWS + p[len(self.ws_host):]

    def cmd(self, script: str, interactive=False, domain=150) -> list[str]:
        inner = (f"cd {CWS} && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 "
                 f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                 f"ROS_DOMAIN_ID={1 + domain % 98} && {script}")
        d = ["docker", "exec"] + (["-i"] if interactive else []) + [self.name, "bash", "-c", inner]
        return d if self.direct else ["sg", "docker", "-c", shlex.join(d)]

    def run(self, script: str, domain=150, **kw):
        return subprocess.run(self.cmd(script, domain=domain), **kw)


def exe(name):
    return f"python3 {CWS}/install/ilc_quad/lib/ilc_quad/{name}"


# -- stages: bank, ilc, dataset ------------------------------------------------------------------
def stage_bank(a, box):
    plans = []
    for name in a.goals.split(","):
        path = os.path.join(a.plans, f"ref_{a.plan_prefix}{name}.npz")
        cfg = json.loads(str(np.load(path)["config"]))
        plans.append(dict(name=name, path=box.path(path), goal=cfg["jump"], box=cfg["box"],
                          margin=cfg["margin"]))
    G = np.array([p["goal"] for p in plans], float)
    center = G.mean(0)
    scale = np.maximum(0.5 * (G.max(0) - G.min(0)), 0.05)
    phases = json.loads(str(np.load(os.path.join(a.plans, f"ref_{a.plan_prefix}{plans[0]['name']}.npz"))["config"]))["phases"]
    # the goal directions the bank spans, in the policy's goal coordinates: goal gradients
    # are supervised along these only (a line of flat jumps says nothing about height)
    Dg = (G - G[0]) / scale
    if np.linalg.matrix_rank(Dg, tol=1e-6) <= 1:
        u = Dg[np.argmax(np.linalg.norm(Dg, axis=1))]
        u = u / np.linalg.norm(u)
        proj = np.outer(u, u)
    else:
        proj = np.eye(2)
    dt = json.loads(str(np.load(os.path.join(a.plans, f"ref_{a.plan_prefix}{plans[0]['name']}.npz"))["config"]))["dt"]
    bank = dict(plans=plans, phases=phases, dt=dt, sx=a.sx, a_scale=a.a_scale,
                g_center=center.tolist(), g_scale=scale.tolist(), goal_proj=proj.tolist(),
                qe=a.qe, qu3=a.qu, k_rscale=a.k_rscale, k_crun=a.k_crun, gamma=a.gamma,
                r_scale=1e4, est_window=a.est_window,
                hist=a.hist,
                prev_action=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "bank.json"), "w") as f:
        json.dump(bank, f, indent=1)
    print(f"bank: {[p['name'] for p in plans]} goals {G.tolist()}")
    return bank


def ilc_conds(a):
    return CHALLENGING if a.ilc_conds == "challenging" else a.ilc_conds.split(",")


def stage_ilc(a, box, bank):
    from concurrent.futures import ThreadPoolExecutor
    root = os.path.join(a.out, "ilc")
    os.makedirs(root, exist_ok=True)
    jobs = []
    for p in bank["plans"]:
        for c, seed in [(c, sd) for c in ilc_conds(a) for sd in a.ilc_seeds]:
            # the first seed's run is <goal>__<cond>; more seeds (more coverage of the same
            # goal: another robot's noise and another path of the ILC) add _s<seed>
            run = f"{p['name']}__{c}" + ("" if seed == a.ilc_seeds[0] else f"_s{seed}")
            if os.path.exists(os.path.join(root, run + ".json")):
                continue
            boxarg = f"--box {p['box']['x_front']} {p['box']['height']}" if p["box"] else ""
            script = (f"{exe('ilc_jump_lockstep.py')} --robot go1 --max-trials {a.ilc_trials} "
                      f"--jump {p['goal'][0]} {p['goal'][1]} {boxarg} "
                      f"--param phases:=[30,30,30] --param margin:={p['margin']} "
                      f"--reference-file {p['path']} {ASYNC} {COND_EXTRA[c]} --seed {seed} "
                      f"--log-dir {box.path(root)} --run-name {run} "
                      f"--summary {box.path(os.path.join(root, run + '.json'))} "
                      f"> {box.path(os.path.join(root, run + '.log'))} 2>&1")
            jobs.append((run, script))
    print(f"ilc: {len(jobs)} runs of {a.ilc_trials} trials ({a.jobs} at a time)")
    t0 = time.time()

    def go(job):
        i, (run, script) = job
        r = box.run(script, domain=100 + i % 100)
        return run, r.returncode

    with ThreadPoolExecutor(a.jobs) as pool:
        for run, rc in pool.map(go, enumerate(jobs)):
            print(f"  {run} done ({rc}) {time.time() - t0:.0f} s", flush=True)
    # what the ILC converged to, per goal, under the nominal (async) reality: the
    # no-network zero-shot baseline interpolates these
    write_ilc_best(a, box, bank)
    print(ilc_table(root))


def write_ilc_best(a, box, bank):
    """policy/ilc_U.npz (the forces) and policy/ilc_best.json (the trial files) of each
    goal's best nominal ILC trial: the no-network baselines fly these, interpolated."""
    root = os.path.join(a.out, "ilc")
    U, paths = {}, {}
    for p in bank["plans"]:
        best = _best_trial(os.path.join(root, f"{p['name']}__nominal"), bank)
        if best is not None:
            U[p["name"]], paths[p["name"]] = best[0], box.path(os.path.realpath(best[1]))
    pol = os.path.join(a.out, "policy")
    os.makedirs(pol, exist_ok=True)
    if len(U) == len(bank["plans"]):
        np.savez(os.path.join(pol, "ilc_U.npz"), **U)
        with open(os.path.join(pol, "ilc_best.json"), "w") as f:
            json.dump(paths, f, indent=1)


def _best_trial(run_dir, bank):
    import glob
    best, best_c = None, np.inf
    qe = np.asarray(bank["qe"][:3])
    for p in sorted(glob.glob(os.path.join(run_dir, "trial_*.npz"))):
        d = np.load(p)
        if bool(d["log_fell"]):
            continue
        cfg = json.loads(str(d["config"]))
        target = d["x_ref"][0, :2] + np.array(cfg["jump"])
        e = np.array([*(d["log_X"][-1, :2] - target), d["log_X"][-1, 2]])
        c = float(e @ (qe * e))
        if c <= best_c:
            best, best_c = (d["U_flown"].copy(), p), c
    return best


def ilc_table(root):
    import glob
    lines = ["ILC runs (last trial landing, falls):"]
    for js in sorted(glob.glob(os.path.join(root, "*.json"))):
        h = json.load(open(js))["history"]
        if not h:
            continue
        fp = "".join("F" if r.get("fell") else "." for r in h)
        l = h[-1]
        lines.append(f"  {os.path.basename(js)[:-5]:26s} {fp:22s} last {l['pos_err'] * 100:4.1f} cm "
                     f"{math.degrees(l['theta_err']):4.1f} deg conv={any(r['converged'] for r in h)}")
    return "\n".join(lines)


def stage_secant(a, box, bank):
    """Fit the secant correction of the landing sensitivity on the ILC's trials and name it
    in the bank, so the dataset's labels (S, the outcome metric) use G_N + C."""
    off = load_offline(a)
    C = secant_fit(off, bank)
    if C is None:
        print("secant: no better than the model on held-out runs; labels unchanged")
        return bank
    path = os.path.join(a.out, "secant_C.npz")
    np.savez(path, C=C)
    bank = dict(bank, secant_C=box.path(path))
    for f in (os.path.join(a.out, "bank.json"), os.path.join(a.out, "policy", "bank.json")):
        if os.path.exists(os.path.dirname(f)):
            with open(f, "w") as fh:
                json.dump(bank, fh, indent=1)
    return bank


def stage_relabel(a, box, bank):
    """The new robot's ILC trials labeled with the structure they measured: the landing
    sensitivity's secant correction C fitted on their own trial pairs (--extra-data: the
    robot's exploration, as first labeled), then every label (the step S, the gains K, the
    co-states, the outcome metric) from G_N + C and B corrected to match (secant_full)
    -> --relabel-out. The training robot's labels are left as they are."""
    parts = [dict(np.load(f)) for f in a.extra_data]
    keys = set(parts[0]).intersection(*parts[1:]) if len(parts) > 1 else set(parts[0])
    off = {k: np.concatenate([p_[k] for p_ in parts]) for k in keys}
    C = secant_fit(off, bank, log=lambda m: print(f"relabel: {m}"))
    if C is None:                      # no better than the model: the labels stay as they are
        print("relabel: the secant does not predict held-out runs better; labels unchanged")
        np.savez(a.relabel_out, **off)
        return
    path = os.path.join(a.out, "relabel_C.npz")
    np.savez(path, C=C)
    bpath = os.path.join(a.out, "relabel_bank.json")
    with open(bpath, "w") as fh:
        json.dump(dict(bank, secant_C=box.path(path), secant_full=int(a.relabel_mode == "full")), fh,
                  indent=1)
    runs = " ".join(box.path(r) for r in a.relabel_runs)
    r = box.run(f"{exe('dilc_execute.py')} dataset --bank {box.path(bpath)} --out "
                f"{box.path(a.relabel_out)} --jobs {a.jobs} --runs {runs}", capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr[-3000:])
        raise SystemExit("relabel: dataset failed")
    d = np.load(a.relabel_out)
    print(f"relabel: {len(a.relabel_runs)} runs -> {len(d['obs'])} episodes in {a.relabel_out}")


def _np_actor(path):
    w = dict(np.load(path))
    def act(o):                                   # o (..., od) -> tanh mean (..., 4)
        h = o
        for i in range(int(w["n_hidden"])):
            h = np.maximum(h @ w[f"W{i}"].T + w[f"b{i}"], 0.0)
        return np.tanh(h @ w["W_mu"].T + w["b_mu"])
    return act


def stage_explore(a, box, bank):
    """
    Explore with the ILC where the policy is least certain: over the goals' hull, the
    disagreement of the --explore-variant's members (independent seeds: the epistemic
    spread of the forces they would fly along each goal's plan) picks --explore-goals goals,
    greedily, at least a share of the hull apart; at each, a short ILC (Stage III,
    --explore-trials) starts from the forces the policy flies there, on --explore-cond's robot.
    Its trials -- directed, model-based exploration toward the goal -- become labeled episodes
    like the ILC's own (OUT/explore_<tag>.npz), for fine-tuning.
    """
    src = os.path.abspath(a.explore_from)
    man = json.load(open(os.path.join(src, "policy", "manifest.json")))
    mem = [m for m in man["members"] if m["variant"] == a.explore_variant]
    last = max(m["updates"] for m in mem)
    mem = [m for m in mem if m["updates"] == last]
    actors = [_np_actor(os.path.join(src, "policy", m["name"] + ".npz")) for m in mem]
    best = min((m for m in mem if m.get("score") is not None), key=lambda m: m["score"],
               default=mem[0])["name"]
    G = np.array([p["goal"] for p in bank["plans"]], float)
    d = G[-1] - G[0]
    cand = [G[0] + t * d for t in np.linspace(0, 1, 81)]
    Ndc, Nsc, _ = bank["phases"]
    Nc = Ndc + Nsc
    od = 9 + 6 * int(bank.get("hist", 0)) + (4 if bank.get("prev_action") else 0)
    mask = np.ones((Nc, 4))
    mask[Ndc:, :2] = 0
    gc, gsc = np.asarray(bank["g_center"]), np.asarray(bank["g_scale"])
    acq = []
    for g in cand:
        o = np.zeros((Nc, od))
        o[:, 6] = 2 * np.arange(Nc) / Nc - 1
        o[:, 7:9] = (g - gc) / gsc
        A = np.stack([f(o) for f in actors]) * mask * np.asarray(bank["a_scale"])
        acq.append(float(A.std(0).mean()))
    acq = np.array(acq)
    if a.explore_acq == "critic":
        # the value function's own prediction: the landing cost the critics expect the policy
        # to incur at each goal on this robot (their value at the jump's start), plus their
        # spread across the members (an upper confidence bound on the cost)
        import glob
        import torch
        Vg = []
        for ck_path in sorted(glob.glob(os.path.join(src, "train", "g*_final.pt"))):
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            names = [n for n, _ in ck["variants"]]
            W = [ck["critic"][f"ls.{i}.W"].numpy() for i in range(3)]
            Bc = [ck["critic"][f"ls.{i}.b"].numpy() for i in range(3)]
            for mi, n in enumerate(names):
                if n.split(".")[0] != a.explore_variant:
                    continue
                gname = os.path.basename(ck_path)[:-len("_final.pt")]
                f = _np_actor(os.path.join(src, "policy", f"{gname}_{n}_u{last}.npz")) \
                    if os.path.exists(os.path.join(src, "policy", f"{gname}_{n}_u{last}.npz")) else None
                if f is None:
                    continue
                vals = []
                for g in cand:
                    o = np.zeros(od)
                    o[6] = -1.0
                    o[7:9] = (g - gc) / gsc
                    act = f(o[None])[0] * mask[0]
                    x = np.concatenate([o * (np.arange(od) < 9), act])
                    qs = []
                    for t in (2 * mi, 2 * mi + 1):
                        h = np.tanh(x @ W[0][t] + Bc[0][t][0])
                        h = np.tanh(h @ W[1][t] + Bc[1][t][0])
                        qs.append(float((h @ W[2][t] + Bc[2][t][0])[0]))
                    vals.append(min(qs))
                Vg.append(vals)
        Vg = np.array(Vg)                                  # (members, goals): values (-cost)
        cost, spread = -Vg.mean(0), Vg.std(0)
        acq = cost + a.explore_kappa * spread
        print(f"explore {a.explore_tag}: critic-predicted cost over the hull {cost.min():.2f}.."
              f"{cost.max():.2f}, spread {spread.mean():.2f} ({len(Vg)} critics)")
    picks, spacing = [], 1.0 / (2 * a.explore_goals)
    if a.explore_acq == "cover":
        # information-seeking with few goals: greedily the goal that is both uncertain (the
        # members' disagreement, scaled to [0, 1]) and far from every goal the ILC already ran
        # on this robot (--explore-data) or picked now (distance in hull lengths): trial pairs
        # at new places are what tell the secant something new
        done = [np.asarray(g, float) for f in a.explore_data
                for g in np.load(f)["goal"][np.load(f)["relabel"] < 0.5]]
        u = (acq - acq.min()) / max(acq.max() - acq.min(), 1e-9)
        L = float(np.linalg.norm(d))
        for _ in range(a.explore_goals):
            dist = np.array([min([np.linalg.norm(c - g) for g in done] + [L]) / L for c in cand])
            # and never within half a spacing of a goal already run or picked
            score = np.where(dist >= 0.5 / a.explore_goals, u + a.explore_kappa * dist, -np.inf)
            i = int(np.argmax(score))
            picks.append((i / (len(cand) - 1), i))
            done.append(cand[i])
    for i in (np.argsort(-acq) if a.explore_acq != "cover" else []):
        t = i / (len(cand) - 1)
        if all(abs(t - p) >= spacing for p, _ in picks):
            picks.append((t, i))
        if len(picks) == a.explore_goals:
            break
    goals = [cand[i] for _, i in sorted(picks)]
    if a.explore_goal_list:                  # given: the few goals the real robot's ILC runs at
        goals = [np.array([float(x), 0.0]) for x in a.explore_goal_list]
    print(f"explore {a.explore_tag}: {len(actors)} members of {a.explore_variant}; "
          f"disagreement (N) over the hull {acq.min():.2f}..{acq.max():.2f}; goals "
          + " ".join(f"{g[0]:.3f}" for g in goals) + f"; ILC from {best}'s jumps")
    out = os.path.join(a.out, f"explore_{a.explore_tag}")
    jumps = " ".join(f"--jump {g[0]:.4f} {g[1]:.4f}" for g in goals)
    sec = ""
    if a.explore_secant:
        # the exploration ILC plans Stage III with the landing sensitivity corrected by a
        # ridge fit over the trials flown so far on this robot (its earlier exploration;
        # before any, the training robot's ILC runs): JumpILC's secant, pooled across runs
        parts = [dict(np.load(os.path.join(a.out, "ilc_episodes.npz")))] + \
            [dict(np.load(f)) for f in a.explore_data]
        keys = set(parts[0]).intersection(*parts[1:])
        off = {k: np.concatenate([p_[k] for p_ in parts]) for k in keys}
        on_robot = np.char.find(off["run"].astype(str), a.explore_cond) >= 0
        C = secant_fit(off, bank, log=lambda m: print(f"explore {a.explore_tag}: {m}"),
                       run_filter=a.explore_cond if on_robot.sum() >= 8 else None)
        if C is not None:
            path = os.path.join(a.out, f"secant_prior_{a.explore_tag}.npz")
            np.savez(path, C=C)
            sec = f"--secant-prior {box.path(path)} "
    if a.explore_member:                     # e.g. mean:VARIANT@N, the members' ensemble
        best = a.explore_member
    r = box.run(f"{exe('dilc_execute.py')} ilc --policy {box.path(os.path.join(src, 'policy'))} "
                f"--member {best} {jumps} --conds {a.explore_cond} --episodes 1 --seed {a.explore_seed} "
                f"--start policy --max-trials {a.explore_trials} --out {box.path(out)} {sec}"
                f"--jobs {a.jobs}", capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("policy")))
    import glob
    runs = sorted(d_ for d_ in glob.glob(os.path.join(out, "*/"))
                  if os.path.exists(os.path.join(d_, "meta.json")))
    npz = os.path.join(a.out, f"explore_{a.explore_tag}.npz")
    r = box.run(f"{exe('dilc_execute.py')} dataset --bank {box.path(os.path.join(a.out, 'bank.json'))} "
                f"--out {box.path(npz)} --jobs {a.jobs} --runs " + " ".join(box.path(x) for x in runs),
                capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr[-3000:])
        raise SystemExit("explore: dataset failed")
    d_ = np.load(npz)
    print(f"explore {a.explore_tag}: {len(runs)} ILC runs, {int((d_['relabel'] < 0.5).sum())} "
          f"trials -> {len(d_['obs'])} labeled episodes in {npz}")


def stage_dataset(a, box):
    import glob
    runs = sorted(d for d in glob.glob(os.path.join(a.out, "ilc", "*/"))
                  if os.path.exists(os.path.join(d, "meta.json")))
    out = os.path.join(a.out, "ilc_episodes.npz")
    script = (f"{exe('dilc_execute.py')} dataset --bank {box.path(os.path.join(a.out, 'bank.json'))} "
              f"--out {box.path(out)} --jobs {a.jobs} --runs "
              + " ".join(box.path(r) for r in runs))
    t0 = time.time()
    r = box.run(script, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit("dataset failed")
    d = np.load(out)
    print(f"dataset: {d['obs'].shape[0]} ILC trials -> {d['obs'].shape[0] * d['act'].shape[1]} "
          f"transitions, {int(d['anchor'].sum())} anchor trials, {int(d['fell'].sum())} falls "
          f"({time.time() - t0:.0f} s); |a| > 0.9 in {np.mean(np.abs(d['act']) > 0.9):.1%} of actions")


# -- rollout workers ------------------------------------------------------------------------------
class Worker:
    """dilc_execute.py serve in the container, over its stdin/stdout."""

    batch = 1                                   # jobs per request

    def __init__(self, box, bank_path, log_path, domain):
        self.log = open(log_path, "w")
        self.p = subprocess.Popen(
            box.cmd(f"{exe('dilc_execute.py')} serve --bank {bank_path}", interactive=True,
                    domain=domain),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.job = None                         # {job id: (member, goal)} in flight

    def send(self, header, payload_bytes=b""):
        hb = json.dumps(header).encode()
        self.p.stdin.write(struct.pack("<I", len(hb)) + hb + struct.pack("<Q", len(payload_bytes))
                           + payload_bytes)
        self.p.stdin.flush()

    def _read(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError("worker exited")
            buf += chunk
        return buf

    def recv(self):
        (n,) = struct.unpack("<I", self._read(4))
        header = json.loads(self._read(n))
        (m,) = struct.unpack("<Q", self._read(8))
        payload = {}
        if m:
            with np.load(io.BytesIO(self._read(m))) as d:
                payload = {k: d[k] for k in d.files}
        return header, payload

    def close(self):
        try:
            self.send(dict(cmd="quit"))
            self.p.stdin.close()
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()
        self.log.close()


class GpuWorker(Worker):
    """ilc_mjx.serve on the host: a batch of nominal jumps at once on one GPU (MJX), serve's protocol."""

    def __init__(self, bank_path, log_path, gpu, batch):
        self.log = open(log_path, "w")
        root = os.environ.get("ILC_MJX_SRC", os.path.join(os.path.dirname(HERE), "..", "ilc_mjx"))
        py = os.environ.get("ILC_MJX_PYTHON", os.path.expanduser("~/miniconda3/envs/ilcmjx/bin/python"))
        self.p = subprocess.Popen([py, "-m", "ilc_mjx.serve", "--bank", bank_path, "--gpu", str(gpu)],
                                  cwd=os.path.abspath(root), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.job = None
        self.batch = int(batch)


# -- the secant correction of the SRB model's landing sensitivities ---------------------------
def _raw_jacobians(A, B, bank):
    """Normalized (as stored) -> SI one-step Jacobians."""
    D = np.asarray(bank["sx"], float)
    asc = np.asarray(bank["a_scale"], float)
    return A * D[:, None] / D[None, :], B * D[:, None] / asc[None, :]


def _landing_maps(A_raw, bank):
    """Phi[k] = d x_N / d x_{k+1} for k = 0..Nc-1: the one-step maps from k+1 to the end of
    contact, then the ballistic flight."""
    Ndc, Nsc, Nfl = bank["phases"]
    Nc, dt = Ndc + Nsc, float(bank.get("dt", 0.01))
    Af = np.eye(6)
    Af[:3, 3:] = dt * np.eye(3)
    P = np.linalg.matrix_power(Af, Nfl)
    Phi = np.zeros((Nc, 6, 6))
    for k in range(Nc - 1, -1, -1):
        Phi[k] = P
        P = P @ A_raw[k]
    return Phi


def load_offline(a):
    """The ILC's trials (OUT/ilc_episodes.npz), plus any --extra-data (the policy's labeled
    rollouts), as one set of per-episode arrays."""
    parts = [dict(np.load(os.path.join(a.out, "ilc_episodes.npz")))]
    for path in a.extra_data:
        parts.append(dict(np.load(path)))
    # dom: 0 for the training robot's ILC trials, 1 for --extra-data (the new robot's)
    for i, p in enumerate(parts):
        p["dom"] = np.full(len(p["act"]), float(i > 0), np.float32)
    keys = set(parts[0]).intersection(*parts[1:]) if len(parts) > 1 else set(parts[0])
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def secant_fit(off, bank, log=print, run_filter=None):
    """
    The ILC's secant (Broyden) correction, fitted once over all its trials: the SRB model's
    landing sensitivity to the forces, G_N (6 x Nc*4, from each trial's A_t, B_t), misses
    what consecutive trials of a run actually did, Delta x_N - G_N Delta U. A correction C
    to G_N is fitted to those residuals (ridge, the weight chosen by leave-one-run-out) --
    what JumpILC's secant option learns within one run, pooled over every goal and robot.
    Returns C, or None when it does not predict held-out runs better than the model.
    Only the ILC's own trials, as flown (no hindsight copies, no policy rollouts).
    """
    keep = np.ones(len(off["act"]), bool)
    if "relabel" in off:
        keep &= off["relabel"] < 0.5
    keep &= ~np.char.startswith(off["run"].astype(str), "rollout")
    if run_filter:                            # e.g. only the new robot's exploration runs
        keep &= np.char.find(off["run"].astype(str), run_filter) >= 0
    off = {k: v[keep] for k, v in off.items()}
    E, Nc = off["act"].shape[:2]
    asc = np.asarray(bank["a_scale"], float)
    G = np.zeros((E, 6, Nc * 4))
    for e in range(E):
        A_raw, B_raw = _raw_jacobians(off["A"][e], off["B"][e], bank)
        Phi = _landing_maps(A_raw, bank)
        for k in range(Nc):
            G[e, :, 4 * k:4 * k + 4] = Phi[k] @ B_raw[k]
    runs = off["run"]
    dU, R, grp = [], [], []
    for r in sorted(set(runs)):
        idx = [i for i in np.flatnonzero(runs == r) if not off["fell"][i]]
        idx.sort(key=lambda i: int(off["trial"][i]))
        for i, j in zip(idx[:-1], idx[1:]):
            du = (off["U"][j] - off["U"][i]).reshape(-1)
            if np.abs(du).max() < 1e-3:
                continue
            dx = off["logX"][j][-1] - off["logX"][i][-1]
            dU.append(du)
            R.append(dx - G[i] @ du)
            grp.append(r)
    dU, R, grp = np.array(dU), np.array(R), np.array(grp)
    qe = np.asarray(bank["qe"], float)
    fit = lambda X, Y, lam: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y).T
    base = float(np.mean((R ** 2) @ qe))
    best = None
    for lam in (1e1, 1e2, 1e3, 1e4, 1e5):
        err = []
        for r in set(grp):
            tr, te = grp != r, grp == r
            C = fit(dU[tr], R[tr], lam)
            err.append(((R[te] - dU[te] @ C.T) ** 2) @ qe)
        e = float(np.mean(np.concatenate(err)))
        if best is None or e < best[1]:
            best = (lam, e)
    lam, e = best
    per_row = lambda Y: np.sqrt(np.mean(Y ** 2, 0))
    C = fit(dU, R, lam)
    Rcv = []
    for r in set(grp):
        tr, te = grp != r, grp == r
        Rcv.append(R[te] - dU[te] @ fit(dU[tr], R[tr], lam).T)
    Rcv = np.concatenate(Rcv)
    log(f"secant: {len(dU)} trial pairs from {len(set(grp))} runs; SRB landing prediction "
        f"rms x {per_row(R)[0] * 100:.2f} cm z {per_row(R)[1] * 100:.2f} cm pitch "
        f"{np.degrees(per_row(R)[2]):.2f} deg -> corrected (held-out runs) x "
        f"{per_row(Rcv)[0] * 100:.2f} cm z {per_row(Rcv)[1] * 100:.2f} cm pitch "
        f"{np.degrees(per_row(Rcv)[2]):.2f} deg (ridge {lam:g}); Qe-cost {base:.2e} -> {e:.2e}")
    return C if e < base else None


def secant_dB(A, B, C, bank, amask):
    """Per-step corrections of the stored (normalized) B_t that move the landing sensitivity
    toward G_N + C. A force acts on the next state through its velocities (the position
    rows' share is dt^2 small), so each sample's correction goes into the velocity rows
    alone: dB_k[3:] = argmin |Phi_k[:, 3:] dB - C_k|_Qe, Phi_k the SRB map from x_{k+1} to
    the landing. (Solving Phi_k dB = C_k exactly instead puts position jumps 100x B's own
    size into the position rows, and the critics fitted to them diverged.)"""
    D = np.asarray(bank["sx"], float)
    asc = np.asarray(bank["a_scale"], float)
    W = np.diag(np.asarray(bank["qe"], float))
    A_raw, _ = _raw_jacobians(A, B, bank)
    Phi = _landing_maps(A_raw, bank)
    Nc = A.shape[0]
    dB = np.zeros((Nc, 6, 4), np.float32)
    for k in range(Nc):
        P = Phi[k][:, 3:]
        d = np.zeros((6, 4))
        d[3:] = np.linalg.solve(P.T @ W @ P, P.T @ W @ C[:, 4 * k:4 * k + 4])
        dB[k] = d / D[:, None] * asc[None, :] * amask[k][None, :]
    return dB


# -- learners: one process per GPU, every variant in it as a batched ensemble ----------------------
VARIANT_KEYS = dict(zeta_s=0.5, zeta_a=0.5, zeta_g=0.0, secant=0.0, hist=0.0, bc_scale=1.0,
                    rl=1.0, bc_all=0.0, sob_v=0.0, sob_x=0.0, sob_g=0.0, hind=1.0,
                    vg_open=0.0, zeta_c=0.0, sob_aug=0.0, rl_h=0.0, anchor_ref=0.0, awr=0.0,
                    sec_c=0.0, cl_co=0.0, sec_cl=0.0, crit_t=0.0, tfrac=-1.0, nomw=1.0, co_fd=0.0,
                    co_pg=0.0, co_clip=5.0, co_ilc=0.0, co_beta=0.5, co_max=0.1, co_gn=0.0, co_delta=0.1)
# co_gn=1 (with co_ilc): in Stage III the ILC step is the Gauss-Newton step on the landing error e (x, z, pitch)
# through the closed-loop landing sensitivity S = d x_N / d a: -co_beta S' (S S' + co_delta tr/3 I)^-1 e
# (JumpILC's Stage III), rms capped at co_max; the earlier stages keep the normalized-gradient step
# co_ilc > 0: the ILC's own update, fitted for the network: on each of the latest episodes the trial's ILC step,
# a* = a - co_beta J g / |g|^2 (the normalized-gradient step on its discounted stage cost J, g = d J / d a from
# its closed-loop co-states), its rms capped at co_max, and the actor regresses onto a* (rl=0: no SAC term) --
# every batch of rollouts one ILC iteration, the network amortizing the steps across goals
# co_pg > 0: the actor's gradient from the co-states themselves (the ILC's update lifted into the policy's
# parameters): d J/d theta = sum_k (d J/d a_k) d pi(o_k)/d theta, d J/d a_k = B_k' m_k the closed-loop co-state
# under the current actor, on the --co-recent latest episodes (rl=0: no SAC term). co_clip: a co-state larger
# than co_clip x the median is scaled down to it (the ILC's step limit; unstable chains over the jump)
# co_fd=1: the critic's co-state targets (zeta_c) on the online rollouts -- each episode's closed-loop
# co-states from its own (FD) Jacobians and the CURRENT actor's feedback d pi / d e, recomputed every
# --costate-every updates (the ILC's per-trial co-states, under the policy being trained)
# rl=0: the actor only clones the ILC's best trials (no critic). bc_all=1: the Qu trust region
# holds the actor near the forces of every ILC trial at its states (TD3+BC-like), not only
# near the best trials. sob_v, sob_x, sob_g: the Sobolev terms from each trial's own A, B
# (value through G_N, d pi/d e -> -K_t, d pi/d g -> the Stage III step S), weights relative
# to the SAC actor gradient (eq. 18 style). hind=0: the hindsight-relabeled episodes are
# left out of this member's losses. vg_open=1: the value-gradient bootstrap holds the next
# action constant (their eq. 6 allows a'(s') = a constant a'): d/ds' Q(s', a') without the
# chain through the actor's feedback d pi/ds'. zeta_c: the critic's state and action
# gradients fit to each ILC trial's own co-states (its A_t, B_t, run backwards from its
# measured errors): the value gradients along the data without bootstrapping through the
# critic. Offline, the one-step bootstrap (zeta_s, zeta_a) is fit almost exactly, yet its
# targets drift from the trials' own A, B chain over the ~60 steps until the early-stance
# action gradients point against the ILC's (cos -0.6 to -0.7). sob_aug: the Sobolev value
# term also at states around each trial, its label carried there by the trial's own gain
# (pi(x + d) = u* - K d: a derivative label held over a neighbourhood). rl_h=1: the SAC
# actor term is taken against the trial's QP curvature R + B'PB around its own forces
# (an ILC-sized step on the critic; without it the actor climbs the critic's gradient
# unbounded, where the SRB model no longer holds). anchor_ref=1: that trust region is
# centred on the actor as it was when RL switched on (--rl-start: the Sobolev network), not
# on the flown forces -- RL adds only what the critic sees beyond the ILC's step; 2: on the
# labels (the ILC step itself); 3: on a slow average of the actor (a moving trust region,
# for fine-tuning in a new environment, where the starting policy is the one to leave). awr: an
# advantage-weighted pull toward the label and the flown forces (the critic's values used
# only at actions in the data). sec_c=1: the critic's landing co-states use the secant-
# corrected landing sensitivity (G_N + C, C fitted from the ILC's own trial pairs). cl_co=1:
# the co-states under the trial's feedback gains, q_k = grad c + gamma (A - B K)' q_k+1 --
# the value gradient of a feedback policy (iLQR/DDP's V_x), consistent with the step's
# closed-loop curvature R + B'PB; open-loop co-states make every sample's step correct the
# whole landing error on its own. sec_cl=1: those closed-loop co-states with each step's B
# corrected by the secant measured on the robot's own trials (--secant-runs: e.g. the new
# robot's exploration ILC runs) -- the approximate simulator's Jacobians improved by data,
# for the critic only (the Sobolev labels keep the nominal model). crit_t=1: the critic
# (its TD values and co-states) fits only the new robot's transitions (--extra-data): the
# value of the policy on the robot it is deployed on, not a mixture with the training robot's.
# tfrac: this member's share of every batch from --extra-data (< 0: --target-frac). nomw < 1:
# the training robot's labels (Sobolev terms, cloning) weigh only nomw near the goals the new
# robot's ILC ran, 1 - (1 - nomw) exp(-(d / --nomw-sigma)^2) at goal distance d: the sim's
# structure kept where the new robot has no data, the new robot's own labels where it has
DEFAULT_VARIANTS = ("base:;goal:zeta_g=0.5;secant:secant=1;hist:hist=1;"
                    "all:zeta_g=0.5,secant=1,hist=1;sac:zeta_s=0,zeta_a=0")


def parse_variants(spec):
    out = []
    for item in filter(None, spec.split(";")):
        name, _, kvs = item.partition(":")
        v = dict(VARIANT_KEYS)
        for kv in filter(None, kvs.split(",")):
            k, _, x = kv.partition("=")
            if k not in v:
                raise ValueError(f"unknown variant key {k!r} (one of {list(v)})")
            v[k] = float(x)
        out.append((name, v))
    return out


def group_main(gi, gpu, a, variants, bank, C_sec, C_cl=None):
    """
    One GPU: M members (the variants) trained side by side as one batched network each for
    the actor and the twin critics -- every member its own parameters, optimizer state,
    entropy weight, value-gradient weights and hinge -- from one shared replay: the ILC's
    trials plus this group's rollouts, flown round-robin by its members. Sharing the data
    makes the variants' comparison one of learning from the same experience.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.cuda.set_device(gpu)
    dev = torch.device(f"cuda:{gpu}")
    cfg = dict(vars(a))
    M = len(variants)
    seed = cfg["seed"] + 1000 * gi
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gname = f"g{gi}"
    names = [f"g{gi}_{n}" for n, _ in variants]          # n: variant name + seed, e.g. goal.s1
    V = {k: torch.tensor([v[k] for _, v in variants], dtype=torch.float32, device=dev)
         for k in VARIANT_KEYS}
    tdir, pdir = os.path.join(a.out, "train"), os.path.join(a.out, "policy")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    logf = open(os.path.join(tdir, f"{gname}.log"), "w")
    T0 = time.time()

    def log(msg):
        line = f"[{gname} gpu{gpu} {time.time() - T0:6.0f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    asc = torch.tensor(bank["a_scale"], dtype=torch.float32, device=dev)
    off = load_offline(a)
    E_off, Nc = off["act"].shape[:2]
    Ndc = int(bank["phases"][0])
    od, ad = off["obs"].shape[2], off["act"].shape[2]
    H = int(bank.get("hist", 0))
    PA = bool(bank.get("prev_action", False))
    i_hist = 9                                   # [err 6 | time | goal 2 | hist 6H | prev a 4]
    i_pa = 9 + 6 * H
    amask_np = np.ones((Nc, ad), np.float32)
    amask_np[Ndc:, 0:2] = 0.0
    # members without history see zeros there (and are exported so)
    omask = torch.ones((M, od), device=dev)
    for m, (_, v) in enumerate(variants):
        if not v["hist"]:
            omask[m, i_hist:] = 0.0
    Pg = torch.tensor(bank["goal_proj"], dtype=torch.float32, device=dev)

    # replay -------------------------------------------------------------------------------------
    KEYS = dict(o=od, a=ad, o2=od, rc=3, gn=18, A=36, B=6 * ad, dB=6 * ad, Cg=12, gg=6,
                done=1, fell=1, vg=1, anchor=1, mask=ad, tgt=ad, Kx=6 * ad, Kg=2 * ad,
                sup=1, relabel=1, mca=3 * ad, mcs=18, Hu=ad * ad, mcac=3 * ad,
                mcacl=3 * ad, mcscl=18, mcacl2=3 * ad, mcscl2=18, dom=1, dgr=1,
                cla=M * 3 * ad, cls=M * 18, cot=M * ad, e3=3, mub=ad)

    class Replay:
        def __init__(self, cap):
            self.cap, self.n, self.i, self.n_online = cap, 0, 0, 0
            self.d = {k: torch.zeros((cap, w), device=dev) for k, w in KEYS.items()}
            self.ep_starts = []                       # online episodes (whole, contiguous)

        def add(self, ep, anchor=0.0, online=False, dom=0.0, dgr=1e3):
            T = ep["act"].shape[0]
            fell = float(ep["fell"])
            done = np.zeros(T, np.float32)
            done[-1] = 1.0
            vg = np.ones(T, np.float32)
            if fell:
                vg[-1] = 0.0            # a fall is a discontinuity: no gradient target there
            dB = secant_dB(ep["A"], ep["B"], C_sec, bank, amask_np) if C_sec is not None \
                else np.zeros((T, 6, ad), np.float32)
            rows = dict(o=ep["obs"][:-1], a=ep["act"], o2=ep["obs"][1:], rc=ep["rc"],
                        gn=ep["gn"].reshape(T, -1), A=ep["A"].reshape(T, -1),
                        B=ep["B"].reshape(T, -1), dB=dB.reshape(T, -1),
                        Cg=ep["Cg"].reshape(T, -1), gg=ep["gg"].reshape(T, -1),
                        done=done[:, None], fell=(done * fell)[:, None], vg=vg[:, None],
                        anchor=np.full((T, 1), anchor, np.float32), mask=amask_np,
                        dom=np.full((T, 1), dom, np.float32),
                        dgr=np.full((T, 1), dgr, np.float32))
            if "tgt" in ep:                       # the ILC's labels (offline trials)
                rows.update(tgt=ep["tgt"], Kx=ep["Kx"].reshape(T, -1),
                            Kg=ep["Kg"].reshape(T, -1),
                            sup=np.full((T, 1), float(ep["sup"]), np.float32),
                            relabel=np.full((T, 1), float(ep["relabel"]), np.float32),
                            mca=ep["mca"].reshape(T, -1), mcs=ep["mcs"].reshape(T, -1),
                            Hu=ep["Hu"].reshape(T, -1))
                mcac = np.array(ep["mca"], np.float32)
                if C_sec is not None and "gcN" in ep:
                    # the landing co-state's action part with the corrected sensitivity:
                    # + gamma^(Nc-1-k) C_k' grad c_N, per unit of normalized action
                    for k in range(T):
                        corr = C_sec[:, 4 * k:4 * k + 4].T @ np.asarray(ep["gcN"], float)
                        mcac[k, 2] += (cfg["gamma"] ** (T - 1 - k)) * corr * \
                            np.asarray(bank["a_scale"]) * amask_np[k]
                rows["mcac"] = mcac.reshape(T, -1)
                if C_cl is not None:
                    # closed-loop co-states with the secant-corrected B (normalized units)
                    Bp = ep["B"] + secant_dB(ep["A"], ep["B"], C_cl, bank, amask_np)
                    An, Kx_ = np.asarray(ep["A"], float), np.asarray(ep["Kx"], float)
                    gn_ = np.asarray(ep["gn"], float)
                    m2a, m2s = np.zeros((T, 3, ad)), np.zeros((T, 3, 6))
                    for s_ in range(3):
                        q_ = np.zeros(6)
                        for k in range(T - 1, -1, -1):
                            qk = gn_[k, s_] + (0.0 if k == T - 1 else cfg["gamma"] * (
                                (An[k + 1] + Bp[k + 1] @ Kx_[k + 1]).T @ q_))
                            m2a[k, s_] = Bp[k].T @ qk
                            m2s[k, s_] = An[k].T @ qk
                            q_ = qk
                    rows.update(mcacl2=m2a.reshape(T, -1), mcscl2=m2s.reshape(T, -1))
                if "mca_cl" in ep:
                    rows.update(mcacl=ep["mca_cl"].reshape(T, -1),
                                mcscl=ep["mcs_cl"].reshape(T, -1))
            if online:                                # co-states (co_fd) need the whole episode
                rows["e3"] = np.repeat(np.asarray(ep["info"], np.float32)[None, :3], T, 0)
                rows["mub"] = np.asarray(ep["mu_b"] if "mu_b" in ep else ep["act"], np.float32) * amask_np
                rows["sup"] = np.zeros((T, 1), np.float32)     # set once its co-states are computed
                if self.i + T <= self.cap:
                    self.ep_starts.append(self.i)
            it = torch.as_tensor((self.i + np.arange(T)) % self.cap, device=dev)
            for k, v in rows.items():
                self.d[k][it] = torch.as_tensor(np.asarray(v, np.float32), device=dev)
            self.i = (self.i + T) % self.cap
            self.n = min(self.n + T, self.cap)
            if online:
                self.n_online += T

        def sample(self, B, pool=None):
            if pool is None and bool((TFm > 0).any()) and len(dom_idx):
                # each member's share (tfrac / --target-frac) from the new robot's transitions
                it = torch.where(torch.rand(M, B, device=dev) < TFm[:, None],
                                 dom_idx[torch.randint(len(dom_idx), (M, B), device=dev)],
                                 torch.randint(self.n, (M, B), device=dev))
            elif pool is None:
                it = torch.randint(self.n, (M, B), device=dev)
            else:
                it = pool[torch.randint(len(pool), (M, B), device=dev)]
            return {k: v[it] for k, v in self.d.items()}   # (M, B, w)

    TFm = torch.where(V["tfrac"] >= 0, V["tfrac"], torch.full_like(V["tfrac"], cfg["target_frac"]))

    def label_w(dom, dgr):
        """nomw: the training robot's labels down-weighted near the new robot's goals."""
        near = torch.exp(-(dgr / cfg["nomw_sigma"]) ** 2)
        w_nom = 1.0 - (1.0 - V["nomw"].view(M, *([1] * (dgr.dim() - 1)))) * near
        return torch.where(dom > 0.5, torch.ones_like(w_nom), w_nom)
    buf = Replay(E_off * Nc + cfg["online_episodes"] * Nc + 1000)
    lab_keys = ("tgt", "Kx", "Kg", "sup", "relabel", "mca", "mcs", "Hu", "gcN", "mca_cl",
                "mcs_cl")
    has_labels = all(k in off for k in lab_keys)
    # each episode's goal distance to the nearest goal the new robot's data labels (dgr)
    G_new = off["goal"][off["dom"] > 0.5] if (off["dom"] > 0.5).any() else np.zeros((0, 2))
    dgr_e = np.array([np.min(np.linalg.norm(G_new - g, axis=1)) if len(G_new) else 1e3
                      for g in off["goal"]], np.float32)
    for e in range(E_off if cfg["offline_data"] else 0):
        buf.add({k: off[k][e] for k in ("obs", "act", "rc", "gn", "A", "B", "Cg", "gg", "fell")
                 + (lab_keys if has_labels else ())},
                anchor=float(off["anchor"][e]), dom=float(off["dom"][e]), dgr=float(dgr_e[e]))
    dom_idx = torch.nonzero(buf.d["dom"][:buf.n, 0] > 0.5)[:, 0]
    # whole trials, for the outcome-aware value term (it couples a trial's samples via G_N)
    if has_labels and cfg["offline_data"]:
        ep_ok = np.flatnonzero(off["sup"] > 0.5)
        EP = dict(o=torch.as_tensor(off["obs"][ep_ok, :-1], device=dev),
                  tgt=torch.as_tensor(off["tgt"][ep_ok], device=dev),
                  GN=torch.as_tensor(off["GN"][ep_ok], device=dev),
                  rel=torch.as_tensor(off["relabel"][ep_ok], device=dev),
                  dom=torch.as_tensor(off["dom"][ep_ok], device=dev),
                  dgr=torch.as_tensor(dgr_e[ep_ok], device=dev))
        ep_dom = torch.nonzero(EP["dom"] > 0.5)[:, 0]
    else:
        EP = None
    amask_t = torch.as_tensor(amask_np, device=dev)
    anchors = torch.nonzero(buf.d["anchor"][:buf.n, 0] > 0.5)[:, 0]
    n_off = buf.n
    off_idx = torch.arange(n_off, device=dev)
    log(f"{M} members {[n for n, _ in variants]}; offline {E_off} ILC trials, {buf.n} "
        f"transitions ({len(anchors)} from the runs' best trials)")

    # batched networks ---------------------------------------------------------------------------
    class BLinear(nn.Module):
        def __init__(self, E, i, o):
            super().__init__()
            k = 1.0 / math.sqrt(i)
            self.W = nn.Parameter(torch.empty(E, i, o).uniform_(-k, k))
            self.b = nn.Parameter(torch.empty(E, 1, o).uniform_(-k, k))

        def forward(self, x):
            return torch.baddbmm(self.b, x, self.W)

    Ha, Hc = cfg["actor_hidden"], cfg["critic_hidden"]

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1, self.l2 = BLinear(M, od, Ha), BLinear(M, Ha, Ha)
            self.mu, self.ls = BLinear(M, Ha, ad), BLinear(M, Ha, ad)

        def forward(self, o):                       # o (M, B, od)
            h = F.relu(self.l2(F.relu(self.l1(o * omask[:, None]))))
            return self.mu(h), self.ls(h).clamp(-5.0, 1.0)

        def sample(self, o):
            mu, ls = self(o)
            eps = torch.randn_like(mu)
            pre = mu + ls.exp() * eps
            logp = (-0.5 * eps ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1) \
                - (2 * (math.log(2) - pre - F.softplus(-2 * pre))).sum(-1)
            return torch.tanh(pre), logp, torch.tanh(mu)

    class Critic(nn.Module):
        """2M critics, member m's twins at 2m, 2m+1: (M, 2, B, in) -> (M, 2, B)."""

        def __init__(self):
            super().__init__()
            self.ls = nn.ModuleList([BLinear(2 * M, od + ad, Hc), BLinear(2 * M, Hc, Hc),
                                     BLinear(2 * M, Hc, 1)])
            self.m2 = omask.repeat_interleave(2, 0)[:, None]

        def forward(self, o, a_):
            if o.dim() == 3:
                o = o.unsqueeze(1).expand(M, 2, *o.shape[1:])
                a_ = a_.unsqueeze(1).expand(M, 2, *a_.shape[1:])
            B = o.shape[2]
            x = torch.cat([o.reshape(2 * M, B, od) * self.m2, a_.reshape(2 * M, B, ad)], -1)
            x = torch.tanh(self.ls[0](x))
            x = torch.tanh(self.ls[1](x))
            return self.ls[2](x).view(M, 2, B)

    actor, qnet, qtgt = Actor().to(dev), Critic().to(dev), Critic().to(dev)
    if cfg["init_from"]:                          # fine-tuning: start where a run ended
        ck = torch.load(os.path.join(cfg["init_from"], "train", f"{gname}_final.pt"),
                        map_location=dev, weights_only=False)
        if cfg["init_variant"]:
            # every member from the checkpoint's members of one variant (seed j <- its j-th
            # seed, cycling): matched starting points for an ablation
            src = [i for i, (n, _) in enumerate(ck["variants"])
                   if n.split(".")[0] == cfg["init_variant"]]
            if not src:
                raise SystemExit(f"--init-variant: no {cfg['init_variant']} in {gname}")
            seeds = {}
            idx = []
            for n, _ in variants:
                v = n.split(".")[0]
                idx.append(src[seeds.get(v, 0) % len(src)])
                seeds[v] = seeds.get(v, 0) + 1
            ia = torch.tensor(idx, device=dev)
            ic = torch.stack([2 * ia, 2 * ia + 1], 1).reshape(-1)
            actor.load_state_dict({k: v[ia] for k, v in ck["actor"].items()})
            qnet.load_state_dict({k: (v[ic] if v.shape[0] == 2 * len(ck["variants"]) else v)
                                  for k, v in ck["critic"].items()})
            log(f"initialized every member from {cfg['init_from']}'s {cfg['init_variant']}")
        else:
            if len(ck["variants"]) != len(variants) or (
                    not cfg["init_loose"] and [n for n, _ in ck["variants"]] != [n for n, _ in variants]):
                raise SystemExit(f"--init-from: {gname}'s members differ")
            actor.load_state_dict(ck["actor"])
            qnet.load_state_dict(ck["critic"])
            log(f"initialized from {cfg['init_from']}")
    if cfg.get("small_init") and not cfg["init_from"]:
        # a fresh actor that starts as the plan itself: the mean head near zero (u = u_ref), small exploration
        with torch.no_grad():
            actor.mu.W.mul_(0.01)
            actor.mu.b.zero_()
            actor.ls.W.mul_(0.01)
            actor.ls.b.fill_(-3.0)
        log("actor: fresh, small init (starts as the open-loop plan)")
    if cfg["fresh_critic"]:                       # the critic from scratch (the actor as initialized)
        qnet = Critic().to(dev)
        log("critic: fresh (not initialized from --init-from)")
    qtgt.load_state_dict(qnet.state_dict())
    for p in qtgt.parameters():
        p.requires_grad_(False)
    qparams, tparams = list(qnet.parameters()), list(qtgt.parameters())
    opt_q = torch.optim.Adam(qparams, lr=cfg["lr"])
    opt_pi = torch.optim.Adam(actor.parameters(), lr=cfg["actor_lr"] or cfg["lr"])
    log_alpha = torch.full((M,), math.log(cfg["alpha"]), device=dev, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=cfg["lr"])

    gamma, rw, fall_pen = cfg["gamma"], cfg["reward_scale"], cfg["fall_penalty"]
    bc_w = V["bc_scale"] * bank["r_scale"] * rw * cfg["qu"]      # Qu: the trust region
    f1, f2 = cfg["stage_schedule"]
    st = dict(n=0, h_s=None, h_a=None, h_g=None, h_c=None, beta_c=torch.zeros(M, device=dev),
              beta_s=torch.ones(M, device=dev), beta_a=torch.ones(M, device=dev),
              beta_g=torch.ones(M, device=dev), b_v=torch.zeros(M, device=dev),
              b_x=torch.zeros(M, device=dev), b_g=torch.zeros(M, device=dev),
              b_a=torch.zeros(M, device=dev), b_w=torch.zeros(M, device=dev))
    on_s, on_a, on_g = (V["zeta_s"] > 0).float(), (V["zeta_a"] > 0).float(), \
        (V["zeta_g"] > 0).float()

    def stage_w():
        p = st["n"] / cfg["pretrain_updates"] if cfg["pretrain_updates"] else st.get("prog", 0.0)
        s = 0 if p < f1 else 1 if p < f2 else 2
        c = np.full(3, cfg["stage_floor"], np.float32)
        c[s] = 1.0
        return torch.tensor(c, device=dev)

    def hinge(res2, key):
        """Their hinge: past the member's running median of the loss, a residual counts
        linearly. res2 (M, 2, B)."""
        med = res2.detach().flatten(1).median(1).values
        st[key] = med if st[key] is None else 0.99 * st[key] + 0.01 * med
        h = st[key].clamp_min(1e-8)[:, None, None]
        return torch.where(res2 <= h, res2, 2.0 * torch.sqrt(res2 * h + 1e-12) - h)

    def recompute_costates():
        """co_fd: every online episode's closed-loop co-states under the CURRENT actor -- the ILC's per-trial
        co-states, stage by stage: q_k = grad c_k(e_k+1) + gamma (A_k+1 + B_k+1 K_k+1)' q_k+1 (the last
        transition ends the episode), K = d pi / d e at the episode's own states -- into the replay as each
        member's targets B_k' q_k (cla) and A_k' q_k (cls), normalized units, as the zeta_c loss reads them."""
        if not buf.ep_starts or not bool((V["co_fd"] > 0).any()):
            return
        st0 = torch.as_tensor(buf.ep_starts, device=dev)
        E = len(st0)
        idx = st0[:, None] + torch.arange(Nc, device=dev)                  # (E, Nc)
        A_ = buf.d["A"][idx].view(E, Nc, 6, 6)
        B_ = buf.d["B"][idx].view(E, Nc, 6, ad)
        gn_ = buf.d["gn"][idx].view(E, Nc, 3, 6)
        o_ = buf.d["o"][idx].reshape(1, E * Nc, od).expand(M, -1, -1).clone().requires_grad_(True)
        a_ = torch.tanh(actor(o_)[0])                                      # (M, E Nc, ad)
        K = torch.stack([torch.autograd.grad(a_[..., j].sum(), o_, retain_graph=j < ad - 1)[0][..., :6]
                         for j in range(ad)], -2)                          # (M, E Nc, ad, 6)
        K = (K.view(M, E, Nc, ad, 6) * amask_t[None, None, :, :, None]).detach()
        with torch.no_grad():
            Acl = A_[None] + B_[None] @ K                                  # (M, E, Nc, 6, 6)
            q = gn_[None, :, Nc - 1].expand(M, -1, -1, -1).clone()         # (M, E, 3 stages, 6), rows
            cla = torch.zeros(M, E, Nc, 3, ad, device=dev)
            cls_ = torch.zeros(M, E, Nc, 3, 6, device=dev)
            for k in range(Nc - 1, -1, -1):
                if k < Nc - 1:
                    q = gn_[None, :, k] + gamma * q @ Acl[:, :, k + 1]
                cla[:, :, k] = q @ B_[None, :, k]
                cls_[:, :, k] = q @ A_[None, :, k]
            # robust: a co-state past co_clip x its median (per member, stage) is scaled down to it
            for X_ in (cla, cls_):
                nrm = X_.norm(dim=-1)                                      # (M, E, Nc, 3)
                med = nrm.flatten(1, 2).median(1).values                   # (M, 3)
                cap = V["co_clip"].view(M, 1, 1, 1) * med[:, None, None, :]
                X_ *= (cap / nrm.clamp_min(1e-12)).clamp(max=1.0)[..., None]
            flat = idx.reshape(-1)
            buf.d["cla"][flat] = cla.permute(1, 2, 0, 3, 4).reshape(E * Nc, M * 3 * ad)
            buf.d["cls"][flat] = cls_.permute(1, 2, 0, 3, 4).reshape(E * Nc, M * 18)
            ok_ = 1.0 - buf.d["fell"][idx[:, -1], 0]                    # a fall: no co-state targets
            buf.d["sup"][flat] = ok_[:, None].expand(E, Nc).reshape(-1, 1)
            st["recent"] = idx[-cfg["co_recent"]:].reshape(-1)              # the actor's pool (co_pg)
            # co_ilc: each episode's ILC step, on its discounted stage cost under the current stage weights
            c_ = stage_w()
            disc = gamma ** torch.arange(Nc, device=dev, dtype=torch.float32)
            J0 = rw * torch.einsum("enS,S,n->e", buf.d["rc"][idx].view(E, Nc, 3), c_, disc)        # (E,)
            g0 = rw * torch.einsum("menSa,S,n->mena", cla, c_, disc) * amask_t[None, None]       # d J0 / d a
            a_fl = buf.d["mub"][idx].view(1, E, Nc, ad)      # the behaviour policy's mean (not its noise)
            step = (V["co_beta"].view(M, 1, 1, 1) * J0.view(1, E, 1, 1) * g0
                    / (g0 ** 2).sum((2, 3), keepdim=True).clamp_min(1e-12))
            rms = torch.sqrt((step ** 2).sum((2, 3), keepdim=True) / amask_t.sum())
            step = step * (V["co_max"].view(M, 1, 1, 1) / rms.clamp_min(1e-12)).clamp(max=1.0)
            if bool((V["co_gn"] > 0).any()) and float(c_[2]) >= 1.0:
                # Stage III: Gauss-Newton on the measured landing error through the closed-loop landing
                # sensitivity; the flight is ballistic: d x_N / d x_Nc = [[I, T I], [0, I]], T the flight time
                Tf = (bank["phases"][2]) * float(bank["dt"])
                Phi = torch.eye(6, device=dev)
                Phi[:3, 3:] = Tf * torch.eye(3, device=dev)
                Dt = torch.as_tensor(bank["sx"], dtype=torch.float32, device=dev)
                qS = (Phi[:3] * Dt[None]).expand(M, E, 3, 6).clone()       # rows: d x_N,i / d e_Nc
                S = torch.zeros(M, E, 3, Nc, ad, device=dev)
                for k in range(Nc - 1, -1, -1):
                    if k < Nc - 1:
                        qS = qS @ Acl[:, :, k + 1]
                    S[:, :, :, k] = qS @ B_[None, :, k]
                Sf = (S * amask_t[None, None, None]).reshape(M, E, 3, Nc * ad)
                e3 = buf.d["e3"][idx[:, 0]]                                 # (E, 3)
                Mm = Sf @ Sf.transpose(-1, -2)
                Mm = Mm + V["co_delta"].view(M, 1, 1, 1) * Mm.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] / 3 \
                    * torch.eye(3, device=dev)
                xg = torch.linalg.solve(Mm, e3[None].expand(M, -1, -1).unsqueeze(-1))
                stp = (Sf.transpose(-1, -2) @ xg).squeeze(-1).view(M, E, Nc, ad) * V["co_beta"].view(M, 1, 1, 1)
                rms_g = torch.sqrt((stp ** 2).sum((2, 3), keepdim=True) / amask_t.sum())
                stp = stp * (V["co_max"].view(M, 1, 1, 1) / rms_g.clamp_min(1e-12)).clamp(max=1.0)
                step = torch.where(V["co_gn"].view(M, 1, 1, 1) > 0, stp, step)
            cot = ((a_fl - step) * amask_t[None, None]).clamp(-0.999, 0.999)
            buf.d["cot"][flat] = cot.permute(1, 2, 0, 3).reshape(E * Nc, M * ad)

    def member_norms(loss, params):
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.sqrt(sum(x.reshape(M, -1).pow(2).sum(1) for x in g if x is not None))

    def clip_members(params, max_norm=100.0):
        n = torch.sqrt(sum(p.grad.reshape(M, -1).pow(2).sum(1) for p in params))
        s = (max_norm / (n + 1e-6)).clamp(max=1.0)
        for p in params:
            p.grad.mul_(s.view(M, *([1] * (p.grad.dim() - 1))) if p.shape[0] == M
                        else s.repeat_interleave(2).view(2 * M, *([1] * (p.grad.dim() - 1))))

    def update(b):
        c = stage_w()
        o, act, o2, msk = b["o"], b["a"] * b["mask"], b["o2"], b["mask"]
        done, fell, vg = b["done"][..., 0], b["fell"][..., 0], b["vg"][..., 0]
        # members with hind=0 do not see the hindsight-relabeled transitions
        wh = 1.0 - b["relabel"][..., 0] * (1.0 - V["hind"][:, None])        # (M, B)
        # crit_t: the critic's losses on the new robot's transitions only
        wq = wh * torch.where(V["crit_t"][:, None] > 0, b["dom"][..., 0], torch.ones_like(wh))
        vg = vg * wq
        Bb = b["B"].view(M, -1, 6, ad) + V["secant"].view(M, 1, 1, 1) * b["dB"].view(M, -1, 6, ad)
        A, Cg = b["A"].view(M, -1, 6, 6), b["Cg"].view(M, -1, 6, 2)
        gn, gg = b["gn"].view(M, -1, 3, 6), b["gg"].view(M, -1, 3, 2)
        alpha = log_alpha.exp().detach()[:, None]
        r = -rw * (b["rc"] @ c) - fall_pen * fell                          # (M, B)
        g_next = -rw * torch.einsum("s,mbsj->mbj", c, gn)                 # d r / d e'
        dr_g = -rw * torch.einsum("s,mbsj->mbj", c, gg)                  # d r / d g, e' held
        # the TD target, and d/do' of the soft value at o' (their eq. 14's bracket)
        o2g = o2.clone().requires_grad_(True)
        a2, logp2, _ = actor.sample(o2g)
        v2 = qtgt(o2g, a2 * msk).min(1).values - alpha * logp2
        gv = torch.autograd.grad(v2.sum(), o2g)[0]
        if bool((V["vg_open"] > 0).any()):            # a' held: no chain through the actor
            o2h = o2.clone().requires_grad_(True)
            qh = qtgt(o2h, (a2 * msk).detach()).min(1).values
            gv_open = torch.autograd.grad(qh.sum(), o2h)[0]
            gv = torch.where(V["vg_open"].view(M, 1, 1) > 0, gv_open, gv)
        y = (r + gamma * (1 - done) * v2).detach()
        cont = (gamma * (1 - done))[..., None]
        gc = g_next + cont * gv[..., :6]                                  # through e'
        # targets for d Q / d o (error, history, previous action), d a, d goal
        T_o = torch.zeros_like(o)
        T_o[..., :6] = torch.einsum("mbij,mbi->mbj", A, gc)
        if H:
            T_o[..., :6] += cont * gv[..., i_hist:i_hist + 6]            # o'_hist1 = e
            for j in range(1, H):
                T_o[..., i_hist + 6 * (j - 1):i_hist + 6 * j] = \
                    cont * gv[..., i_hist + 6 * j:i_hist + 6 * (j + 1)]
        T_a = torch.einsum("mbij,mbi->mbj", Bb, gc)
        if PA:
            T_a = T_a + cont * gv[..., i_pa:i_pa + ad]                     # o'_prev = a
        T_g = torch.einsum("mbij,mbi->mbj", Cg, gc) + dr_g + cont * gv[..., 7:9]
        T_o, T_a, T_g = T_o.detach(), T_a.detach(), T_g.detach()
        dyn = torch.ones(od, device=dev)
        dyn[6:9] = 0.0                                                    # time, goal: not state
        og = o.unsqueeze(1).expand(M, 2, *o.shape[1:]).clone().requires_grad_(True)
        ag = act.unsqueeze(1).expand(M, 2, *act.shape[1:]).clone().requires_grad_(True)
        qv = qnet(og, ag)
        lq = (((qv - y[:, None]) ** 2) * wq[:, None]).sum((1, 2)) \
            / (2 * wq.sum(1)).clamp_min(1.0)                               # (M,)
        gs, ga = torch.autograd.grad(qv.sum(), [og, ag], create_graph=True)
        w = (vg / vg.sum(1, keepdim=True).clamp_min(1.0))[:, None]        # (M, 1, B)
        ls_ = (hinge((((gs - T_o[:, None]) * dyn) ** 2).sum(-1), "h_s") * w).sum((1, 2))
        la_ = (hinge((((ga - T_a[:, None]) * msk[:, None]) ** 2).sum(-1), "h_a") * w).sum((1, 2))
        lg_ = (hinge((((gs[..., 7:9] - T_g[:, None]) @ Pg) ** 2).sum(-1), "h_g") * w).sum((1, 2))
        # the trials' co-states: exact value gradients along the ILC's own data
        lc_ = torch.zeros(M, device=dev)
        if bool((V["zeta_c"] > 0).any()):
            wc = (b["sup"][..., 0] * wq)[:, None]                          # (M, 1, B)
            mca_m = torch.where(V["sec_c"].view(M, 1, 1) > 0, b["mcac"], b["mca"])
            mca_m = torch.where(V["cl_co"].view(M, 1, 1) > 0, b["mcacl"], mca_m)
            mcs_m = torch.where(V["cl_co"].view(M, 1, 1) > 0, b["mcscl"], b["mcs"])
            mca_m = torch.where(V["sec_cl"].view(M, 1, 1) > 0, b["mcacl2"], mca_m)
            mcs_m = torch.where(V["sec_cl"].view(M, 1, 1) > 0, b["mcscl2"], mcs_m)
            mi_ = torch.arange(M, device=dev)
            cla_m = b["cla"].view(M, -1, M, 3 * ad)[mi_, :, mi_]          # member m's own slot
            cls_m = b["cls"].view(M, -1, M, 18)[mi_, :, mi_]
            mca_m = torch.where(V["co_fd"].view(M, 1, 1) > 0, cla_m, mca_m)
            mcs_m = torch.where(V["co_fd"].view(M, 1, 1) > 0, cls_m, mcs_m)
            Ta_c = -rw * torch.einsum("s,mbsj->mbj", c, mca_m.view(M, -1, 3, ad))
            Ts_c = -rw * torch.einsum("s,mbsj->mbj", c, mcs_m.view(M, -1, 3, 6))
            res = (((ga - Ta_c[:, None]) * msk[:, None]) ** 2).sum(-1) \
                + ((gs[..., :6] - Ts_c[:, None]) ** 2).sum(-1)
            lc_ = (hinge(res, "h_c") * wc).sum((1, 2)) / (2 * wc.sum((1, 2))).clamp_min(1.0)
        if st["n"] % 20 == 0:                                             # their eq. 18
            nq = member_norms(lq.sum(), qparams)
            if lc_.requires_grad:                     # (no member with zeta_c: no co-state loss)
                st["beta_c"] = V["zeta_c"] * nq / member_norms(lc_.sum(), qparams).clamp_min(1e-8)
            st["beta_s"] = V["zeta_s"] * nq / member_norms(ls_.sum(), qparams).clamp_min(1e-8)
            st["beta_a"] = V["zeta_a"] * nq / member_norms(la_.sum(), qparams).clamp_min(1e-8)
            st["beta_g"] = V["zeta_g"] * nq / member_norms(lg_.sum(), qparams).clamp_min(1e-8)
        loss_q = (lq + on_s * st["beta_s"] * ls_ + on_a * st["beta_a"] * la_
                  + on_g * st["beta_g"] * lg_ + st["beta_c"] * lc_).sum()
        opt_q.zero_grad(set_to_none=True)
        loss_q.backward()
        clip_members(qparams)
        opt_q.step()
        # actor: SAC, plus the Qu-weighted pull toward the ILC's forces on its best trials
        a_pi, logp, _ = actor.sample(o)
        q_pi = qnet(o, a_pi * msk).min(1).values
        lbc = torch.zeros(M, device=dev)
        if len(anchors):
            ba = buf.sample(cfg["batch"] // 4, anchors)
            _, _, mu_a = actor.sample(ba["o"])
            lw_a = label_w(ba["dom"][..., 0], ba["dgr"][..., 0])
            lbc = ((((mu_a - ba["a"]) * asc * ba["mask"]) ** 2).sum(-1) * lw_a).mean(1)
        if n_off and bool((V["bc_all"] > 0).any()):    # every ILC trial's forces at its states
            bo = buf.sample(cfg["batch"] // 4, off_idx)
            _, _, mu_o = actor.sample(bo["o"])
            l_all = (((mu_o - bo["a"]) * asc * bo["mask"]) ** 2).sum(-1).mean(1)
            lbc = torch.where(V["bc_all"] > 0, l_all, lbc)
        l_sac = ((alpha * logp - q_pi) * wh).sum(1) / wh.sum(1).clamp_min(1.0)
        rl_on = st["n"] >= max(cfg["rl_start"] * cfg["pretrain_updates"], cfg.get("critic_warmup", 0))
        if rl_on and st.get("actor_ref") is None:     # the policy RL starts from, frozen
            import copy
            st["actor_ref"] = copy.deepcopy(actor)
            for p_ in st["actor_ref"].parameters():
                p_.requires_grad_(False)
        rl_eff = V["rl"] * float(rl_on)
        if bool((V["rl_h"] > 0).any()):
            # against the trial's QP curvature: 1/2 (a - c)' H (a - c), c the flown forces or
            # (anchor_ref) the policy RL started from
            centre = b["a"]
            ar = V["anchor_ref"].view(M, 1, 1)
            if st.get("actor_ref") is not None and bool((V["anchor_ref"] == 1).any()):
                with torch.no_grad():
                    a_ref = torch.tanh(st["actor_ref"](o)[0])
                centre = torch.where(ar == 1, a_ref, centre)
            # 2: the labels (the ILC step; RL adds the critic's correction on top of it)
            centre = torch.where(ar == 2, b["tgt"], centre)
            if bool((V["anchor_ref"] == 3).any()):    # 3: a slow average of the actor itself
                if st.get("actor_ema") is None:
                    import copy
                    st["actor_ema"] = copy.deepcopy(actor)
                    for p_ in st["actor_ema"].parameters():
                        p_.requires_grad_(False)
                with torch.no_grad():
                    a_ema = torch.tanh(st["actor_ema"](o)[0])
                centre = torch.where(ar == 3, a_ema, centre)
            da = (a_pi - centre) * msk
            Hq = 0.5 * rw * torch.einsum("mbi,mbij,mbj->mb", da, b["Hu"].view(M, -1, ad, ad), da)
            l_h = (Hq * wh * b["sup"][..., 0]).sum(1) / wh.sum(1).clamp_min(1.0)
            l_sac = l_sac + V["rl_h"] * l_h
        # the Sobolev terms: each trial's own A, B (its ILC problem) shape the network
        l_v = l_x = l_g = l_a = l_w = torch.zeros(M, device=dev)
        if EP is not None and bool((V["sob_v"] + V["sob_x"] + V["sob_g"] + V["sob_aug"] > 0).any()):
            ws = b["sup"][..., 0] * wh * label_w(b["dom"][..., 0], b["dgr"][..., 0])   # labeled
            og2 = o.clone().requires_grad_(True)
            mu = torch.tanh(actor(og2)[0])                               # (M, B, 4)
            J = torch.stack([torch.autograd.grad(mu[..., j].sum(), og2, create_graph=True)[0]
                             for j in range(ad)], -2)                     # (M, B, 4, od)
            nrm = ws.sum(1).clamp_min(1.0)
            Kx, Kg = b["Kx"].view(M, -1, ad, 6), b["Kg"].view(M, -1, ad, 2)
            l_x = ((((J[..., :6] - Kx) * msk[..., None]) ** 2).sum((-1, -2)) * ws).sum(1) / nrm
            l_g = (((((J[..., 7:9] - Kg) @ Pg) * msk[..., None]) ** 2).sum((-1, -2)) * ws).sum(1) / nrm
            # value, outcome-aware: the landing change G_N (u_pi - u*) in Qe, plus Qu |.|^2
            Be = cfg["sob_episodes"]
            ie = torch.randint(EP["o"].shape[0], (M, Be), device=dev)
            if bool((TFm > 0).any()) and len(ep_dom):
                ie = torch.where(torch.rand(M, Be, device=dev) < TFm[:, None],
                                 ep_dom[torch.randint(len(ep_dom), (M, Be), device=dev)], ie)
            we = (1.0 - EP["rel"][ie] * (1.0 - V["hind"][:, None])) \
                * label_w(EP["dom"][ie], EP["dgr"][ie])                    # (M, Be)
            mu_e = torch.tanh(actor(EP["o"][ie].reshape(M, Be * Nc, od))[0]).view(M, Be, Nc, ad)
            dlt = (mu_e - EP["tgt"][ie]) * amask_t
            e_land = torch.einsum("mbqkj,mbkj->mbq", EP["GN"][ie], dlt)    # (M, Be, 6)
            qe_t = torch.tensor(bank["qe"], device=dev)
            cost = (e_land ** 2 * qe_t).sum(-1) + bank.get("qu3", 1e-5) * ((dlt * asc) ** 2).sum((-1, -2))
            l_v = bank["r_scale"] * rw * (cost * we).sum(1) / we.sum(1).clamp_min(1.0)
            if bool((V["sob_aug"] > 0).any()):
                # the label carried to perturbed states by the trial's own gain
                dlt_o = torch.randn(M, o.shape[1], 6, device=dev) * cfg["aug_sigma"]
                o_p = o.clone()
                o_p[..., :6] = o_p[..., :6] + dlt_o
                mu_p = torch.tanh(actor(o_p)[0])
                tgt_p = b["tgt"] + torch.einsum("mbij,mbj->mbi", Kx, dlt_o)
                l_a = ((((mu_p - tgt_p) * msk) ** 2).sum(-1) * ws).sum(1) / nrm
            l_w = torch.zeros(M, device=dev)
            if bool((V["awr"] > 0).any()):
                # advantage-weighted: toward the label and the flown forces, each weighted by
                # how much the critic prefers it to the policy's own action
                with torch.no_grad():
                    v_pi = qnet(o, mu.detach() * msk).min(1).values
                    q_t = qnet(o, b["tgt"] * msk).min(1).values
                    q_d = qnet(o, b["a"] * msk).min(1).values
                    bta = cfg["awr_beta"]
                    w_t = torch.exp(((q_t - v_pi) / bta).clamp(-5, 3))
                    w_d = torch.exp(((q_d - v_pi) / bta).clamp(-5, 3))
                mu_w = torch.tanh(actor(o)[0])
                l_w = ((w_t * (((mu_w - b["tgt"]) * msk) ** 2).sum(-1)
                        + w_d * (((mu_w - b["a"]) * msk) ** 2).sum(-1)) / (w_t + w_d) * ws
                       ).sum(1) / nrm
            if st["n"] % 20 == 0:                                         # eq. 18 style
                ap = list(actor.parameters())
                ns = member_norms(l_sac.sum(), ap)
                ref = torch.where(rl_eff > 0, ns, torch.ones_like(ns))
                for key, l_, z in (("b_v", l_v, V["sob_v"]), ("b_x", l_x, V["sob_x"]),
                                   ("b_g", l_g, V["sob_g"]), ("b_a", l_a, V["sob_aug"]),
                                   ("b_w", l_w, V["awr"])):
                    if l_.requires_grad:
                        st[key] = z * ref / member_norms(l_.sum(), ap).clamp_min(1e-8)
        # rl=0 members: no SAC term -- the supervised network (Sobolev) or behaviour cloning
        l_co = torch.zeros(M, device=dev)
        if bool((V["co_pg"] > 0).any()) and st.get("recent") is not None:
            br = buf.sample(cfg["batch"], st["recent"])
            mu_r = torch.tanh(actor(br["o"])[0])
            mi_ = torch.arange(M, device=dev)
            cla_r = br["cla"].view(M, -1, M, 3, ad)[mi_, :, mi_]          # (M, B, 3, ad)
            g_r = (rw * torch.einsum("s,mbsa->mba", c, cla_r) * br["mask"]).detach()   # d J / d a
            w_r = br["sup"][..., 0]
            l_co = ((g_r * mu_r).sum(-1) * w_r).sum(1) / w_r.sum(1).clamp_min(1.0)
        l_ilc = torch.zeros(M, device=dev)
        if bool((V["co_ilc"] > 0).any()) and st.get("recent") is not None:
            br = buf.sample(cfg["batch"], st["recent"])
            mu_r = torch.tanh(actor(br["o"])[0])
            mi_ = torch.arange(M, device=dev)
            cot_r = br["cot"].view(M, -1, M, ad)[mi_, :, mi_]
            w_r = br["sup"][..., 0]
            l_ilc = ((((mu_r - cot_r) * br["mask"]) ** 2).sum(-1) * w_r).sum(1) / w_r.sum(1).clamp_min(1.0)
        loss_pi = (V["co_ilc"] * l_ilc + V["co_pg"] * l_co + rl_eff * l_sac
                   + torch.where(V["rl"] > 0, bc_w, (V["bc_scale"] > 0).float()) * lbc
                   + st["b_v"] * l_v + st["b_x"] * l_x + st["b_g"] * l_g
                   + st["b_a"] * l_a + st["b_w"] * l_w).sum()
        opt_pi.zero_grad(set_to_none=True)
        loss_pi.backward()
        opt_pi.step()
        if st.get("actor_ema") is not None:
            with torch.no_grad():
                ep_, ap_ = list(st["actor_ema"].parameters()), list(actor.parameters())
                torch._foreach_mul_(ep_, 1 - cfg["tau_actor"])
                torch._foreach_add_(ep_, ap_, alpha=cfg["tau_actor"])
        la = -(log_alpha[:, None] * (logp.detach() + cfg["target_entropy"])).mean(1).sum()
        opt_alpha.zero_grad(set_to_none=True)
        la.backward()
        opt_alpha.step()
        with torch.no_grad():
            torch._foreach_mul_(tparams, 1 - cfg["tau"])
            torch._foreach_add_(tparams, qparams, alpha=cfg["tau"])
        st["n"] += 1
        return torch.stack([lq.detach(), ls_.detach(), la_.detach(), lg_.detach(), lbc.detach(),
                            q_pi.mean(1).detach(), alpha[:, 0], l_v.detach(), l_x.detach(),
                            l_g.detach(), lc_.detach()])                  # (11, M)

    STAT = ("lq", "ls", "la", "lg", "bc", "q", "alpha", "Lv", "Lx", "Lg", "lc")

    def fmt(stats):
        v = torch.stack(stats).mean(0) if isinstance(stats, list) else stats
        v = v.cpu().numpy()
        by = {}
        for m, n in enumerate(names):                    # mean over a variant's seeds
            by.setdefault(n.split("_", 1)[1].split(".")[0], []).append(v[:, m])
        return " | ".join(f"{n}: " + " ".join(f"{k}={x:.3g}" for k, x in zip(STAT, np.mean(c, 0)))
                          for n, c in by.items())

    def export(m):
        """Member m's actor as dilc_execute.NumpyActor's npz (inputs it never saw zeroed)."""
        g = lambda layer: (layer.W[m].detach().T.cpu().numpy().astype(np.float32),
                           layer.b[m, 0].detach().cpu().numpy().astype(np.float32))
        W0, b0 = g(actor.l1)
        W0 = W0 * omask[m].cpu().numpy()[None, :]
        W1, b1 = g(actor.l2)
        Wm, bm = g(actor.mu)
        Wl, bl = g(actor.ls)
        return dict(W0=W0, b0=b0, W1=W1, b1=b1, W_mu=Wm, b_mu=bm, W_ls=Wl, b_ls=bl,
                    n_hidden=np.int64(2))

    def to_bytes(w):
        bio = io.BytesIO()
        np.savez(bio, **w)
        return bio.getvalue()

    # pretraining on the ILC's own trials, through its stage schedule ------------------------------
    t_pre = time.time()
    tags = []                                     # (tag, episodes, updates) of every snapshot
    pre_snaps = {int(f * cfg["pretrain_updates"]) for f in cfg["pretrain_snapshots"]}
    for u in range(cfg["pretrain_updates"]):
        s = update(buf.sample(cfg["batch"]))
        if u == 0 or u == cfg["pretrain_updates"] - 1 or (u + 1) % 5000 == 0:
            log(f"pretrain {u}: " + fmt(s))
        if u + 1 in pre_snaps and u + 1 < cfg["pretrain_updates"]:
            for m in range(M):
                np.savez(os.path.join(pdir, f"{names[m]}_u{u + 1}.npz"), **export(m))
            tags.append((f"u{u + 1}", 0, u + 1))
    if cfg["pretrain_updates"]:
        torch.cuda.synchronize()
        log(f"pretrain: {cfg['pretrain_updates']} batched updates, "
            f"{(time.time() - t_pre) / cfg['pretrain_updates'] * 1e3:.1f} ms each")
    if cfg["online_episodes"] == 0:
        # offline only: the controller is synthesized from the ILC's trials alone
        for m in range(M):
            np.savez(os.path.join(pdir, f"{names[m]}_u{st['n']}.npz"), **export(m))
        tags.append((f"u{st['n']}", 0, st["n"]))
        torch.save(dict(actor=actor.state_dict(), critic=qnet.state_dict(), variants=variants,
                        log_alpha=log_alpha.detach().cpu()),
                   os.path.join(tdir, f"{gname}_final.pt"))
        _write_members(tdir, gname, gi, names, variants, tags, st["n"], 0, T0)
        log(f"done (offline): {st['n']} batched updates ({M} members), {time.time() - T0:.0f} s")
        return

    # online: this group's workers fly jumps while it learns ---------------------------------------
    box = Container(a.ws)
    bank_path = box.path(os.path.join(a.out, "bank.json"))
    W = cfg["workers"]
    if cfg["gpu_rollouts"] is not None:          # the GPU sim (ilc_mjx): nominal, synchronous, batched
        gpus_r = cfg["gpu_rollouts"] or [gpu]
        workers = [GpuWorker(os.path.join(a.out, "bank.json"), os.path.join(tdir, f"{gname}_gpu{j}.log"),
                             gpus_r[(gi + j) % len(gpus_r)], cfg["gpu_batch"])
                   for j in range(cfg["gpu_workers"])]
    else:
        workers = [Worker(box, bank_path, os.path.join(tdir, f"{gname}_w{j}.log"),
                          domain=100 + (gi * W + j) % 100) for j in range(W)]
    sel = selectors.DefaultSelector()
    for w in workers:
        sel.register(w.p.stdout, selectors.EVENT_READ, w)
    G = np.array([p["goal"] for p in bank["plans"]], float)
    Dg = G - G[0]
    collinear = np.linalg.matrix_rank(Dg, tol=1e-6) <= 1

    def sample_goal():
        if cfg["rollout_goals"]:               # only these goals on the robot (few real goals)
            return np.array([cfg["rollout_goals"][rng.integers(len(cfg["rollout_goals"]))], 0.0])
        if rng.random() < cfg["p_bank_goal"]:
            return G[rng.integers(len(G))]
        if collinear:
            sproj = Dg @ Dg[np.argmax(np.linalg.norm(Dg, axis=1))]
            lo, hi = G[np.argmin(sproj)], G[np.argmax(sproj)]
            return lo + rng.random() * (hi - lo)
        return rng.dirichlet(np.ones(len(G))) @ G

    curve = open(os.path.join(tdir, f"{gname}_curve.csv"), "w", newline="")
    cw = csv.writer(curve)
    cw.writerow(["wall", "episodes", "updates", "member", "goal_dx", "goal_dz", "cond", "ex_cm",
                 "ez_cm", "eth_deg", "fell", "rear_ms", "stage3_cost"])
    cache = {}
    snaps = sorted(set(int(f * cfg["online_episodes"]) for f in cfg["snapshots"]))
    n_sent = n_done = job = 0
    stats, fd_ok = [], []
    co = dict(dirty=False, last=0)
    t_on = time.time()
    try:
        while True:
            for w in workers:
                owed = cfg["pretrain_updates"] + cfg["utd"] * buf.n_online - st["n"]
                if w.job is not None or n_sent >= cfg["online_episodes"] \
                        or owed >= cfg["max_backlog"]:
                    continue
                m = n_sent % M
                key = (m, st["n"] // 25)
                if key not in cache:
                    cache = {k: v for k, v in cache.items() if k[1] == key[1]}
                    cache[key] = to_bytes(export(m))
                n_b = min(w.batch, cfg["online_episodes"] - n_sent)
                jobs = []
                for _ in range(n_b):
                    job += 1
                    jobs.append(dict(id=job, goal=[float(v) for v in sample_goal()],
                                     seed=int(seed * 100003 + job)))
                fd_hdr = dict(eps_a=cfg["fd_jac"][0], eps_s=cfg["fd_jac"][1]) if cfg["fd_jac"] else None
                if isinstance(w, GpuWorker):
                    hdr = dict(cmd="batch", jobs=jobs, stochastic=True)
                if cfg["gpu_explore"]:
                    hdr["explore"] = list(cfg["gpu_explore"])
                else:
                    j0 = jobs[0]
                    cond = cfg["rollout_cond"] if cfg["rollout_cond"] != "mix" else \
                        "dr" if rng.random() >= cfg["p_named_cond"] else \
                        CHALLENGING[rng.integers(len(CHALLENGING))]
                    hdr = dict(cmd="episode", id=j0["id"], goal=j0["goal"], cond=cond, seed=j0["seed"],
                               stochastic=True, land_time=1.0)
                if fd_hdr:
                    hdr["fd"] = fd_hdr
                w.send(hdr, cache[key])
                w.job = {j["id"]: (m, j["goal"]) for j in jobs}
                n_sent += n_b
            owed = cfg["pretrain_updates"] + cfg["utd"] * buf.n_online - st["n"]
            busy = any(w.job is not None for w in workers)
            if not busy and n_sent >= cfg["online_episodes"] and owed < 1:
                break
            for k_, _ in sel.select(timeout=0 if owed >= 1 else 0.2):
                w = k_.data
                header, ep = w.recv()
                m, g = w.job.pop(header.get("id"))
                if not w.job:
                    w.job = None
                if not header.get("ok"):
                    log(f"rollout failed: {header.get('error')}")
                    n_sent -= 1
                    continue
                buf.add(ep, online=True)
                n_done += 1
                st["prog"] = n_done / max(cfg["online_episodes"], 1)
                co["dirty"] = True
                fd_ok.append(header.get("fd_ok", 0.0))
                info = ep["info"]
                cw.writerow([f"{time.time() - T0:.1f}", n_done, st["n"], names[m], *g,
                             header["cond"], f"{info[0] * 100:.2f}", f"{info[1] * 100:.2f}",
                             f"{math.degrees(info[2]):.2f}", int(info[3]), f"{info[5]:.0f}",
                             f"{ep['rc'][-1, 2]:.2f}"])
                if n_done in snaps:
                    for mm in range(M):
                        np.savez(os.path.join(pdir, f"{names[mm]}_e{n_done}.npz"), **export(mm))
                    tags.append((f"e{n_done}", n_done, st["n"]))
                if n_done % 100 == 0:
                    curve.flush()
                    el = time.time() - t_on
                    log(f"episodes {n_done}/{cfg['online_episodes']} updates {st['n']} "
                        f"({el:.0f} s online"
                        + (f", FD Jacobians at {np.mean(fd_ok):.0%} of samples" if cfg["fd_jac"] else "")
                        + "): " + (fmt(stats) if stats else ""))
                    stats.clear()
                    fd_ok.clear()
            if (co["dirty"] and st["n"] - co["last"] >= 20) or st["n"] - co["last"] >= cfg["costate_every"]:
                recompute_costates()                  # co_fd: the current actor's co-states
                co.update(dirty=False, last=st["n"])
            for _ in range(int(max(0, min(owed, cfg["updates_per_poll"])))):
                stats.append(update(buf.sample(cfg["batch"])))
    finally:
        for w in workers:
            w.close()
        curve.close()
    if f"e{n_done}" not in [t for t, _, _ in tags]:
        for m in range(M):
            np.savez(os.path.join(pdir, f"{names[m]}_e{n_done}.npz"), **export(m))
        tags.append((f"e{n_done}", n_done, st["n"]))
    torch.save(dict(actor=actor.state_dict(), critic=qnet.state_dict(), variants=variants,
                    log_alpha=log_alpha.detach().cpu()), os.path.join(tdir, f"{gname}_final.pt"))
    _write_members(tdir, gname, gi, names, variants, tags, st["n"], n_done, T0)
    log(f"done: {n_done} episodes, {st['n']} batched updates ({M} members), "
        f"{time.time() - T0:.0f} s")


def _write_members(tdir, gname, gi, names, variants, tags, updates, episodes, T0):
    members = [dict(name=f"{names[m]}_{tag}", group=gi, variant=variants[m][0].split(".")[0],
                    tag=tag, episodes=e, updates=u)
               for m in range(len(names)) for tag, e, u in tags]
    with open(os.path.join(tdir, f"{gname}_members.json"), "w") as f:
        json.dump(dict(members=members, updates=updates, episodes=episodes,
                       wall=time.time() - T0), f, indent=1)


def _group_entry(gi, gpus, a, variants, bank, C_sec, C_cl=None):
    group_main(gi, gpus[gi % len(gpus)], a, variants, bank, C_sec, C_cl)


def stage_train(a, box, bank):
    import torch.multiprocessing as mp
    gpus = [int(g) for g in a.gpus.split(",")]
    variants = [(f"{name}.s{r}", v) for r in range(a.seeds_per_group)
                for name, v in parse_variants(a.variants)]
    n = a.groups or len(gpus)
    off = load_offline(a)
    C_sec = secant_fit(off, bank) if any(v["secant"] or v["sec_c"] for _, v in variants) \
        else None
    if C_sec is None and any(v["secant"] for _, v in variants):
        print("secant: the correction does not predict held-out runs better; left out")
    print(f"train: {n} groups on GPUs {gpus}, {len(variants)} members each "
          f"({', '.join(v for v, _ in variants)}), {a.workers} rollout workers each")
    t0 = time.time()
    C_cl = None
    if any(v["sec_cl"] for _, v in variants):
        C_cl = secant_fit(off, bank, run_filter=a.secant_runs or None)
        print(f"secant for the critic's co-states: runs matching {a.secant_runs!r}"
              + ("" if C_cl is not None else " -- no better than the model; not used"))
    mp.start_processes(_group_entry, args=(gpus, a, variants, bank, C_sec, C_cl), nprocs=n,
                       join=True, start_method="spawn")
    members = []
    for gi in range(n):
        members += json.load(open(os.path.join(a.out, "train", f"g{gi}_members.json")))["members"]
    pdir = os.path.join(a.out, "policy")
    with open(os.path.join(pdir, "bank.json"), "w") as f:
        json.dump(bank, f, indent=1)
    man = dict(best=members[-1]["name"], members=members, eval_goals=eval_goals(bank))
    with open(os.path.join(pdir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"policy: {pdir}, {len(members)} member snapshots ({time.time() - t0:.0f} s)")


def eval_goals(bank):
    G = np.array([p["goal"] for p in bank["plans"]], float)
    Dg = G - G[0]
    if np.linalg.matrix_rank(Dg, tol=1e-6) <= 1:
        order = np.argsort(Dg @ Dg[np.argmax(np.linalg.norm(Dg, axis=1))])
        return [(0.5 * (G[order[i]] + G[order[i + 1]])).tolist() for i in range(len(G) - 1)]
    return [G.mean(0).tolist()] + [(0.5 * (G[i] + G[j])).tolist()
                                   for i in range(len(G)) for j in range(i + 1, len(G))]


def stage_eval(a, box, bank):
    """Every member snapshot, zero-shot, on the held-out goals under every challenging
    condition (same robots as the ILC benchmark), and the manifest's best set by it."""
    pdir = os.path.join(a.out, "policy")
    man = json.load(open(os.path.join(pdir, "manifest.json")))
    jumps = " ".join(f"--jump {g[0]} {g[1]}" for g in man["eval_goals"])
    out = os.path.join(a.out, "eval_members.json")
    each = "each" + (f":{a.eval_variants}" if a.eval_variants else "")
    script = (f"{exe('dilc_execute.py')} run --policy {box.path(pdir)} --member {each} {jumps} "
              f"--conds {a.final_conds} --episodes {a.final_episodes} --seed {a.final_seed} "
              f"--jobs {a.jobs} --json {box.path(out)}")
    t0 = time.time()
    r = box.run(script, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr[-3000:])
        raise SystemExit("eval failed")
    rows = json.load(open(out))
    by = {}
    for x in rows:
        by.setdefault(x["member"], []).append(0.01 * x["landing_cost"] + 20 * x["fell"])
    score = {k: float(np.mean(v)) for k, v in by.items()}
    for m in man["members"]:
        m["score"] = score.get(m["name"], m.get("score"))
    man["best"] = min(score, key=score.get)
    with open(os.path.join(pdir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"eval: {len(rows)} jumps, {time.time() - t0:.0f} s; score (mean over "
          f"{len(next(iter(by.values())))} jumps; lower is better) by variant and snapshot:")
    table = {}
    for m in man["members"]:
        if m.get("score") is not None:
            table.setdefault((m["variant"], m.get("tag", f"e{m['episodes']}")), []).append(m["score"])
    for (v, e), s in sorted(table.items()):
        print(f"  {v:10s} @ {e:7s}: mean {np.mean(s):6.2f} median {np.median(s):6.2f}  "
              f"members {' '.join(f'{x:.2f}' for x in s)}")
    print(f"best member: {man['best']} ({score[man['best']]:.2f})")


def main():
    ws = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output folder, inside the workspace")
    ap.add_argument("--stages", default="bank,ilc,dataset,train,eval")
    ap.add_argument("--ws", default=ws, help="colcon workspace on the host (mounted at /ilc_ws)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() - 2)
    g = ap.add_argument_group("goals and ILC data")
    g.add_argument("--goals", default="f40_m85,f50_m85,f60_m85",
                   help="the ILC'd goals: plans ref_<prefix><goal>.npz")
    g.add_argument("--plans", default=os.path.join(HERE, "..", "experiments", "go1", "plans"))
    g.add_argument("--plan-prefix", default="s8_go1_")
    g.add_argument("--ilc-conds", default="challenging",
                   help="realities the ILC runs under (all asynchronous)")
    g.add_argument("--ilc-trials", type=int, default=20)
    g.add_argument("--ilc-seeds", type=int, nargs="+", default=[1],
                   help="an ILC run per goal, condition and seed (coverage)")
    g.add_argument("--sx", type=float, nargs=6, default=[0.01, 0.01, 0.03, 0.1, 0.1, 0.5],
                   help="state error scale in the observation")
    g.add_argument("--a-scale", type=float, nargs=4, default=[25.0, 50.0, 25.0, 50.0],
                   help="force offset at |a| = 1, N (front fx, fz, rear fx, fz)")
    g.add_argument("--qe", type=float, nargs=6, default=[3.0, 3.0, 3.0, 0.01, 0.01, 0.01],
                   help="the ILC's Qe (ilc_jump_lockstep.py's default)")
    g.add_argument("--qu", type=float, default=1e-5, help="the ILC's Stage III Qu")
    g.add_argument("--k-rscale", type=float, default=0.1,
                   help="the feedback labels' Riccati R, x Qu (0.1: the best LQR baseline)")
    g.add_argument("--k-crun", type=float, default=0.01,
                   help="the feedback labels' running (stage tracking) cost, x Qe")
    g.add_argument("--est-window", type=float, default=0.025)
    g.add_argument("--hist", type=int, default=3,
                   help="previous state errors in the observation (members with hist=0 "
                        "see zeros there)")
    t = ap.add_argument_group("VG-SAC")
    t.add_argument("--gpus", default="0,1,2,3")
    t.add_argument("--groups", type=int, default=0, help="learner processes (0: one per GPU)")
    t.add_argument("--variants", default=DEFAULT_VARIANTS,
                   help="members of every group, 'name:k=v,k=v;...' over "
                        + ", ".join(f"{k} ({v:g})" for k, v in VARIANT_KEYS.items()))
    t.add_argument("--seeds-per-group", type=int, default=2,
                   help="members per variant in every group (the batched networks keep the "
                        "GPU busy: 12 members at batch 1024 take about as long per update as "
                        "one did at 512)")
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() - 4) // 4),
                   help="rollout workers per group")
    t.add_argument("--online-episodes", type=int, default=800, help="per group")
    t.add_argument("--pretrain-snapshots", type=float, nargs="*", default=[],
                   help="also keep every member at these shares of the pretraining")
    t.add_argument("--offline-data", type=int, default=1,
                   help="0: no ILC trials in the replay (SAC from scratch)")
    t.add_argument("--rollout-cond", default="mix",
                   help="online rollouts' reality: mix (randomized + challenging) or a name")
    t.add_argument("--rollout-goals", type=float, nargs="*", default=[],
                   help="online rollouts only at these goals (dx, m)")
    t.add_argument("--gpu-rollouts", type=int, nargs="*", default=None,
                   help="fly the rollouts in the GPU sim (ilc_mjx: MJX, nominal, synchronous, PD landing) on "
                        "these GPUs (none listed: the group's own) instead of the container's async sim")
    t.add_argument("--gpu-workers", type=int, default=1, help="GPU rollout servers per group")
    t.add_argument("--co-recent", type=int, default=256, help="co_pg: the latest episodes the actor learns from")
    t.add_argument("--actor-lr", type=float, default=0.0, help="the actor's learning rate (0: --lr)")
    t.add_argument("--costate-every", type=int, default=200,
                   help="co_fd: updates between recomputing the online episodes' co-states")
    t.add_argument("--fresh-critic", action="store_true",
                   help="--init-from: the actor only; the critic starts from scratch")
    t.add_argument("--gpu-batch", type=int, default=128, help="jumps per GPU request")
    t.add_argument("--gpu-explore", type=float, nargs=2, default=[], metavar=("SIG_Q", "SIG_A"),
                   help="GPU rollouts' exploration: each jump's own stance (planar joint offsets ~ N(0, SIG_Q) rad) "
                        "and smooth action offsets of rms SIG_A over the contact samples -- the dynamics stay nominal")
    t.add_argument("--fd-jac", type=float, nargs=2, default=[], metavar=("EPS_A", "EPS_S"),
                   help="the rollouts' Jacobians from the simulator by finite differences "
                        "(dilc_execute.FDJac: action step EPS_A normalized, state step EPS_S x sx) "
                        "instead of the SRB model's -- VG-SAC-FD")
    t.add_argument("--snapshots", type=float, nargs="*", default=[0.5],
                   help="also keep every member at these shares of the episodes")
    t.add_argument("--pretrain-updates", type=int, default=3000)
    t.add_argument("--utd", type=float, default=0.25,
                   help="batched updates (each updates every member) per online transition")
    t.add_argument("--updates-per-poll", type=int, default=16)
    t.add_argument("--max-backlog", type=int, default=600,
                   help="updates the learner may owe before rollouts wait for it")
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--gamma", type=float, default=0.99)
    t.add_argument("--tau", type=float, default=0.005)
    t.add_argument("--alpha", type=float, default=0.02, help="initial entropy weight")
    t.add_argument("--target-entropy", type=float, default=-4.0)
    t.add_argument("--actor-hidden", type=int, default=256)
    t.add_argument("--critic-hidden", type=int, default=512)
    t.add_argument("--reward-scale", type=float, default=0.01)
    t.add_argument("--tau-actor", type=float, default=0.005,
                   help="the actor average's rate (anchor_ref=3)")
    t.add_argument("--rl-start", type=float, default=0.0,
                   help="share of the updates before the actor's RL term switches on "
                        "(before: the Sobolev network alone)")
    t.add_argument("--awr-beta", type=float, default=0.1, help="advantage temperature (awr)")
    t.add_argument("--critic-warmup", type=int, default=0,
                   help="updates before the actor's RL term switches on, the critic alone fitting the "
                        "policy it starts from (an inherited critic is not that policy's)")
    t.add_argument("--init-from", default="", help="fine-tune: a run folder to start from")
    t.add_argument("--small-init", action="store_true",
                   help="without --init-from: the actor's mean head near zero (starts as the plan, u = u_ref)")
    t.add_argument("--secant-runs", default="",
                   help="sec_cl: fit the critic's secant on runs whose name holds this")
    t.add_argument("--init-variant", default="",
                   help="--init-from: every member from this variant's members (matched starts)")
    t.add_argument("--init-loose", action="store_true",
                   help="--init-from by member position (new variant settings on old weights)")
    t.add_argument("--extra-data", nargs="*", default=[],
                   help="more labeled episodes (dilc_execute.py rollouts) for the replay")
    t.add_argument("--target-frac", type=float, default=0.0,
                   help="share of every batch (and of the value term's trials) drawn from "
                        "--extra-data, the new robot's episodes")
    t.add_argument("--nomw-sigma", type=float, default=0.03,
                   help="nomw: goal distance (m) over which the new robot's labels take over")
    t.add_argument("--aug-sigma", type=float, default=2.0,
                   help="state perturbation of the Sobolev augmentation (normalized units)")
    t.add_argument("--sob-episodes", type=int, default=16,
                   help="whole trials per update for the outcome-aware value term")
    t.add_argument("--fall-penalty", type=float, default=20.0)
    t.add_argument("--stage-schedule", type=float, nargs=2, default=[0.2, 0.5],
                   help="share of the pretraining where Stage II, then III takes over")
    t.add_argument("--stage-floor", type=float, default=0.0,
                   help="weight kept on the other stages' costs")
    t.add_argument("--p-bank-goal", type=float, default=0.25)
    t.add_argument("--p-named-cond", type=float, default=0.3)
    t.add_argument("--seed", type=int, default=0)
    x = ap.add_argument_group("ILC exploration for fine-tuning (stage explore)")
    x.add_argument("--explore-from", default="", help="a trained run folder")
    x.add_argument("--explore-variant", default="", help="whose members' disagreement")
    x.add_argument("--explore-goals", type=int, default=8)
    x.add_argument("--explore-acq", default="disagree", choices=("disagree", "critic", "cover"),
                   help="where to explore: the members' disagreement, or the critics' "
                        "predicted landing cost plus their spread (Deep-ILC's value function)")
    x.add_argument("--explore-kappa", type=float, default=1.0)
    x.add_argument("--explore-goal-list", type=float, nargs="*", default=[],
                   help="explore at these goals (dx, m) instead of the acquisition's")
    x.add_argument("--explore-member", default="", help="the policy the ILC starts from (default: the best)")
    x.add_argument("--explore-secant", action="store_true",
                   help="the exploration ILC's Stage III with the pooled secant correction")
    x.add_argument("--explore-data", nargs="*", default=[],
                   help="this robot's earlier exploration (labeled npz), for --explore-secant")
    x.add_argument("--explore-cond", default="nominal",
                   help="the robot explored (a named condition: the new environment)")
    x.add_argument("--explore-trials", type=int, default=4, help="ILC trials per goal")
    x.add_argument("--explore-seed", type=int, default=7001)
    x.add_argument("--explore-tag", default="r1")
    e = ap.add_argument_group("evaluation")
    e.add_argument("--eval-variants", default="", help="only these variants (comma list)")
    e.add_argument("--relabel-runs", nargs="*", default=[],
                   help="relabel: the new robot's ILC run folders (trial files)")
    e.add_argument("--relabel-out", default="", help="relabel: the relabeled episodes (npz)")
    e.add_argument("--relabel-mode", default="full", choices=("full", "step"),
                   help="full: every label from the corrected sensitivity (B corrected for K, "
                        "co-states); step: only the Stage III step S, the targets, the metric")
    e.add_argument("--final-conds", default="challenging")
    e.add_argument("--final-episodes", type=int, default=2)
    e.add_argument("--final-seed", type=int, default=101)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    a.plans = os.path.abspath(a.plans)
    box = Container(a.ws)
    stages = a.stages.split(",")
    bank = stage_bank(a, box) if "bank" in stages else \
        json.load(open(os.path.join(a.out, "bank.json")))
    if "ilc" in stages:
        stage_ilc(a, box, bank)
    if "secant" in stages:
        bank = stage_secant(a, box, bank)
    if "explore" in stages:
        stage_explore(a, box, bank)
    if "relabel" in stages:
        stage_relabel(a, box, bank)
    if "dataset" in stages:
        stage_dataset(a, box)
    if "train" in stages:
        stage_train(a, box, bank)
    if "eval" in stages:
        stage_eval(a, box, bank)


if __name__ == "__main__":
    main()
