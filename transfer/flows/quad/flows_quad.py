#!/usr/bin/env python3
"""
flows_quad.py (2026-10-22): the quad part of the flow analysis (../SPEC.md). Staged for the bigger machine; see
../run_all.sh and FLOWS.md.

The map: one control sample (10 ms) of the push. The closed-loop SRB state x = (CoM x, z, pitch, vx, vz, pitch rate)
at contact sample k -> k + 1 under the policy, for k < Nc. V(x) = F(x) - x.
  real   the "real" robot (dilc_execute's CPU async sim in the container: perturbed dynamics + timing + sensing), from
         its recorded TRUE pose (rec_true_*: mocap-free), velocities by a centred 20 ms slope -- what the robot did
  sim0   the nominal GPU sim's one-sample flight from the same state (restored, stance feet put on the floor, as
         deploy.py --jac-at real does) under the START network pi_0 at the robot's observation there
  simk   the same under the iteration-k network pi_k (how much the network itself changed the sim's field)
  plan   the goal's reference flow x_ref[k+1] - x_ref[k] (the TO plan for that goal)
Errors (in units of the bank's state scale sx, mean over the push's samples and the jumps):
  field_err      |V_real - V_plan|  the GOAL-referenced error (primary: on the car and the pogo the stage moves the
                                    real flow to the goal, not back to the sim flow -- ../../../../src/paper/HANDOFF_2026-10-22.md §2)
  field_err_sim  |V_real - V_sim0|  the SPEC's sim-referenced error (the corruption dV_0 at k = 0)
  policy_change  |V_simk - V_sim0|
  tube_err       |x_real - x_ref|   how far the visited states are from the plan
  task_err       the jump's score 0.01 x landing cost + 20 per fall (quad_table.py's), and ok5
Goal sets: IN = the stage's 6 training goals; HULL / OUT = the 8 planeval (validation, seed 301) goals inside /
outside their convex hull in (x, h). Never the test goals (planefinal, seed 701, spent).

Data: every stage cell <arm>_sd<seed>/<robot>/it<k>/policy (k = 0..4: it0 the start, it4 the final network).
  fly       flies each it<k> network at IN + HULL + OUT, EPISODES jumps each, on the cell's robot (validation seed 301,
            the same seed for every k so the timing draws repeat) -> <cell>/it<k>/flows_val/ (recorded episodes)
  compute   the fields per (cell, k) -> cache/<start>/<cell id>_it<k>.npz (GPU)
  aggregate summary.json (SPEC schema, lists [iter][unit], unit = robot:seed), summary_extra.json, figures
  portraits phase portraits (vx, vz) and (pitch, pitch rate) for one start / robot / goal over the iterations
    python flows_quad.py fly [--starts ...] [--par 4]          (container: ilc_quad; DILC_FLY_JOBS jumps in parallel)
    python flows_quad.py compute [--starts ...] [--gpu 1]
    python flows_quad.py aggregate
    python flows_quad.py portraits --portrait-start "ours+DR" --portrait-robot real_r4 --portrait-goal 0.6 0.1
Paths are absolute (/home/henry/ilc_ws). Every step skips outputs that exist.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

WS = "/home/henry/ilc_ws"
CWS = "/ilc_ws"
P = f"{WS}/log/dilc/plane"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{WS}/src/ilc_mjx")
sys.path.insert(0, f"{WS}/src/ilc_mjx/scripts")

IN = [(0.45, 0.0), (0.6, 0.0), (0.5, 0.15), (0.6, 0.1), (0.5, 0.05), (0.575, 0.15)]          # deploy.py --goals
VAL = [(0.4375, 0.0), (0.5625, 0.0), (0.53, 0.11), (0.51, 0.13), (0.56, 0.12), (0.52, 0.16), (0.4625, 0.06),
       (0.6125, 0.07)]                                                                          # planeval (cpu_eval.sh)
VAL_SEED = 301
ITERS = [0, 1, 2, 3, 4]
# start policy -> the stage arms' seed dirs (each holds <robot>/it<k>/policy). Missing ones are skipped.
STARTS = {
    "ours+DR": f"{P}/tr/recipe_sd*",
    "ours+DR random-dir (control)": f"{P}/tr/ctl_random_sd*",
    "PPO+DR": f"{P}/baseline_audit/after/hw24_on_ppo_dr_fix_s0_sd*",
    "ours+DR on clean twins": f"{P}/clean/recipe_sd*",
    "random-dir on clean twins": f"{P}/clean/ctl_random_sd*",
    "ours+DR pool": f"{P}/tr/pool_sd*",
}
for _d in sorted(glob.glob(f"{P}/stage_c_*")):                       # run_remaining_quad.sh's new-network arms
    for _arm in ("recipe", "pool", "ctl_random"):
        STARTS[f"{os.path.basename(_d)[8:]} {_arm}"] = f"{_d}/{_arm}_sd*"


def in_hull(g):
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull
    pts = np.array(IN)
    h = ConvexHull(pts)
    return bool(Path(pts[h.vertices]).contains_point(g, radius=1e-9))


SETS = {g: "IN" for g in IN}
SETS.update({g: ("HULL" if in_hull(g) else "OUT") for g in VAL})


def cells(start):
    out = []
    for sd in sorted(glob.glob(STARTS[start])):
        for rd in sorted(glob.glob(f"{sd}/real_*")):
            if all(os.path.exists(f"{rd}/it{k}/policy/policy.npz") for k in ITERS):
                out.append(rd)
    return out


def unit(cell):
    return f"{os.path.basename(cell)}:{os.path.basename(os.path.dirname(cell)).rsplit('_sd', 1)[1]}"


# -- fly ------------------------------------------------------------------------------------------------------------
def fly_one(cell, k, episodes, jobs):
    pdir, out = f"{cell}/it{k}/policy", f"{cell}/it{k}/flows_val"
    js = f"{out}/flows_val.json"
    if os.path.exists(js):
        return
    os.makedirs(out, exist_ok=True)
    rb = os.path.basename(cell)
    j = " ".join(f"--jump {g[0]} {g[1]}" for g in IN + VAL)
    ov = f"export PYTHONPATH={os.environ['ILC_QUAD_OVERLAY']}:$PYTHONPATH && " if os.environ.get("ILC_QUAD_OVERLAY") else ""
    c = lambda p: CWS + os.path.abspath(p)[len(WS):]
    cmd = (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && {ov}"
           f"python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py run --policy {c(pdir)} --member policy --conds {rb} "
           f"{j} --episodes {episodes} --seed {VAL_SEED} --jobs {jobs} --json {c(js)}.tmp --save-dir {c(out)}")
    t0 = time.time()
    with open(f"{out}/fly.log", "w") as lf:                  # the container's output (a failed flight says why here)
        subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"], stdout=lf, stderr=subprocess.STDOUT,
                       check=False, timeout=3600)
    if os.path.exists(js + ".tmp"):
        os.replace(js + ".tmp", js)
    print(f"flew {unit(cell)} it{k}: {len(glob.glob(out + '/*.npz'))} episodes, {time.time() - t0:.0f} s", flush=True)


def cmd_fly(a):
    work = [(c, k) for s in a.starts for c in cells(s) for k in ITERS]
    print(f"{len(work)} (cell, iteration) flights of {len(IN + VAL)} goals x {a.episodes}", flush=True)
    jobs = int(os.environ.get("DILC_FLY_JOBS", "8"))
    with ThreadPoolExecutor(a.par) as ex:
        list(ex.map(lambda ck: fly_one(ck[0], ck[1], a.episodes, jobs), work))


# -- compute --------------------------------------------------------------------------------------------------------
def real_states(z, env, ks, window=0.02):
    """(qpos, qvel) at samples ks from the TRUE recorded pose (rec_true_*), velocities by a centred least-squares slope
    over `window` (JumpEnv.restore's base-velocity code, centred, without the stance-foot projection)."""
    from ilc_mjx.jump import _quat_mul
    import jax.numpy as jnp
    m = env.m
    t, pos, quat, q = (np.asarray(z[k_], float) for k_ in ("rec_t", "rec_true_pos", "rec_true_quat", "rec_true_q"))
    qadr, vadr = np.asarray(env.qm.qpos_adr), np.asarray(env.qm.qvel_adr)
    Q, V = [], []
    for k in ks:
        i = min(int(k) * env.tps, len(t) - 1)
        sel = np.abs(t - t[i]) <= window / 2 + 1e-9
        tt = t[sel] - t[i]
        slope = lambda y: np.polyfit(tt, y, 1)[0] if len(tt) > 2 else np.zeros(y.shape[1:])
        qs = quat[sel] * np.sign((quat[sel] * quat[i]).sum(1, keepdims=True))
        dqt = slope(qs)
        w0, xv = quat[i][0], quat[i][1:]
        prod = np.array(_quat_mul(jnp.asarray([w0, -xv[0], -xv[1], -xv[2]]), jnp.asarray(dqt)))
        qp, qv = np.array(m.qpos0, float), np.zeros(m.nv)
        qp[0:3], qp[3:7], qp[qadr] = pos[i], quat[i] / np.linalg.norm(quat[i]), q[i]
        qv[0:3], qv[3:6], qv[vadr] = slope(pos[sel]), 2.0 * prod[1:], slope(q[sel])
        Q.append(qp)
        V.append(qv)
    return np.array(Q), np.array(V)


class Fields:
    def __init__(self, bank_path, gpu):
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax
        import jax.numpy as jnp
        from ilc_mjx.jump import Config, JumpEnv
        self.jax, self.jnp, self.JumpEnv = jax, jnp, JumpEnv
        self.bank = json.load(open(bank_path))
        self.env = env = JumpEnv(self.bank, Config(est_window=0.025))
        self.sx = np.asarray(self.bank["sx"], float)[:6]
        tmpl = env.stand()

        def srb_of(qpos, qvel, ref):
            B = qpos.shape[0]
            dx = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), tmpl).replace(qpos=qpos, qvel=qvel)
            dx = jax.vmap(lambda d, r: env.mjx.forward(env._model(r), d))(dx, ref)
            return jax.vmap(lambda d: env.srb(d, jnp.zeros(2)))(dx)

        def step(qpos, qvel, ks, acts, ref):
            B = qpos.shape[0]
            dx = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), tmpl).replace(qpos=qpos, qvel=qvel)
            dx = jax.vmap(lambda d, r: env.mjx.forward(env._model(r), d))(dx, ref)

            def one(d, k, a_, r):
                u, _ = env._forces(a_, k, r)
                return env.srb(d, jnp.zeros(2)), env.srb(env._fly_sample(d, k, u, r, True), jnp.zeros(2))
            return jax.vmap(one)(dx, ks, acts, ref)
        self.srb_of, self.step = jax.jit(srb_of), jax.jit(step)
        self.actor = jax.jit(jax.vmap(lambda w_, o: JumpEnv.actor(w_, o), (None, 0)))
        self._fz = {}

    def foot_z(self, g, w0):
        """the nominal sim's feet heights (Nc, 4) at goal g under pi_0 (rollout's foot_z): restore puts the stance feet
        at this soft-contact depth (as deploy.py --jac-at real does). Without it the feet sit exactly on the floor, carry
        no load in the first sample, and the one-sample flight falls freely (dvz = -g dt: the smoke test's artifact)"""
        if g not in self._fz:
            o = self.env.rollout(w0, self.env.references(np.array([g])), self.jax.random.PRNGKey(0), stochastic=False)
            self._fz[g] = np.asarray(o["foot_z"])[0]
        return self._fz[g]

    def weights(self, path):
        return {k: self.jnp.asarray(v, self.jnp.float32) for k, v in np.load(path).items() if k != "n_hidden"}

    def episode(self, z, w0, wk):
        env, jnp = self.env, self.jnp
        Nc, N = env.Nc, env.N
        g = tuple(np.round(np.asarray(z["goal"], float), 4))
        ref1 = env.references(np.array([g]))
        xr = np.asarray(ref1["x_ref"])[0]                                    # (N + 1, 6)
        refB = env.references(np.repeat([g], Nc, 0))
        ks = np.arange(Nc)
        rec = {k_[4:]: z[k_] for k_ in z.files if k_.startswith("rec_")}
        rec["pos"], rec["quat"], rec["q"] = rec["true_pos"], rec["true_quat"], rec["true_q"]
        Qr, Vr = env.restore(rec, ks, ks < env.Ndc, foot_z=self.foot_z(g, w0))   # projected: the sim's start states
        Qt, Vt = real_states(z, env, np.arange(Nc + 1))                      # what the robot did
        refB1 = env.references(np.repeat([g], Nc + 1, 0))
        xt = np.array(self.srb_of(jnp.asarray(Qt, jnp.float32), jnp.asarray(Vt, jnp.float32), refB1))
        xt[:, :2] += xr[0, :2] - xt[0, :2]                                   # the plan's frame: start where it starts
        obs = jnp.asarray(np.asarray(z["obs"], float)[:Nc], jnp.float32)
        out = {}
        for nm, w in (("sim0", w0), ("simk", wk)):
            acts = self.actor(w, obs)
            s0, s1 = self.step(jnp.asarray(Qr, jnp.float32), jnp.asarray(Vr, jnp.float32), jnp.asarray(ks),
                               acts, refB)
            out[nm] = np.asarray(s1) - np.asarray(s0)
        info = np.asarray(z["info"], float)
        return dict(goal=np.array(g), x_real=xt, v_real=np.diff(xt, axis=0), v_sim0=out["sim0"], v_simk=out["simk"],
                    v_plan=np.diff(xr[:Nc + 1], axis=0), x_ref=xr[:Nc + 1], fell=bool(z["fell"]), e3=info[:3],
                    ok5=bool((not bool(z["fell"])) and abs(info[0]) <= 0.05 and abs(info[1]) <= 0.03))


def cmd_compute(a):
    F = None
    for s in a.starts:
        for c in cells(s):
            cid = unit(c).replace(":", "_")
            for k in ITERS:
                dst = f"{HERE}/cache/{s.replace(' ', '_').replace('/', '-')}/{cid}_it{k}.npz"
                eps = sorted(glob.glob(f"{c}/it{k}/flows_val/*.npz"))
                if os.path.exists(dst) or not eps:
                    continue
                if F is None:
                    F = Fields(f"{c}/it0/policy/bank.json", a.gpu)
                w0, wk = F.weights(f"{c}/it0/policy/policy.npz"), F.weights(f"{c}/it{k}/policy/policy.npz")
                js = {(tuple(np.round(r["goal"], 4)), r["seed"]): r for r in json.load(open(f"{c}/it{k}/flows_val/flows_val.json"))} \
                    if os.path.exists(f"{c}/it{k}/flows_val/flows_val.json") else {}
                rows = []
                for f in eps:
                    z = np.load(f)
                    try:
                        r = F.episode(z, w0, wk)
                    except Exception as e:                                   # a broken recording: skip it, say so
                        print(f"  skip {f}: {e}", flush=True)
                        continue
                    sd_ = int(os.path.basename(f)[:-4].rsplit("_s", 1)[1])       # policy_g<x>_<h>_<robot>_s<seed>.npz
                    v = js.get((tuple(r["goal"]), sd_))
                    r["score"] = 0.01 * v["landing_cost"] + 20.0 * v["fell"] if v else np.nan
                    rows.append(r)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                np.savez(dst, **{key: np.array([r_[key] for r_ in rows]) for key in rows[0]}) if rows else None
                print(f"{s} {unit(c)} it{k}: {len(rows)} episodes", flush=True)


# -- aggregate ------------------------------------------------------------------------------------------------------
def load_cache(s):
    d = f"{HERE}/cache/{s.replace(' ', '_').replace('/', '-')}"
    out = {}
    for f in sorted(glob.glob(f"{d}/*_it*.npz")):
        cid, k = os.path.basename(f)[:-4].rsplit("_it", 1)
        out[(cid, int(k))] = dict(np.load(f))
    return out


def metrics(z, sx, sel):
    """per jump means over the push of the normalized errors, the jumps in `sel` (no falls for the fields)"""
    nf = sel & ~z["fell"]
    nv = lambda v: np.linalg.norm(v / sx, axis=-1).mean(-1)                 # (jumps,)
    m = {}
    m["field_err"] = nv(z["v_real"] - z["v_plan"])[nf]
    m["field_err_sim"] = nv(z["v_real"] - z["v_sim0"])[nf]
    m["policy_change"] = nv(z["v_simk"] - z["v_sim0"])[nf]
    m["tube_err"] = nv(z["x_real"] - z["x_ref"])[nf]
    m["task_err"] = z["score"][sel] if "score" in z else np.full(sel.sum(), np.nan)
    m["ok5"] = z["ok5"][sel].astype(float)
    return {k: float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else float("nan") for k, v in m.items()}


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float("nan")] * 3
    r = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return [float(x.mean()), *map(float, np.percentile(x[r].mean(1), [2.5, 97.5]))]


def cmd_aggregate(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bank = json.load(open(glob.glob(f"{P}/tr/recipe_sd9001/real_r1/it0/policy/bank.json")[0]))
    sx = np.asarray(bank["sx"], float)[:6]
    summ = dict(domain="quad", map="one control sample (10 ms) of the push, SRB state", proj=["vx", "vz"], starts={},
                notes=__doc__.strip().split("\n\n")[1] + " Lists are [iter][unit], unit = robot:seed.")
    extra = {}
    for s in STARTS:
        C = load_cache(s)
        units = sorted({cid for cid, _ in C})
        if not units:
            continue
        st = dict(iters=ITERS, robots=[u[::-1].replace("_", ":", 1)[::-1] for u in units],
                  field_err={}, task_err={})
        ex = {}
        for S in ("IN", "HULL", "OUT"):
            per = {key: [] for key in ("field_err", "field_err_sim", "policy_change", "tube_err", "task_err", "ok5")}
            for k in ITERS:
                row = {key: [] for key in per}
                for u in units:
                    z = C.get((u, k))
                    if z is None:
                        for key in row:
                            row[key].append(float("nan"))
                        continue
                    sel = np.array([SETS.get(tuple(np.round(g, 4))) == S for g in z["goal"]])
                    mt = metrics(z, sx, sel) if sel.any() else {key: float("nan") for key in row}
                    for key in row:
                        row[key].append(mt[key])
                for key in per:
                    per[key].append(row[key])
            st["field_err"][S], st["task_err"][S] = per["field_err"], per["task_err"]
            ex[S] = {key: dict(per_iter=per[key], ci=[boot(v) for v in per[key]]) for key in per}
        summ["starts"][s] = st
        extra[s] = ex
    json.dump(summ, open(f"{HERE}/summary.json", "w"), indent=1)
    json.dump(extra, open(f"{HERE}/summary_extra.json", "w"), indent=1)
    for key, fn, lab in (("field_err", "fig_field_err.png", "|V_real - V_plan| (goal-referenced)"),
                         ("field_err_sim", "fig_field_err_sim.png", "|V_real - V_sim,pi0| (sim-referenced)"),
                         ("task_err", "fig_task_err.png", "score (0.01 landing cost + 20/fall)"),
                         ("ok5", "fig_ok5.png", "ok5")):
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        for i, S in enumerate(("IN", "HULL", "OUT")):
            for s, ex in extra.items():
                ci = np.array(ex[S][key]["ci"])
                if np.isfinite(ci[:, 0]).any():
                    ax[i].plot(ITERS, ci[:, 0], marker="o", label=s)
                    ax[i].fill_between(ITERS, ci[:, 1], ci[:, 2], alpha=0.15)
            ax[i].set_title(S)
            ax[i].set_xlabel("hardware-stage iteration")
        ax[0].set_ylabel(lab)
        ax[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{HERE}/{fn}", dpi=130)
        plt.close(fig)
    print(f"summary: {list(summ['starts'])}")


def cmd_portraits(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = load_cache(a.portrait_start)
    g0 = tuple(np.round(a.portrait_goal, 4))
    us = sorted({cid for cid, _ in C if cid.startswith(a.portrait_robot + "_")})
    if not us:
        sys.exit(f"no cache for {a.portrait_start} / {a.portrait_robot}")
    u = us[0]
    for proj, (i, j), nm in (("(vx, vz)", (3, 4), "v"), ("(pitch, pitch rate)", (2, 5), "pitch")):
        fig, ax = plt.subplots(1, len(ITERS), figsize=(3.2 * len(ITERS), 3.2), sharex=True, sharey=True)
        for c_, k in enumerate(ITERS):
            z = C.get((u, k))
            if z is None:
                continue
            for e in np.where([tuple(np.round(g, 4)) == g0 for g in z["goal"]])[0]:
                X, Xr = z["x_real"][e], z["x_ref"][e]
                ax[c_].plot(Xr[:, i], Xr[:, j], "k-", lw=1, label="plan" if e == 0 else None)
                ax[c_].quiver(X[:-1, i], X[:-1, j], z["v_real"][e][:, i], z["v_real"][e][:, j], color="C3",
                              angles="xy", scale_units="xy", scale=1, width=0.004, label="real" if e == 0 else None)
                ax[c_].quiver(X[:-1, i], X[:-1, j], z["v_sim0"][e][:, i], z["v_sim0"][e][:, j], color="C0",
                              angles="xy", scale_units="xy", scale=1, width=0.003, alpha=0.6,
                              label="sim pi_0" if e == 0 else None)
            ax[c_].set_title(f"it {k}")
        ax[0].legend(fontsize=7)
        fig.suptitle(f"{a.portrait_start}, {u}, goal {g0}: {proj}")
        fig.tight_layout()
        fig.savefig(f"{HERE}/portrait_{a.portrait_start.replace(' ', '_')}_{u}_{nm}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fly", "compute", "aggregate", "portraits", "list"))
    ap.add_argument("--starts", nargs="+", default=None)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--par", type=int, default=3, help="fly: cells flown at once (each DILC_FLY_JOBS jumps in parallel)")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--portrait-start", default="ours+DR")
    ap.add_argument("--portrait-robot", default="real_r4")
    ap.add_argument("--portrait-goal", type=float, nargs=2, default=[0.6, 0.1])
    a = ap.parse_args()
    a.starts = a.starts or [s for s in STARTS if cells(s)]
    if a.cmd == "list":
        for s in STARTS:
            print(f"{s:40s} {len(cells(s))} cells")
        print("goal sets:", {g: SETS[g] for g in IN + VAL})
    else:
        dict(fly=cmd_fly, compute=cmd_compute, aggregate=cmd_aggregate, portraits=cmd_portraits)[a.cmd](a)
