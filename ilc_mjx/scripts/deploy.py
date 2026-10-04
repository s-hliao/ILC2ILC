#!/usr/bin/env python3
"""
deploy.py --policy DIR --member M --robots real_r1 [...] --out OUT: the hardware stage of deep ILC -- few-shot ILC
of the goal-conditioned policy on the robot (here ilc_quad's CPU async real-like robots), the structure from the
nominal simulator, every trial used. Per robot, from the same starting policy:

  each iteration (--iters, one jump at each of --goals: 10 x 3 = 30 jumps)
    1. the current policy flies --goals on the robot (dilc_execute.py run in the container, recorded)
    2. the gradient of each trial's MEASURED stage cost in its actions, g = d J / d a_k:
         fd   the nominal GPU sim's one-step FD Jacobians along the same goal, chained into closed-loop co-states
              with the policy's feedback K = d pi / d e at the REAL states (ilc_mjx)
         srb  the SRB model's closed-loop co-states along the real trial (JumpILC's model, dilc_execute)
    3. the ILC step per goal along -g (normalized; --momentum averages a goal's directions over its trials), of a
       fixed rms (--step-rms, shrinking as 1/sqrt(trials at the goal); halved after a fall, which rolls the goal's
       targets back to its best trial)
    4. the network regresses onto every trial's targets (weights halving with age: all runs used, the latest count
       most), anchored to its own previous actions on the GPU sim's states over the goal range, weight
       --anchor-w x N0 / (N0 + real trials)
  then the evaluation on the robot: --eval val (goals between the training goals, seed 301: for tuning) or final
  (the reserved test, goals 0.4375..0.5625 x 4 seeds from 701, and --perturbed its 8 perturbations).
-> OUT/<robot>/: per iteration the policy and the real trials, deploy.log, summary.json.
Run from src/ilc_mjx with the ilcmjx env; the container must be up.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--member", required=True)
ap.add_argument("--robots", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--goals", type=float, nargs="+", default=[0.425, 0.5, 0.575])
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--grad", default="fd", choices=("fd", "srb"))
ap.add_argument("--step-rms", type=float, default=0.05)
ap.add_argument("--momentum", type=float, default=0.0)
ap.add_argument("--update", default="grad", choices=("grad", "gn"),
                help="grad: fixed-size steps along -dJ/da; gn: the ILC's Gauss-Newton step on the landing error "
                     "(x, z, pitch) through the closed-loop landing sensitivity G_N (fd Jacobians), "
                     "-gn-beta G_N' (G_N G_N' + gn-delta I)^-1 e, its rms capped at --step-rms")
ap.add_argument("--gn-beta", type=float, default=0.5)
ap.add_argument("--gn-delta", type=float, default=0.1, help="damping, relative to tr(G_N G_N')/3")
ap.add_argument("--stage-w", type=float, nargs=3, default=[0.0, 0.0, 1.0])
ap.add_argument("--anchor-n0", type=float, default=6.0)
ap.add_argument("--anchor-w", type=float, default=1.0)
ap.add_argument("--age-decay", type=float, default=0.5)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=9001)
ap.add_argument("--eval", default="val", choices=("val", "final", "holdout", "none"))
ap.add_argument("--perturbed", action="store_true")
ap.add_argument("--target-base", default="flown", choices=("flown", "own"),
                help="the step starts from the action the robot flew (after the controller's clipping: stays "
                     "feasible) or from the network's own output (unclipped)")
ap.add_argument("--eval-episodes", type=int, default=4)
ap.add_argument("--secant", type=float, default=0.0,
                help="gn: > 0 corrects the landing sensitivity from the robot's own trial pairs at a goal (Broyden, this "
                     "weight): S + C, C += w (de - (S + C) da) da' / |da|^2, da the step commanded, de the measured change")
ap.add_argument("--rollback", type=float, nargs=2, default=[], metavar=("RATIO", "ABS"),
                help="a goal whose J exceeds RATIO x its best AND its best + ABS (a margin noise rarely crosses) rolls "
                     "its targets back to its best trial's actions, its steps halved")
ap.add_argument("--stall", type=int, default=0,
                help="> 0: a goal whose J has not fallen below 0.9 x its J this many trials earlier stops stepping for good: "
                     "its targets roll back to its best trial (a goal the robot cannot reach must not drag its neighbours "
                     "through the shared network)")
ap.add_argument("--bold", type=float, default=0.0,
                help="> 0: a goal whose J rose past (1 + this) x its previous J halves its step scale, else it grows "
                     "x1.25 back to 1 (the ILC's step-size safeguard, with a noise margin)")
ap.add_argument("--control", default="none", choices=("none", "random"),
                help="causal control: random replaces every step by a random direction of the same rms (same loop, "
                     "safeguards, regression and anchor), so any gain left is the refit's, not the ILC's")
ap.add_argument("--reps", type=int, default=1,
                help="jumps per goal per iteration (seeds): gn steps on their mean landing error and sensitivity, the "
                     "safeguards on their mean J, every jump a target (the noise floor: --iters 5 --reps 2 is 30 jumps)")
ap.add_argument("--jac-at", default="sim", choices=("sim", "real"),
                help="gn: the one-step Jacobians of the nominal sim along its own trajectory for the goal (sim), or with the "
                     "sim restored to each trial's measured states (real: mocap pose and joint encoders, the stance feet "
                     "at the sim's contact depth), the trial's flown actions")
ap.add_argument("--backend", default="container", choices=("container", "manual"),
                help="container: the sim robots fly (dilc_execute in the ilc_quad container); manual: a real robot -- "
                     "each iteration writes OUT/<robot>/it<k>/REQUEST.md (the jumps to fly with policy_jump_node) and "
                     "waits for their episodes in OUT/<robot>/it<k>/real; --robots are then just labels")
ap.add_argument("--gate", type=float, nargs=3, default=None, metavar=("FALLS", "RATIO", "DA"),
                help="safety gate on every update before it may fly (default on with --backend manual: 0 1.5 0.3): "
                     "in the nominal GPU sim over the goal range, nominal and from perturbed stances, the new policy "
                     "may fall at most FALLS more jumps and score at most RATIO x the current one (+0.5), and its "
                     "actions on the robot's own recent states may move at most DA (normalized) from the current "
                     "ones; else the update is rejected (the current policy flies again, its steps halved)")
ap.add_argument("--manual-timeout", type=float, default=24 * 3600, help="manual: seconds to wait for the jumps")
ap.add_argument("--gpu", default="1")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx import host_path  # noqa: E402
from ilc_mjx.jump import JumpEnv  # noqa: E402

WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.abspath(p)[len(WS):]
EVAL = dict(val=([0.45, 0.4625, 0.5375, 0.55], 301), final=([0.4375, 0.4875, 0.5125, 0.5625], 701),
            holdout=([0.4375, 0.4625, 0.4875, 0.5125, 0.5375, 0.5625], 1301))   # holdout: the frozen final test only

bank = json.load(open(os.path.join(a.policy, "bank.json")))
env = JumpEnv(bank)
Nc, Ndc = env.Nc, env.Ndc
gamma = float(bank.get("gamma", 0.99))
D = np.asarray(bank["sx"], float)
c = np.asarray(a.stage_w, float)
mask = np.ones((Nc, 4))
mask[Ndc:, :2] = 0.0
W0 = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(a.policy, a.member + ".npz")).items()
      if k != "n_hidden"}
actor_mean = jax.jit(jax.vmap(lambda w_, o: JumpEnv.actor(w_, o), (None, 0)))
dpi_fn = jax.jit(jax.vmap(jax.jacfwd(lambda o, w_: JumpEnv.actor(w_, o)), (0, None)))
srb_maker = None
if a.grad == "srb" or a.update == "gn":
    from dilc_execute import EpisodeMaker, GoalBank
    gb = GoalBank(dict(bank, plans=[dict(p, path=host_path(p["path"])) for p in bank["plans"]]))
    srb_maker = EpisodeMaker(gb, "/home/henry/mujoco_menagerie")


def export(w, d):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "policy.npz"), **{k: np.asarray(v) for k, v in w.items()}, n_hidden=np.array(2))
    for f in ("bank.json", "ilc_U.npz", "ilc_best.json"):
        if os.path.exists(os.path.join(a.policy, f)):
            shutil.copy(os.path.join(a.policy, f), d)
    json.dump(dict(best="policy", eval_goals=None, members=[dict(name="policy", group=0, variant="policy", tag="policy",
                                                                 episodes=0, updates=0, score=None)]),
              open(os.path.join(d, "manifest.json"), "w"))


def fly(pdir, conds, jumps, seed, json_out, save_dir="", episodes=1):
    """dilc_execute.py run on the robot (the container's CPU async sim)."""
    j = " ".join(f"--jump {g} 0" for g in jumps)
    sd = f"--save-dir {cpath(save_dir)}" if save_dir else ""
    cmd = (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && "
           f"python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py run --policy {cpath(pdir)} --member policy "
           f"--conds {conds} {j} --episodes {episodes} --seed {seed} --jobs 8 --json {cpath(json_out)} {sd}")
    for attempt in range(2):                     # a hung jump (the executor's pool can deadlock) must not stall the run
        try:
            subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False, timeout=900)
            break
        except subprocess.TimeoutExpired:
            tag = os.path.abspath(json_out)[len(WS):]
            subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad pkill -f {CWS + tag}"], check=False)
            print(f"flight timed out ({json_out}), attempt {attempt + 1}", flush=True)
    return json.load(open(json_out)) if os.path.exists(json_out) else []


def score(r):
    return 0.01 * r["landing_cost"] + 20.0 * r["fell"]


def adam_fit(w, loss, steps):
    vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
    mo = jax.tree_util.tree_map(jnp.zeros_like, w)
    vo = jax.tree_util.tree_map(jnp.zeros_like, w)
    aux = None
    for s in range(1, steps + 1):
        (_, aux), gr = vg(w)
        mo = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, mo, gr)
        vo = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ ** 2, vo, gr)
        w = jax.tree_util.tree_map(lambda p, m_, v_: p - a.lr * (m_ / (1 - 0.9 ** s))
                                   / (jnp.sqrt(v_ / (1 - 0.999 ** s)) + 1e-8), w, mo, vo)
    return w, aux


def gradient(z, w, gi, An, Bn, g):
    """d J / d a_k of one real trial (J its discounted stage cost, 0.01 x), normalized action units."""
    obs = np.asarray(z["obs"], float)
    if a.grad == "srb":
        rf = srb_maker.bank.reference(np.array([g, 0.0]))[0]
        lab = srb_maker.ilc_labels(rf, np.asarray(z["U"], float), np.asarray(z["logX"], float), np.array([g, 0.0]), rf)
        return 0.01 * np.einsum("ksa,s->ka", lab["mca_cl"], c) * (gamma ** np.arange(Nc))[:, None] * mask
    K = np.asarray(dpi_fn(jnp.asarray(obs[:Nc], jnp.float32), w))[..., :6]        # at the REAL states
    gk = 0.01 * np.einsum("ksi,s->ki", np.asarray(z["gn"], float), c)               # measured cost gradients
    m = gk[Nc - 1]
    G = np.zeros((Nc, 4))
    for k in range(Nc - 1, -1, -1):
        if k < Nc - 1:
            m = gk[k] + gamma * (An[gi, k + 1] + Bn[gi, k + 1] @ K[k + 1]).T @ m
        G[k] = gamma ** k * (Bn[gi, k].T @ m) * mask[k]
    return G


PHI = None
if a.update == "gn":                     # the ballistic flight map takeoff -> touchdown (exact for the CoM)
    Af = srb_maker.srb.f_lin(np.zeros(6), np.zeros(4), np.zeros(2), np.zeros(2), env.dt)[0].full()
    PHI = np.linalg.matrix_power(Af, env.N - Nc)


def landing_sensitivity(z, w, gi, An, Bn):
    """G_N (3, Nc, 4): d (landing x, z, pitch) / d a_k, closed loop (the policy's feedback at the REAL states)."""
    K = np.asarray(dpi_fn(jnp.asarray(np.asarray(z["obs"], float)[:Nc], jnp.float32), w))[..., :6]
    S = np.zeros((3, Nc, 4))
    for i in range(3):
        m = PHI[i] * D                                              # d x_N,i / d e_Nc (normalized)
        for k in range(Nc - 1, -1, -1):
            if k < Nc - 1:
                m = (An[gi, k + 1] + Bn[gi, k + 1] @ K[k + 1]).T @ m
            S[i, k] = (Bn[gi, k].T @ m) * mask[k]
    return S


GATE = a.gate if a.gate is not None else ([0, 1.5, 0.3] if a.backend == "manual" else None)


def gate(w_old, w_new, O_real, g_lo, g_hi, it):
    """The safety gate (--gate): the update in the nominal GPU sim against the current policy, and how far it moves
    the actions on the robot's own states."""
    goals = np.stack([np.linspace(g_lo, g_hi, 16), np.zeros(16)], 1)
    ref = env.references(np.concatenate([goals, goals]))
    q_off, _ = env.explore_noise(32, np.random.default_rng(777 + it), 0.08, 0.0)
    q_off[:16] = 0.0                                     # 16 nominal stances, 16 perturbed ones
    xr = np.asarray(ref["x_ref"])
    tg = xr[:, env.N].copy()
    tg[:, :2] = xr[:, 0, :2] + np.concatenate([goals, goals])
    tg[:, 2] = 0.0
    qe, rs = np.asarray(bank["qe"], float), float(bank["r_scale"])
    res = []
    for w_ in (w_old, w_new):
        o = env.rollout(w_, ref, jax.random.PRNGKey(0), stochastic=False, q_offset=q_off)
        e = o["X"][:, env.N] - tg
        sc = 0.01 * rs * (e * qe * e).sum(1) + 20 * o["fell"]
        res.append((int(o["fell"].sum()), float(np.nanmean(sc))))
    da = float(np.abs((np.asarray(actor_mean(w_new, O_real)) - np.asarray(actor_mean(w_old, O_real)))
                      * np.tile(mask, (len(O_real) // Nc, 1))).max())
    (f0, s0), (f1, s1) = res
    ok = f1 <= f0 + GATE[0] and s1 <= GATE[1] * s0 + 0.5 and da <= GATE[2]
    return ok, (f"{'pass' if ok else 'REJECTED'}: sim falls {f0} -> {f1}, sim score {s0:.2f} -> {s1:.2f}, "
                f"max action change on the robot's states {da:.3f}")


def request_jumps(pdir, robot, it, rdir, out):
    """--backend manual: the jumps this iteration needs, written for the operator (REQUEST.md), then wait for their
    episodes (policy_jump_node.py writes them) in rdir."""
    os.makedirs(rdir, exist_ok=True)
    need = {round(g, 4): a.reps for g in a.goals}
    lines = [f"# {robot}, iteration {it}: {len(a.goals) * a.reps} jumps", "",
             f"Policy: `{pdir}`. Save the episodes to `{rdir}` (episode_dir). For each goal, {a.reps} jump(s):", ""]
    for g in a.goals:
        ref = os.path.join(rdir, f"ref_{g:.4f}.npz")
        lines += [f"## goal {g:.4f} m", "```",
                  f"python3 policy_jump_node.py prepare --policy {pdir} --goal {g} 0 --out {ref}",
                  f"ros2 launch ilc_quad policy_jump_go1.launch.py pose_topic:=<mocap topic> policy_dir:={pdir} \\",
                  f"    reference_file:={ref} jump_dx:={g} jump_dz:=0.0 episode_dir:={rdir} episode_tag:={robot}_it{it}",
                  f"ros2 service call /start_trial std_srvs/srv/Trigger    # x{a.reps}", "```", ""]
    lines += ["A fall is data too: keep its episode (the update rolls that goal back). To abort the run, write "
              f"`{os.path.join(rdir, 'ABORT')}`.", ""]
    open(os.path.join(out, f"it{it}", "REQUEST.md"), "w").write("\n".join(lines))
    json.dump(dict(robot=robot, it=it, policy=pdir, episode_dir=rdir, goals=a.goals, reps=a.reps),
              open(os.path.join(out, f"it{it}", "REQUEST.json"), "w"), indent=1)
    print(f"[{robot}] it {it}: waiting for {len(a.goals) * a.reps} jumps -- {os.path.join(out, f'it{it}', 'REQUEST.md')}",
          flush=True)
    t0 = time.time()
    while True:
        if os.path.exists(os.path.join(rdir, "ABORT")):
            raise SystemExit(f"[{robot}] aborted by the operator")
        have = {}
        for f in os.listdir(rdir):
            if f.endswith(".npz") and not f.startswith("ref_"):
                try:
                    have[round(float(np.load(os.path.join(rdir, f))["goal"][0]), 4)] = \
                        have.get(round(float(np.load(os.path.join(rdir, f))["goal"][0]), 4), 0) + 1
                except Exception:                        # still being written
                    pass
        if all(have.get(g, 0) >= n for g, n in need.items()):
            return
        if time.time() - t0 > a.manual_timeout:
            raise SystemExit(f"[{robot}] timed out waiting for the jumps")
        time.sleep(5)


def run_robot(robot):
    out = os.path.join(a.out, robot)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "deploy.log"), "w")

    def say(msg):
        print(f"[{robot}] {msg}", flush=True)
        logf.write(msg + "\n")
        logf.flush()
    w = dict(W0)
    rng = np.random.default_rng(a.seed)
    g_lo, g_hi = min(a.goals) - 0.025, max(a.goals) + 0.025
    trials, best, hist = [], {g: None for g in a.goals}, []
    scale = {g: 1.0 for g in a.goals}
    sec = {g: None for g in a.goals}         # Broyden correction of the landing sensitivity, per goal
    prev = {g: None for g in a.goals}        # (e3, commanded step, J) of the goal's previous trial
    frozen = {g: False for g in a.goals}
    Jh = {g: [] for g in a.goals}
    mom = {g: None for g in a.goals}
    for it in range(a.iters + 1):
        pdir = os.path.join(out, f"it{it}", "policy")
        export(w, pdir)
        if it == a.iters:
            break
        t0 = time.time()
        rdir = os.path.join(out, f"it{it}", "real")
        if a.backend == "container":                    # (manual: jumps already flown for this iteration are kept)
            shutil.rmtree(rdir, ignore_errors=True)
        if a.backend == "manual":
            request_jumps(pdir, robot, it, rdir, out)
        else:
            fly(pdir, robot, a.goals, a.seed + 101 * it, os.path.join(out, f"it{it}", "real.json"), rdir,
                episodes=a.reps)
        eps = {}
        for f in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
            if not f.endswith(".npz") or f.startswith("ref_"):
                continue
            z = np.load(os.path.join(rdir, f))
            eps.setdefault(round(float(z["goal"][0]), 4), []).append(z)
        An = Bn = None
        if a.grad == "fd" or a.update == "gn":
            goals = np.array([[g, 0.0] for g in a.goals])
            sim = env.rollout(w, env.references(goals), jax.random.PRNGKey(it), stochastic=False, fd=True)
            An = sim["A"] * D[None, None, None, :] / D[None, None, :, None]
            Bn = sim["B"] / D[None, None, :, None]
        line = []
        for gi, g in enumerate(a.goals):
            zs = eps.get(round(g, 4))
            if not zs:
                line.append(f"g{g:.3f} missing")
                continue
            z = zs[0]
            own = lambda z_: (np.asarray(actor_mean(w, jnp.asarray(np.asarray(z_["obs"], float)[:Nc], jnp.float32))) * mask
                              if a.target_base == "own" else np.asarray(z_["act"], float))
            obs, fell, act = np.asarray(z["obs"], float), any(bool(z_["fell"]) for z_ in zs), own(z)
            Js = [float(0.01 * (np.asarray(z_["rc"]) @ c * gamma ** np.arange(Nc)).sum()) for z_ in zs]
            J = float(np.mean(Js))
            hist.extend(dict(it=it, goal=g, J=J_, fell=bool(z_["fell"])) for J_, z_ in zip(Js, zs))
            Jh[g].append(J)
            if a.stall > 0 and not frozen[g] and len(Jh[g]) > a.stall and min(Jh[g][-a.stall:]) > 0.9 * Jh[g][-a.stall - 1]:
                frozen[g] = True                                   # no progress: stop stepping this goal
                trials[:] = [t for t in trials if t["goal"] != g]
            if frozen[g]:
                if best[g] is not None and not any(t["goal"] == g for t in trials):
                    trials.append(dict(goal=g, it=it, obs=best[g][1], tgt=best[g][2]))
                for t in trials:                                   # the hold keeps full weight
                    if t["goal"] == g:
                        t["it"] = it
                line.append(f"g{g:.3f} J {J:.2f} FROZEN (held at its best)")
                continue
            runaway = bool(a.rollback) and best[g] is not None and J > a.rollback[0] * best[g][0] \
                and J > best[g][0] + a.rollback[1]
            if fell or runaway:                                    # back to the best trial, half the steps
                scale[g] *= 0.5
                if best[g] is not None:
                    trials.append(dict(goal=g, it=it, obs=best[g][1], tgt=best[g][2]))
                line.append(f"g{g:.3f} {'FELL' if fell else f'J {J:.2f} RUNAWAY'} (rollback, scale {scale[g]:.2f})")
                continue
            if best[g] is None or J < best[g][0]:
                best[g] = (J, obs[:Nc], act)
            if a.bold > 0 and prev[g] is not None:
                scale[g] = scale[g] * 0.5 if J > (1 + a.bold) * prev[g][2] else min(1.0, scale[g] * 1.25)
            if a.update == "gn":
                if a.jac_at == "real":
                    Sl = []
                    for z_ in zs:
                        rec = {k_[4:]: z_[k_] for k_ in z_.files if k_.startswith("rec_")}
                        ks = np.arange(Nc)
                        Qr, Vr = env.restore(rec, ks, ks < Ndc, foot_z=sim["foot_z"][gi])
                        refr = env.references(np.repeat([[g, 0.0]], Nc, 0))
                        Ar, Br = env.fd_at(Qr, Vr, ks, np.asarray(z_["act"], float)[:Nc], refr)
                        ok_ = np.isfinite(Ar).all((1, 2)) & np.isfinite(Br).all((1, 2))
                        Ar = np.where(ok_[:, None, None], Ar, sim["A"][gi])        # a failed sample: the sim's own
                        Br = np.where(ok_[:, None, None], Br, sim["B"][gi])
                        Anr = (Ar * D[None, None, :] / D[None, :, None])[None]
                        Bnr = (Br / D[None, :, None])[None]
                        Sl.append(landing_sensitivity(z_, w, 0, Anr, Bnr).reshape(3, -1))
                    S = np.mean(Sl, 0)
                else:
                    S = np.mean([landing_sensitivity(z_, w, gi, An, Bn).reshape(3, -1) for z_ in zs], 0)
                e3 = np.mean([np.asarray(z_["info"], float)[:3] for z_ in zs], 0)   # measured landing error (m, m, rad)
                if a.secant > 0:
                    if sec[g] is None:
                        sec[g] = np.zeros_like(S)
                    if prev[g] is not None and np.linalg.norm(prev[g][1]) > 1e-9:
                        da_, de_ = prev[g][1].ravel(), e3 - prev[g][0]
                        sec[g] += a.secant * np.outer(de_ - (S + sec[g]) @ da_, da_) / (da_ @ da_)
                    S = S + sec[g]
                M_ = S @ S.T
                stp = a.gn_beta * (S.T @ np.linalg.solve(M_ + a.gn_delta * np.trace(M_) / 3 * np.eye(3), e3))
                step = stp.reshape(Nc, 4) * scale[g]
                rms = np.sqrt((step ** 2).sum() / mask.sum())
                step *= min(1.0, a.step_rms / max(rms, 1e-12))
                if a.control == "random":
                    r_ = rng.standard_normal(step.shape) * mask
                    step = r_ * np.sqrt((step ** 2).sum() / max((r_ ** 2).sum(), 1e-12))
                for z_ in zs:
                    trials.append(dict(goal=g, it=it, obs=np.asarray(z_["obs"], float)[:Nc],
                                       tgt=np.clip((own(z_) - step) * mask, -0.999, 0.999)))
                prev[g] = (e3.copy(), -step.copy(), J)                     # the change commanded: -step
                line.append(f"g{g:.3f} J {J:.2f} e {e3[0] * 100:+.1f}/{e3[1] * 100:+.1f}cm {np.degrees(e3[2]):+.1f}deg "
                            f"(gn rms {min(rms, a.step_rms):.3f}, scale {scale[g]:.2f})")
                continue
            G = gradient(z, w, gi, An, Bn, g)
            d = G / max(np.linalg.norm(G), 1e-12)
            if a.momentum > 0 and mom[g] is not None:
                d = a.momentum * mom[g] + (1 - a.momentum) * d
                d = d / max(np.linalg.norm(d), 1e-12)
            mom[g] = d
            n_g = sum(1 for t in trials if t["goal"] == g) + 1
            size = a.step_rms * scale[g] / np.sqrt(n_g)
            step = d * size * np.sqrt(mask.sum())
            if a.control == "random":
                r_ = rng.standard_normal(step.shape) * mask
                step = r_ * np.linalg.norm(step) / max(np.linalg.norm(r_), 1e-12)
            trials.append(dict(goal=g, it=it, obs=obs[:Nc], tgt=np.clip((act - step) * mask, -0.999, 0.999)))
            line.append(f"g{g:.3f} J {J:.2f} (step {size:.3f})")
        say(f"it {it}: " + " | ".join(line) + f"  [{time.time() - t0:.0f} s]")
        if not trials:
            continue
        lam = a.anchor_w * a.anchor_n0 / (a.anchor_n0 + len(a.goals) * a.reps * (it + 1))
        ga = np.stack([np.linspace(g_lo, g_hi, 16), np.zeros(16)], 1)
        anc = env.rollout(w, env.references(ga), jax.random.PRNGKey(1000 + it), stochastic=False)
        Oa = jnp.asarray(anc["obs"][:, :Nc].reshape(-1, anc["obs"].shape[-1]), jnp.float32)
        Ma = jnp.asarray(np.tile(mask, (16, 1)), jnp.float32)
        Aa = actor_mean(w, Oa)
        To = jnp.asarray(np.concatenate([t["obs"] for t in trials]), jnp.float32)
        Tt = jnp.asarray(np.concatenate([t["tgt"] for t in trials]), jnp.float32)
        Tw = jnp.asarray(np.concatenate([np.full(Nc, a.age_decay ** (it - t["it"])) for t in trials]), jnp.float32)
        Tm = jnp.asarray(np.tile(mask, (len(trials), 1)), jnp.float32)

        def loss(w_):
            l_t = ((((actor_mean(w_, To) - Tt) * Tm) ** 2).sum(-1) * Tw).sum() / Tw.sum()
            l_a = (((actor_mean(w_, Oa) - Aa) * Ma) ** 2).sum(-1).mean()
            return l_t + lam * l_a, (l_t, l_a)
        w_prev = w
        w, (lt, la) = adam_fit(w, loss, a.steps)
        say(f"   update: {len(trials)} targets, fit {float(lt):.2e}, anchor {float(la):.2e} (weight {lam:.2f})")
        if GATE is not None:
            ok, msg = gate(w_prev, w, To, g_lo, g_hi, it)
            say(f"   gate: {msg}")
            if not ok:                                   # the current policy flies again, smaller steps
                w = w_prev
                for g in a.goals:
                    scale[g] *= 0.5
    res = dict(robot=robot, hist=hist, args=vars(a))
    if a.eval != "none":
        goals, seed = EVAL[a.eval]
        for tag, pd_ in (("start", os.path.join(out, "it0", "policy")), ("final", os.path.join(out, f"it{a.iters}", "policy"))):
            rr = fly(pd_, robot, goals, seed, os.path.join(out, f"eval_{a.eval}_{tag}.json"), episodes=a.eval_episodes)
            res[f"{a.eval}_{tag}"] = float(np.mean([score(r) for r in rr])) if rr else None
            res[f"{a.eval}_{tag}_falls"] = int(sum(r["fell"] for r in rr))
            if a.perturbed:
                rp = fly(pd_, f"/ilc_ws/log/dilc/final701/rob_{robot}.txt", goals, seed,
                         os.path.join(out, f"eval_{a.eval}_rob_{tag}.json"), episodes=a.eval_episodes)
                res[f"rob_{tag}"] = float(np.mean([score(r) for r in rp])) if rp else None
                res[f"rob_{tag}_falls"] = int(sum(r["fell"] for r in rp))
        fm = lambda v: "n/a" if v is None else f"{v:.2f}"
        say(f"{a.eval}: start {fm(res[f'{a.eval}_start'])} -> final {fm(res[f'{a.eval}_final'])} "
            f"(falls {res[f'{a.eval}_start_falls']} -> {res[f'{a.eval}_final_falls']})"
            + (f"; perturbed {fm(res['rob_start'])} -> {fm(res['rob_final'])}" if "rob_final" in res else ""))
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return res


allres = [run_robot(r) for r in a.robots]
if a.eval != "none":
    k = a.eval
    s = [r[f"{k}_start"] for r in allres]
    f = [r[f"{k}_final"] for r in allres]
    print(f"ALL {k}: start {np.mean(s):.2f} -> final {np.mean(f):.2f} ({' '.join(f'{x:.2f}' for x in f)})")
    json.dump(allres, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
