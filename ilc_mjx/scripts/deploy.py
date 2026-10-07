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
ap.add_argument("--goals", nargs="+", default=["0.425", "0.5", "0.575"],
                help="training goals: X (a flat jump, m) or X,H (onto a box H m high, X m ahead: a box bank)")
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
ap.add_argument("--est-window", type=float, default=0.0,
                help="s: the sim's policy observes the robots' estimator (as the network was trained: jump.Config)")
ap.add_argument("--clear", type=float, default=0.0,
                help="m (box banks, --update gn): the legs' and trunk's clearance to the box the ILC keeps -- a trial "
                     "whose least clearance (measured from its record, in single stance and in flight) is under it adds "
                     "a GN row pushing it back up, through the same closed-loop chain as the landing rows (0: none)")
ap.add_argument("--stage-w", type=float, nargs=3, default=[0.0, 0.0, 1.0])
ap.add_argument("--anchor-n0", type=float, default=6.0)
ap.add_argument("--anchor-w", type=float, default=1.0)
ap.add_argument("--age-decay", type=float, default=0.5)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--revisit", action="store_true",
                help="with --train-starts: every other iteration, each goal re-flies the start of its worst jump so far "
                     "(the hardest starts get repeated ILC steps, the rest stay random)")
ap.add_argument("--train-starts", type=float, default=0.0, metavar="S",
                help="container: every hardware jump starts from its own posture, the front and rear leg pairs bent by "
                     "s_f, s_r ~ U(-S, S) x (hip +0.1, knee -0.2) rad (+-1: the test's crouch / tall / nose-up / "
                     "nose-down); its ILC step linearizes the GPU sim flown from the same start. 0: the nominal stance")
ap.add_argument("--reg-jac", type=float, default=0.0,
                help="feedback anchor: weight on the change of the policy's feedback d pi / d (error states) at the anchor "
                     "and trial states (not decayed): the update moves the actions' offset, not the sim's feedback")
ap.add_argument("--anchor-pert", action="store_true",
                help="anchor states also from 16 perturbed stances (sigma_q 0.08), not only the 16 nominal ones")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=9001)
ap.add_argument("--eval", default="val", choices=("val", "final", "holdout", "boxval", "boxfinal", "comboval", "combofinal", "planeval", "planefinal",
                                       "none"))
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
ap.add_argument("--stall-sat", type=float, default=0.0, metavar="BIND",
                help="gn, with --stall: the stall guard freezes a goal only when motor saturation explains the stall -- "
                     "over its last --stall trials, the whole correction of the landing error (the Gauss-Newton step "
                     "at gain 1, uncapped) lost on average at least this share of its predicted effect to the motors' "
                     "limits (re-solved without the pushes that would ask a leg pair for more torque than it has left, "
                     "measured from the trial's torques); a goal that "
                     "stalls with its motors free keeps stepping. 0: any stall freezes (the blind guard). Without a "
                     "torque record in the jumps nothing counts as saturated")
ap.add_argument("--sat-project", action="store_true",
                help="gn: the step is re-solved without its pushes that would ask a leg pair for more torque than it has "
                     "left (the motors would clip them): the correction goes through the actuators and instants with "
                     "headroom")
ap.add_argument("--sat-thr", type=float, default=0.95,
                help="the log's saturation share: the push's leg-pair steps whose most loaded thigh or calf motor commands "
                     ">= this fraction of the Go1 datasheet torque-speed envelope at the measured joint speed")
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
ap.add_argument("--jac-dyn", type=float, nargs=8, default=None, metavar="P",
                help="the ILC's Jacobians from a fixed, deliberately different model (JumpEnv.DYN_KEYS: mass_scale com_x "
                     "motor_scale curve speed_scale friction joint_friction payload) -- how wrong may the model be?")
ap.add_argument("--jac-flip", action="store_true", help="gn: the landing sensitivity negated (the 180-degree control)")
ap.add_argument("--gpu", default="1")
a = ap.parse_args()
# goals are (x, h) pairs: h = 0 a flat jump, h > 0 onto a box h high (a box bank)
a.goals = [tuple(float(v) for v in (g.split(",") + ["0"])[:2]) for g in a.goals]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx import host_path  # noqa: E402
from ilc_mjx.jump import Config, JumpEnv  # noqa: E402
from ilc_mjx.robot import PLANAR_TO_CANONICAL  # noqa: E402

WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.abspath(p)[len(WS):]
EVAL = dict(val=([0.45, 0.4625, 0.5375, 0.55], 301), final=([0.4375, 0.4875, 0.5125, 0.5625], 701),
            holdout=([0.4375, 0.4625, 0.4875, 0.5125, 0.5375, 0.5625], 1301),   # holdout: the frozen final test only
            # box banks (plans (0.50, 0.10) (0.50, 0.15) (0.50, 0.20) (0.55, 0.10) (0.60, 0.10)): goals inside
            # their triangle, none of them a plan's or a training goal
            boxval=([(0.53, 0.11), (0.51, 0.13), (0.56, 0.12), (0.52, 0.16)], 301),
            boxfinal=([(0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17)], 701),
            # mixed flat + box banks: both reserved tests at once (8 goals)
            comboval=([0.45, 0.4625, 0.5375, 0.55, (0.53, 0.11), (0.51, 0.13), (0.56, 0.12), (0.52, 0.16)], 301),
            combofinal=([0.4375, 0.4875, 0.5125, 0.5625, (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17)], 701),
            # the 2D plane bank (log/dilc/plane: flat x 0.40-0.65, boxes up to 0.20 m, front face at x/2): goals across
            # it, off the plans and the sim stage's eval grid; planefinal keeps boxfinal's four (the JumpILC baselines')
            planeval=([0.4375, 0.5625, (0.53, 0.11), (0.51, 0.13), (0.56, 0.12), (0.52, 0.16), (0.4625, 0.06),
                       (0.6125, 0.07)], 301),
            planefinal=([0.4875, 0.6125, (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035),
                         (0.6375, 0.09)], 701))
EVAL = {k: ([tuple(g) if isinstance(g, tuple) else (float(g), 0.0) for g in v[0]], v[1]) for k, v in EVAL.items()}
gvec = lambda g: np.array(g, float)                                  # goal -> (x, h)
gkey = lambda v: (round(float(v[0]), 4), round(float(v[1]), 4))     # a goal or an episode's goal -> dict key
gname = lambda g: f"{g[0]:.3f}" + (f",{g[1]:.3f}" if g[1] else "")  # for the log
gfile = lambda g: f"{g[0]:.4f}" + (f"_{g[1]:.4f}" if g[1] else "")  # for file names (flat: as before)

bank = json.load(open(os.path.join(a.policy, "bank.json")))
env = JumpEnv(bank, Config(est_window=a.est_window))
Nc, Ndc = env.Nc, env.Ndc
gamma = float(bank.get("gamma", 0.99))
D = np.asarray(bank["sx"], float)
c = np.asarray(a.stage_w, float)
mask = np.ones((Nc, 4))
mask[Ndc:, :2] = 0.0
A_SCALE = np.asarray(bank["a_scale"], float)              # newtons per normalized action unit: front fx fz, rear fx fz
# the Go1's datasheet motor envelope (canonical joint order, hip thigh calf per leg): the torque limit up to half the
# no-load speed, then down linearly to none at it (while motoring) -- what the motors give, known on the real robot
GO1_TAU = np.tile([23.7, 23.7, 35.55], 4)
GO1_W = np.tile([30.1, 30.1, 20.06], 4)
PAIR_J = [PLANAR_TO_CANONICAL[0] + PLANAR_TO_CANONICAL[1], PLANAR_TO_CANONICAL[2] + PLANAR_TO_CANONICAL[3]]


def headroom(z):
    """(Nc, 2): each leg pair's torque headroom over each control step -- 1 - the commanded torque over the envelope
    at the measured joint speed, of its most loaded thigh or calf motor (the median over the step's motor ticks);
    1 in swing and without a torque record (an older episode)."""
    out = np.ones((Nc, 2))
    if "rec_tau_total" not in z.files:
        return out
    t, tau, dq = np.asarray(z["rec_t"], float), np.asarray(z["rec_tau_total"], float), np.asarray(z["rec_dq"], float)
    drop = np.clip((GO1_W - np.abs(dq)) / (0.5 * GO1_W), 0.0, 1.0)
    avail = np.where(tau * dq > 0, GO1_TAU * drop, GO1_TAU)
    ratio = np.abs(tau) / np.maximum(avail, 1e-3)
    k = np.floor(t / env.dt + 1e-6).astype(int)
    for kk in range(Nc):
        sel = k == kk
        if sel.any():
            for p in range(2):
                out[kk, p] = 1.0 - np.median(ratio[sel][:, PAIR_J[p]].max(1))
    out[mask[:, ::2] == 0] = 1.0
    return np.clip(out, 0.0, 1.0)


def gn_step(S, e3, off=None, beta=None):
    """-step of the Gauss-Newton ILC (flattened, normalized actions), the columns `off` held. S, e3: the landing's 3
    rows (and any clearance rows below them)."""
    if off is not None:
        S = np.where(off[None, :], 0.0, S)
    M_ = S @ S.T
    r = len(e3)
    return (a.gn_beta if beta is None else beta) * (S.T @ np.linalg.solve(M_ + a.gn_delta * np.trace(M_) / r * np.eye(r), e3))


def sat_step(S, e3, stp, H, U, beta=None):
    """The GN step (gain beta) re-solved without its pushes that would ask a leg pair for more torque than it has
    left: a force change along the pair's flown force, relative to it, above the pair's headroom H at that step (an
    active set, held entries at their flown value, a few passes); and the share of the step's predicted landing
    correction this loses (0: the motors' limits cost nothing)."""
    off = np.zeros(Nc * 4, bool)
    free = stp
    for _ in range(6):
        du = -free.reshape(Nc, 4) * A_SCALE
        viol = np.zeros((Nc, 4), bool)
        for p in range(2):
            up = U[:, 2 * p:2 * p + 2]
            rel = (du[:, 2 * p:2 * p + 2] * up).sum(1) / np.maximum((up ** 2).sum(1), 1.0)
            viol[:, 2 * p:2 * p + 2] = (rel > H[:, p])[:, None]
        new = off | (viol.ravel() & (mask.ravel() > 0))
        if (new == off).all():
            break
        off = new
        free = gn_step(S, e3, off, beta)
    lost = 1.0 - np.linalg.norm(S @ free) / max(np.linalg.norm(S @ stp), 1e-12)
    return free, float(np.clip(lost, 0.0, 1.0)), int(off.sum())


W0 = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(a.policy, a.member + ".npz")).items()
      if k != "n_hidden"}
actor_mean = jax.jit(jax.vmap(lambda w_, o: JumpEnv.actor(w_, o), (None, 0)))
# the policy's feedback: d action / d (SRB error, past errors) -- obs = error 6, time 1, goal 2, past errors 18, prev action 4
FB = np.r_[0:6, 9:27]
feedback = jax.vmap(lambda w_, o: jax.jacfwd(lambda o_: JumpEnv.actor(w_, o_))(o)[:, FB], (None, 0))
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
    j = " ".join(f"--jump {g[0]} {g[1]}" for g in jumps)
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
            subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad pkill -9 -f {CWS + tag}"], check=False)   # -9: rclpy keeps SIGTERM
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
        rf = srb_maker.bank.reference(gvec(g))[0]
        lab = srb_maker.ilc_labels(rf, np.asarray(z["U"], float), np.asarray(z["logX"], float), gvec(g), rf)
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


def clearance_rows(zs, w, gi, An, Bn, g):
    """Per trial at goal g, the GN rows keeping the legs' and trunk's clearance to the box at --clear: its least
    clearance (jump.clearance_rec, from the trial's record) in single stance and in flight; under --clear, a row
    d clearance / d a (its gradient w.r.t. the base pose through the closed-loop chain: the sim's Jacobians along
    the goal's jump, the policy's feedback at the trial's states) and the residual clearance - --clear (m)."""
    rows, res, cmins = [], [], []
    bx = np.asarray(env.references([gvec(g)])["box_xh"])[0]
    if bx[1] < 0.005:
        return rows, res, cmins
    N_ = env.N
    for z_ in zs:
        rec = {k_[4:]: z_[k_] for k_ in z_.files if k_.startswith("rec_")}
        if "pos" not in rec:
            continue
        C, Gr = env.clearance_rec(rec, bx, np.arange(N_))
        K = np.asarray(dpi_fn(jnp.asarray(np.asarray(z_["obs"], float)[:Nc], jnp.float32), w))[..., :6]
        Dn = np.zeros((Nc + 1, 6, Nc * 4))                      # d e_k / d a (normalized state), forward
        for k in range(Nc):
            Dn[k + 1] = (An[gi, k] + Bn[gi, k] @ K[k]) @ Dn[k]
            Dn[k + 1][:, 4 * k:4 * k + 4] += Bn[gi, k] * mask[k][None]
        # the flight window ends 6 samples before the touchdown: legs coming down onto the box top then are the
        # landing, not a clip (on the robot the touchdown's timing varies by a few samples)
        for lo_, hi_ in ((Ndc, Nc), (Nc, N_ - 6)):
            k = lo_ + int(np.argmin(C[lo_:hi_]))
            cmins.append(float(C[k]))
            if C[k] >= a.clear:
                continue
            Dr = D[:, None] * Dn[min(k, Nc)]                        # raw SRB units
            Dk = Dr[:3] + max(k - Nc, 0) * env.dt * Dr[3:]           # in flight: ballistic from takeoff
            rows.append(Gr[k] @ Dk)
            res.append(C[k] - a.clear)
    return rows, res, cmins


GATE = a.gate if a.gate is not None else ([0, 1.5, 0.3] if a.backend == "manual" else None)


def anchor_goals(n=16, rng_seed=0):
    """n goals spanning the training goals (flat: a line 2.5 cm past both ends; box goals: their hull, by fixed
    Dirichlet weights, the training goals themselves among them; both, a mixed bank: half on the flat goals' line,
    half over the box goals and the flat goals no shorter than the shortest box goal -- not the whole hull, whose
    short jumps onto a box would put it under the standing feet)."""
    G = np.array(a.goals, float)
    rng = np.random.default_rng(1234 + rng_seed)

    def line(F, m):
        return np.stack([np.linspace(F[:, 0].min() - 0.025, F[:, 0].max() + 0.025, m), np.full(m, F[0, 1])], 1)

    def hull(B, m):
        return np.concatenate([B, rng.dirichlet(np.ones(len(B)), max(m - len(B), 0)) @ B])[:m]
    if np.ptp(G[:, 1]) < 1e-9:
        return line(G, n)
    Gf, Gb = G[G[:, 1] < 1e-9], G[G[:, 1] >= 1e-9]
    if not len(Gf):
        return hull(Gb, n)
    Bx = np.concatenate([Gb, Gf[Gf[:, 0] >= Gb[:, 0].min() - 1e-9]])
    return np.concatenate([line(Gf, n // 2), hull(Bx, n - n // 2)])


def gate(w_old, w_new, O_real, g_lo, g_hi, it):
    """The safety gate (--gate): the update in the nominal GPU sim against the current policy, and how far it moves
    the actions on the robot's own states."""
    goals = anchor_goals()
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
    need = {gkey(g): a.reps for g in a.goals}
    lines = [f"# {robot}, iteration {it}: {len(a.goals) * a.reps} jumps", "",
             f"Policy: `{pdir}`. Save the episodes to `{rdir}` (episode_dir). For each goal, {a.reps} jump(s):", ""]
    for g in a.goals:
        ref = os.path.join(rdir, f"ref_{gfile(g)}.npz")
        bx = np.asarray(env.references([gvec(g)])["box_xh"])[0] if env.has_box else np.array([0.25, 0.0])
        box_args = f" box_x_front:={bx[0]:.4f} box_height:={bx[1]:.4f}" if bx[1] >= 0.005 else ""
        lines += [f"## goal {gname(g)} m" + (f" -- BOX: front face {bx[0]:.3f} m ahead of the standing CoM, "
                                            f"{bx[1]:.3f} m tall" if box_args else ""), "```",
                  f"python3 policy_jump_node.py prepare --policy {pdir} --goal {g[0]} {g[1]} --out {ref}",
                  f"ros2 launch ilc_quad policy_jump_go1.launch.py pose_topic:=<mocap topic> policy_dir:={pdir} \\",
                  f"    reference_file:={ref} jump_dx:={g[0]} jump_dz:={g[1]}{box_args} episode_dir:={rdir} episode_tag:={robot}_it{it}",
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
                    k_ = gkey(np.load(os.path.join(rdir, f))["goal"])
                    have[k_] = have.get(k_, 0) + 1
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
    # --train-starts: each iteration's start per goal, drawn up front (its own stream: the rest is unchanged)
    srng = np.random.default_rng(a.seed + 7)
    starts = []
    for _ in range(a.iters):
        sf = srng.uniform(-a.train_starts, a.train_starts, len(a.goals))
        sr = srng.uniform(-a.train_starts, a.train_starts, len(a.goals))
        starts.append({g: [0.1 * f, -0.2 * f, 0.1 * r, -0.2 * r] for g, f, r in zip(a.goals, sf, sr)})
    jstart = {g: [] for g in a.goals}            # (J, start) of every jump, per goal (--revisit)
    g_lo, g_hi = None, None                       # (the gate's goals: anchor_goals)
    trials, best, hist = [], {g: None for g in a.goals}, []
    scale = {g: 1.0 for g in a.goals}
    sec = {g: None for g in a.goals}         # Broyden correction of the landing sensitivity, per goal
    prev = {g: None for g in a.goals}        # (e3, commanded step, J) of the goal's previous trial
    frozen = {g: False for g in a.goals}
    Jh = {g: [] for g in a.goals}
    Bh = {g: [] for g in a.goals}            # saturation bind of each GN step, per goal (--stall-sat)
    Ch = {g: [] for g in a.goals}            # per iteration the goal's least clearance to the box (--clear)
    Fh = {g: [] for g in a.goals}            # per iteration whether the goal's trial fell
    mom ={g: None for g in a.goals}
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
            # a flight that hangs returns no jumps: fly the missing goals again (same seeds, so the same jumps), and
            # abort rather than update without them (a run with skipped jumps is not the method)
            if a.train_starts > 0 and a.revisit and it % 2 == 1:
                for g in a.goals:                       # the worst start so far, again
                    if jstart[g]:
                        starts[it][g] = max(jstart[g], key=lambda t: t[0])[1]
            todo = list(a.goals)
            for attempt in range(4):
                if a.train_starts > 0:                  # each goal's jump from its own start
                    for g in todo:
                        cf = os.path.join(out, f"it{it}", f"start_{gfile(g)}.txt")
                        o = starts[it][g]
                        open(cf, "w").write(f"{robot}+start|={robot} --param stand_offset:=["
                                            + ",".join(f"{v:.4f}" for v in o) + "]\n")
                        fly(pdir, cpath(cf), [g], a.seed + 101 * it, os.path.join(out, f"it{it}", f"real_{gfile(g)}.json"),
                            rdir, episodes=a.reps)
                else:
                    fly(pdir, robot, todo, a.seed + 101 * it, os.path.join(out, f"it{it}", "real.json"), rdir,
                        episodes=a.reps)
                have = {}
                for f in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
                    if f.endswith(".npz") and not f.startswith("ref_"):
                        g_ = gkey(np.load(os.path.join(rdir, f))["goal"])
                        have[g_] = have.get(g_, 0) + 1
                todo = [g for g in a.goals if have.get(gkey(g), 0) < a.reps]
                if not todo:
                    break
                say(f"   no jumps yet for goals {todo} (attempt {attempt + 1}); flying them again")
            if todo:
                raise SystemExit(f"{robot} it {it}: no jumps for goals {todo} after 4 attempts; aborting the run")
        eps = {}
        for f in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
            if not f.endswith(".npz") or f.startswith("ref_"):
                continue
            z = np.load(os.path.join(rdir, f))
            eps.setdefault(gkey(z["goal"]), []).append(z)
        An = Bn = None
        if a.grad == "fd" or a.update == "gn":
            goals = np.array([gvec(g) for g in a.goals])
            q_off = None
            if a.train_starts > 0:                      # the GPU sim flown from each jump's own start
                q_off = np.zeros((len(a.goals), 12))
                for gi_, g_ in enumerate(a.goals):
                    for i_, idx in enumerate(PLANAR_TO_CANONICAL):
                        q_off[gi_, idx] = starts[it][g_][i_]
            sim = env.rollout(w, env.references(goals), jax.random.PRNGKey(it), stochastic=False, fd=True,
                              q_offset=q_off,
                              dyn=None if a.jac_dyn is None else env.dyn_arrays(np.tile(a.jac_dyn, (len(goals), 1))))
            An = sim["A"] * D[None, None, None, :] / D[None, None, :, None]
            Bn = sim["B"] / D[None, None, :, None]
        line = []
        for gi, g in enumerate(a.goals):
            zs = eps.get(gkey(g))
            if not zs:
                line.append(f"g{gname(g)} missing")
                continue
            z = zs[0]
            own = lambda z_: (np.asarray(actor_mean(w, jnp.asarray(np.asarray(z_["obs"], float)[:Nc], jnp.float32))) * mask
                              if a.target_base == "own" else np.asarray(z_["act"], float))
            obs, fell, act = np.asarray(z["obs"], float), any(bool(z_["fell"]) for z_ in zs), own(z)
            Js = [float(0.01 * (np.asarray(z_["rc"]) @ c * gamma ** np.arange(Nc)).sum()) for z_ in zs]
            J = float(np.mean(Js))
            hist.extend(dict(it=it, goal=g, J=J_, fell=bool(z_["fell"])) for J_, z_ in zip(Js, zs))
            Jh[g].append(J)
            Fh[g].append(bool(fell))
            if a.train_starts > 0:
                jstart[g].append((J, list(starts[it][g])))
            stalled = a.stall > 0 and not frozen[g] and len(Jh[g]) > a.stall \
                and min(Jh[g][-a.stall:]) > 0.9 * Jh[g][-a.stall - 1]
            sat_note = ""
            if stalled and a.stall_sat > 0:                        # freeze only if the motors explain the stall
                bw = Bh[g][-a.stall:]
                sat_bound = len(bw) == a.stall and float(np.mean(bw)) >= a.stall_sat
                sat_note = f" (stalled, saturation bind {np.mean(bw) if bw else 0:.2f}" \
                           + (": saturation-bound)" if sat_bound else ": motors free, steps on)")
                # or the box explains it: a stalled goal that clipped (clearance < 0) or fell on most of its last
                # --stall iterations is out of this robot's reach as surely as a motor-bound one -- stepping it on
                # only drags the shared network (heavy real_r5's box goals)
                cw, fw = Ch[g][-a.stall:], Fh[g][-a.stall:]
                geo_bound = (len(cw) == a.stall and sum(c < 0 for c in cw) >= a.stall - 1) or \
                    (len(fw) == a.stall and sum(fw) >= a.stall - 1)
                if geo_bound and not sat_bound:
                    sat_note = f" (stalled, clipping / falling on {max(sum(c < 0 for c in cw), sum(fw))} of the last " \
                               f"{a.stall}: box-bound)"
                stalled = sat_bound or geo_bound
            if stalled:
                frozen[g] = True                                   # no progress: stop stepping this goal
                trials[:] = [t for t in trials if t["goal"] != g]
            if frozen[g]:
                if best[g] is not None and not any(t["goal"] == g for t in trials):
                    trials.append(dict(goal=g, it=it, obs=best[g][1], tgt=best[g][2]))
                for t in trials:                                   # the hold keeps full weight
                    if t["goal"] == g:
                        t["it"] = it
                line.append(f"g{gname(g)} J {J:.2f} FROZEN (held at its best){sat_note}")
                continue
            runaway = bool(a.rollback) and best[g] is not None and J > a.rollback[0] * best[g][0] \
                and J > best[g][0] + a.rollback[1]
            if fell and a.update == "gn" and a.clear > 0 and env.has_box and An is not None:
                # a fall with the legs or trunk too close to the box (a clip): the step on its clearance rows and its
                # predicted landing (the measured one is the clip's), from the trial's own actions, the scale kept --
                # a rollback would only repeat the clip with ever smaller steps
                crow, cres, cmins = clearance_rows(zs, w, gi, An, Bn, g)
                if cmins:
                    Ch[g].append(min(cmins))
                if crow:
                    # with the measured touchdown error too (the fall comes after the touchdown; the clearance alone
                    # would push a long jump higher, i.e. longer still). Not the ballistic prediction from the takeoff
                    # state: on the robot that state is the estimator's, its velocity off by up to ~0.7 m/s at the
                    # push's end -- 20-35 cm over the flight
                    eb = np.mean([np.asarray(z_["info"], float)[:2] for z_ in zs], 0)
                    Sl = np.mean([landing_sensitivity(z_, w, gi, An, Bn).reshape(3, -1) for z_ in zs], 0)[:2]
                    Sc, rc = np.vstack([Sl, np.array(crow)]), np.concatenate([eb, np.array(cres)])
                    step = gn_step(Sc, rc).reshape(Nc, 4)
                    rms = np.sqrt((step ** 2).sum() / mask.sum())
                    step *= min(1.0, a.step_rms / max(rms, 1e-12))
                    for z_ in zs:
                        trials.append(dict(goal=g, it=it, obs=np.asarray(z_["obs"], float)[:Nc],
                                           tgt=np.clip((own(z_) - step) * mask, -0.999, 0.999)))
                    line.append(f"g{gname(g)} FELL, clr {100 * min(cmins):.1f}cm, landing {100 * eb[0]:+.1f}/"
                                f"{100 * eb[1]:+.1f}cm: step on both (rms {min(rms, a.step_rms):.3f}, {len(crow)} clr rows)")
                    continue
            if fell or runaway:                                    # back to the best trial, half the steps
                scale[g] *= 0.5
                if best[g] is not None:
                    trials.append(dict(goal=g, it=it, obs=best[g][1], tgt=best[g][2]))
                line.append(f"g{gname(g)} {'FELL' if fell else f'J {J:.2f} RUNAWAY'} (rollback, scale {scale[g]:.2f})")
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
                        refr = env.references(np.repeat([gvec(g)], Nc, 0))
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
                if a.jac_flip:
                    S = -S
                e3_land = e3
                clr_txt = ""
                if a.clear > 0 and env.has_box:
                    crow, cres, cmins = clearance_rows(zs, w, gi, An, Bn, g)
                    if cmins:
                        Ch[g].append(min(cmins))
                    if crow:
                        S, e3 = np.vstack([S, np.array(crow)]), np.concatenate([e3, np.array(cres)])
                    clr_txt = f", clr {100 * min(cmins):.1f}cm rows {len(crow)}" if cmins else ""
                stp = gn_step(S, e3)
                sat_txt = ""
                if a.sat_project or a.stall_sat > 0:
                    Hm = np.min([headroom(z_) for z_ in zs], 0)
                    Uf = np.mean([np.asarray(z_["U"], float)[:Nc] for z_ in zs], 0)
                    free, _, n_off = sat_step(S, e3, stp, Hm, Uf)
                    # the bind: of the WHOLE correction (gain 1, uncapped) -- can the motors remove this error at all?
                    _, bind, _ = sat_step(S, e3, gn_step(S, e3, beta=1.0), Hm, Uf, beta=1.0)
                    satf = float(((Hm < 1 - a.sat_thr) & (mask[:, ::2] > 0)).sum() / mask[:, ::2].sum())
                    Bh[g].append(bind)
                    for h_ in hist[-len(zs):]:
                        h_.update(sat=satf, bind=bind)
                    if a.sat_project:
                        stp = free
                    sat_txt = f", sat {100 * satf:.0f}% bind {bind:.2f} held {n_off}"
                step = stp.reshape(Nc, 4) * scale[g]
                rms = np.sqrt((step ** 2).sum() / mask.sum())
                step *= min(1.0, a.step_rms / max(rms, 1e-12))
                if a.control == "random":
                    r_ = rng.standard_normal(step.shape) * mask
                    step = r_ * np.sqrt((step ** 2).sum() / max((r_ ** 2).sum(), 1e-12))
                for z_ in zs:
                    trials.append(dict(goal=g, it=it, obs=np.asarray(z_["obs"], float)[:Nc],
                                       tgt=np.clip((own(z_) - step) * mask, -0.999, 0.999)))
                e3 = e3_land
                prev[g] = (e3.copy(), -step.copy(), J)                     # the change commanded: -step
                line.append(f"g{gname(g)} J {J:.2f} e {e3[0] * 100:+.1f}/{e3[1] * 100:+.1f}cm {np.degrees(e3[2]):+.1f}deg "
                            f"(gn rms {min(rms, a.step_rms):.3f}, scale {scale[g]:.2f}{sat_txt}{clr_txt}){sat_note}")
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
            line.append(f"g{gname(g)} J {J:.2f} (step {size:.3f})")
        say(f"it {it}: " + " | ".join(line) + f"  [{time.time() - t0:.0f} s]")
        if not trials:
            continue
        lam = a.anchor_w * a.anchor_n0 / (a.anchor_n0 + len(a.goals) * a.reps * (it + 1))
        ga = anchor_goals(rng_seed=it)
        if a.anchor_pert:                                # the tube: nominal and perturbed stances
            ga = np.concatenate([ga, ga])
            q_a, _ = env.explore_noise(32, np.random.default_rng(5000 + it), 0.08, 0.0)
            q_a[:16] = 0.0
            anc = env.rollout(w, env.references(ga), jax.random.PRNGKey(1000 + it), stochastic=False, q_offset=q_a)
        else:
            anc = env.rollout(w, env.references(ga), jax.random.PRNGKey(1000 + it), stochastic=False)
        Oa = jnp.asarray(anc["obs"][:, :Nc].reshape(-1, anc["obs"].shape[-1]), jnp.float32)
        Ma = jnp.asarray(np.tile(mask, (len(ga), 1)), jnp.float32)
        Aa = actor_mean(w, Oa)
        To = jnp.asarray(np.concatenate([t["obs"] for t in trials]), jnp.float32)
        Tt = jnp.asarray(np.concatenate([t["tgt"] for t in trials]), jnp.float32)
        Tw = jnp.asarray(np.concatenate([np.full(Nc, a.age_decay ** (it - t["it"])) for t in trials]), jnp.float32)
        Tm = jnp.asarray(np.tile(mask, (len(trials), 1)), jnp.float32)

        if a.reg_jac > 0:
            Of = jnp.concatenate([Oa, To])
            Mf = jnp.concatenate([Ma, Tm])
            Kf = feedback(w, Of)                         # the current feedback, held

        def loss(w_):
            l_t = ((((actor_mean(w_, To) - Tt) * Tm) ** 2).sum(-1) * Tw).sum() / Tw.sum()
            l_a = (((actor_mean(w_, Oa) - Aa) * Ma) ** 2).sum(-1).mean()
            if a.reg_jac > 0:
                l_a = l_a + a.reg_jac / lam * ((((feedback(w_, Of) - Kf) * Mf[..., None]) ** 2).sum((1, 2)).mean())
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
            for _ in range(3):                           # an evaluation that hung: fly it again
                rr = fly(pd_, robot, goals, seed, os.path.join(out, f"eval_{a.eval}_{tag}.json"),
                         episodes=a.eval_episodes)
                if rr:
                    break
            res[f"{a.eval}_{tag}"] = float(np.mean([score(r) for r in rr])) if rr else None
            res[f"{a.eval}_{tag}_falls"] = int(sum(r["fell"] for r in rr))
            if a.perturbed:
                for _ in range(3):
                    rp = fly(pd_, f"/ilc_ws/log/dilc/final701/rob_{robot}.txt", goals, seed,
                             os.path.join(out, f"eval_{a.eval}_rob_{tag}.json"), episodes=a.eval_episodes)
                    if rp:
                        break
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
    s = [np.nan if r[f"{k}_start"] is None else r[f"{k}_start"] for r in allres]     # None: an evaluation that hung
    f = [np.nan if r[f"{k}_final"] is None else r[f"{k}_final"] for r in allres]
    print(f"ALL {k}: start {np.nanmean(s):.2f} -> final {np.nanmean(f):.2f} ({' '.join(f'{x:.2f}' for x in f)})")
    json.dump(allres, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
