#!/usr/bin/env python3
"""ilc2real_figs.py: the ILC2Real figure set (2D goal plane, Go1 jumping, sim-to-sim on the CPU "real" robots).
Every number is read from the eval files (nothing hard-coded). Success = no fall, |ex| <= 5 cm, |ez| <= 3 cm on the
reserved test goals (planefinal) unless stated; intervals are 95% Wilson intervals over jumps.

Perturbations are separated by kind:
  goal generalization      test goals the hardware stage never flies (and the 2D maps over all 76 valid goals)
  dynamics                 each CPU robot's physics (motor scale, torque-speed curve, mass, CoM, delays, mocap), and
                           the GPU-sim sweeps of one dynamics parameter at a time
  start state              the stance perturbed: a block under the front feet (1.5 / 2 cm), crouched, tall, nose up,
                           nose down (the 'rob' evals, every robot)
  sensing / actuation      bad mocap (120 Hz, 15 ms), 10 ms actuation delay

-> src/ilc_mjx/figures/ilc2real/NN_name.png. Run: ~/miniconda3/envs/ilcmjx/bin/python ilc2real_figs.py"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # ilc_ws/src
_LOG = os.path.normpath(os.path.join(_REPO, '..', 'log', 'dilc'))                                    # run records (not in git)

P = os.path.join(_LOG, "plane")
OUT = os.path.join(_REPO, "ilc_mjx", "figures", "ilc2real")
os.makedirs(OUT, exist_ok=True)
# reference palette (dataviz skill, light surface): categorical slots in fixed order; sequential blue; diverging
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARK = ["o", "s", "D", "^", "v", "P", "X", "*"]
HATCH = ["", "//", "..", "xx", "\\\\", "++", "oo", "--"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#d95926", "#ec835a", "#f0efec", "#5598e7", "#1c5cab"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.titlesize": 10, "axes.titlelocation": "left",
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
ROBOTS = ["real_r1", "real_s1", "real_r4", "real_r5"]
PERT_START = ["blk15", "blk2", "crouch", "tall", "noseup", "nosedn"]
PERT_SENSE = ["mocapbad", "delay10"]
PERT_NAME = dict(blk15="block 1.5 cm", blk2="block 2 cm", crouch="crouched", tall="tall", noseup="nose up",
                 nosedn="nose down", mocapbad="bad mocap", delay10="10 ms delay")
# method -> (run, nominal tag, perturbed tag); the zero-shot rows read 'start', adapted rows 'final'
METHODS = [("Ours, zero-shot", "hw6g16_v10s2it20", "start"), ("Ours, 24 real jumps", "gate_loose", "final"),
           ("Ours, 96 real jumps", "hw6g16_v10s2it20", "final"), ("Our learner + DR (A), zero-shot", "base/eval_dr_plain", "start"),
           ("PPO + DR", "base/eval_ppo_plain", "start"), ("RMA", "base/eval_rma", "start"),
           ("RMA, cross-trial", "base/eval_rma_ctx", "final"), ("FADA (adapted)", "base/eval_fada", "final"),
           ("Our learner + DR (A), 24 real jumps", "drA_hw24", "final"),
           ("Our learner + DR (B), zero-shot", "drB_hw24", "start"),
           ("Our learner + DR (B), 24 real jumps", "drB_hw24", "final")]
# (categorical slot, hatch) per method: the 8 hues in fixed order; the DR family (our learner + DR, fine-tuned A /
# from scratch B, zero-shot / after our hardware stage) shares slot 3 and is told apart by hatching (no 9th hue)
STYLE = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (3, 1), (3, 2), (3, 3)]
ok = lambda r: (not r["fell"]) and abs(r["ex"]) <= 0.05 and abs(r["ez"]) <= 0.03


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def load(run, tag, rob=False, robots=ROBOTS):
    fn = f"eval_planefinal_{'rob_' if rob else ''}{tag}.json"
    return [dict(r, robot=rb) for rb in robots for f in glob.glob(os.path.join(P, run, rb, fn)) for r in json.load(open(f))]


def kind_of(r):
    c = r.get("cond", "")
    if "+" not in c:
        return "nominal"
    p = c.split("+", 1)[1]
    return "start" if p in PERT_START else "sense" if p in PERT_SENSE else "other"


def style(ax, grid_y=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def available():
    out = []
    for i, (lab, run, tag) in enumerate(METHODS):
        nom = load(run, tag)
        if nom:
            out.append((i, lab, run, tag, nom, load(run, tag, rob=True)))
    return out


# -- 1. methods by perturbation kind ------------------------------------------------------------------------------
def fig_methods():
    ms = available()
    kinds = [("nominal start", lambda m: m[4]), ("starting condition (6 kinds)", lambda m: [r for r in m[5] if kind_of(r) == "start"]),
             ("environment: sensing / actuation (2 kinds)", lambda m: [r for r in m[5] if kind_of(r) == "sense"])]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.2), sharey="row", gridspec_kw=dict(height_ratios=[3, 1.4]))
    for j, (title, sel) in enumerate(kinds):
        for row, metric in ((0, "ok"), (1, "fell")):
            ax = axs[row, j]
            for k, m in enumerate(ms):
                rs = sel(m)
                if not rs:
                    continue
                cnt = sum(ok(r) for r in rs) if metric == "ok" else sum(bool(r["fell"]) for r in rs)
                p, lo, hi = wilson(cnt, len(rs))
                ax.bar(k, 100 * p, 0.78, color=CAT[STYLE[m[0]][0]], hatch=HATCH[STYLE[m[0]][1]], edgecolor=SURF, lw=0.6, zorder=2)
                ax.errorbar(k, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color=INK2, lw=1, capsize=2, zorder=3)
                if row == 0:
                    ax.text(k, 100 * hi + 1.5, f"{100 * p:.0f}", ha="center", fontsize=7.5, color=INK2)
            ax.set_xticks([])
            style(ax)
            if row == 0:
                ax.set_title(title)
        axs[0, j].set_ylim(0, 75)
        axs[1, j].set_ylim(0, 40)
    axs[0, 0].set_ylabel("success (%)")
    axs[1, 0].set_ylabel("falls (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT[STYLE[m[0]][0]], hatch=HATCH[STYLE[m[0]][1]], ec=SURF) for m in ms]
    fig.legend(handles, [m[1] for m in ms], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Transfer to the 4 CPU robots by perturbation kind (reserved test goals; 95% intervals)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "01_methods_by_perturbation_kind.png")


# -- 2. per perturbation heatmap -------------------------------------------------------------------------------
def fig_per_pert():
    ms = available()
    cols = ["nominal"] + PERT_START + PERT_SENSE
    M = np.full((len(ms), len(cols)), np.nan)
    for i, m in enumerate(ms):
        for j, c in enumerate(cols):
            rs = m[4] if c == "nominal" else [r for r in m[5] if r.get("cond", "").endswith("+" + c)]
            if rs:
                M[i, j] = np.mean([ok(r) for r in rs])
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(ms) + 1.8))
    im = ax.imshow(100 * M, cmap=SEQ, vmin=0, vmax=60, aspect="auto")
    for i in range(len(ms)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color=INK if M[i, j] < 0.35 else "#ffffff")
    ax.set_xticks(range(len(cols)), ["nominal"] + [PERT_NAME[c] for c in cols[1:]], rotation=0, fontsize=8)
    ax.set_yticks(range(len(ms)), [m[1] for m in ms])
    for x in (0.5, 6.5):
        ax.axvline(x, color=SURF, lw=3)
    yb = len(ms) - 0.5 + 0.85
    ax.text(3.5, yb, "starting condition", ha="center", va="top", color=INK2, fontsize=9)
    ax.text(7.5, yb, "environment: sensing / actuation", ha="center", va="top", color=INK2, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("success (%)")
    ax.set_title("Success per perturbation (%; 4 robots x 8 test goals x 4 episodes per cell)")
    save(fig, "02_success_per_perturbation.png")


# -- 3. dynamics: per robot ----------------------------------------------------------------------------------------
ROBOT_DESC = {"real_r1": "real_r1\nmotors 0.93, curve\nmass 0.97, 8 ms",
              "real_s1": "real_s1\nmotors 1.0, curve\nmass 1.05, j-fric .2",
              "real_r4": "real_r4\nmotors 0.86, 8 ms\nBAD mocap",
              "real_r4m": "real_r4m\n= r4, good mocap",
              "real_r5": "real_r5\nmotors 0.94, curve\nmass 1.14"}


def fig_robots():
    rows = [("Ours, zero-shot", lambda rb: load("hw6g16_v10s2it20", "start", robots=[rb]) or load("base6_r4m", "start", robots=[rb])),
            ("Ours, 24 real jumps", lambda rb: load("gate_loose", "final", robots=[rb]) or load("base6_r4m", "final", robots=[rb])),
            ("Ours, 96 real jumps", lambda rb: load("hw6g16_v10s2it20", "final", robots=[rb])),
            ("Our learner + DR", lambda rb: load("base/eval_dr_plain", "start", robots=[rb]))]
    rbs = ["real_r1", "real_s1", "real_r4", "real_r4m", "real_r5"]
    cidx = [0, 1, 2, 3]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    w = 0.2
    for k, (lab, f) in enumerate(rows):
        for i, rb in enumerate(rbs):
            rs = f(rb)
            if not rs:
                continue
            p, lo, hi = wilson(sum(ok(r) for r in rs), len(rs))
            x = i + (k - 1.5) * w
            ax.bar(x, 100 * p, w * 0.92, color=CAT[cidx[k]], hatch=HATCH[cidx[k]], edgecolor=SURF, lw=0.6,
                   label=lab if i == 0 else None, zorder=2)
            ax.errorbar(x, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color=INK2, lw=0.9, capsize=1.5, zorder=3)
            ax.text(x, 100 * p + 1, f"{100 * p:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2, zorder=4)
    ax.set_xticks(range(len(rbs)), [ROBOT_DESC[r] for r in rbs], fontsize=7.5)
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 85)
    style(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    ax.set_title("Dynamics: each robot (nominal starts, test goals; 95% intervals)")
    # signed landing bias per robot: box goals, upright
    ax = axs[1]
    for k, (lab, f) in enumerate(rows[:3]):
        vals = []
        for rb in rbs:
            rs = [r for r in f(rb) if not r["fell"] and r["goal"][1] >= 0.005]
            vals.append(100 * np.mean([r["ex"] for r in rs]) if rs else np.nan)
        ax.plot(range(len(rbs)), vals, color=CAT[cidx[k]], marker=MARK[cidx[k]], lw=2, ms=6, mec=SURF, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhspan(-5, 5, color=GRID, alpha=0.6, zorder=0)
    ax.text(len(rbs) - 1, 5.6, "+-5 cm tolerance", ha="right", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(rbs)), [r.replace("real_", "") for r in rbs])
    ax.set_ylabel("mean landing x error, box goals (cm)")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Dynamics: the short-landing bias per robot")
    fig.tight_layout()
    save(fig, "03_dynamics_per_robot.png")


# -- 4. dynamics: one-parameter sweeps (GPU sim) -------------------------------------------------------------------
ROBOT_DYN = {"mass_scale": {"r1": 0.969, "r4": 0.982, "r5": 1.14, "s1": 1.05}, "com_x": {"r1": -0.0109, "r5": 0.0155, "s1": 0.01},
             "motor_scale": {"r1": 0.929, "r4": 0.864, "r5": 0.943}, "friction": {"r1": 0.63, "r5": 0.59},
             "joint_friction": {"r1": 0.11, "r4": 0.02, "r5": 0.08, "s1": 0.2}, "payload": {"r1": 0.42}}
SWEEP_LABEL = {"mass_scale": "mass scale", "com_x": "CoM x offset (m)", "motor_scale": "motor strength scale",
               "friction": "ground friction", "joint_friction": "joint friction (N m)", "payload": "payload (kg)"}


def fig_dyn_sweeps():
    f = os.path.join(P, "figdata", "dyn_sweep.json")
    if not os.path.exists(f):
        print("skip dyn sweeps (no data yet)")
        return
    d = json.load(open(f))
    names = list(SWEEP_LABEL)
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 6.4), sharey=True)
    for a, name in zip(axs.ravel(), names):
        for k, (lab, sw) in enumerate(d["sweeps"].items()):
            pts = sw[name]
            ci = {0: 0, 1: 3, 2: 4}[k]
            a.plot([q["v"] for q in pts], [100 * q["ok"] for q in pts], color=CAT[ci], marker=MARK[ci], lw=2, ms=5, mec=SURF, label=lab)
        for rb, v in ROBOT_DYN.get(name, {}).items():
            a.axvline(v, color=MUTED, lw=0.8, ls=":")
            a.text(v, 101, rb, ha="center", fontsize=7, color=INK2)
        nomv = {"mass_scale": 1.0, "com_x": 0.0, "motor_scale": 1.0, "friction": 1.0, "joint_friction": 0.0, "payload": 0.0}[name]
        a.axvline(nomv, color=INK2, lw=1)
        a.set_xlabel(SWEEP_LABEL[name])
        a.set_ylim(0, 108)
        style(a)
    for a in axs[:, 0]:
        a.set_ylabel("success over the 76-goal plane (%)")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Dynamics perturbations one at a time (GPU sim, nominal starts; solid line = the nominal model, "
                 "dotted = the CPU robots' values)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "04_dynamics_sweeps_sim.png")


# -- 5. goal plane maps (2D) ---------------------------------------------------------------------------------------
def fig_goal_maps():
    gfiles = {s: sorted(glob.glob(os.path.join(P, "figdata", f"grid_{s}_real_*.json")))
              for s in ("zeroshot", "after24")}
    dyn = os.path.join(P, "figdata", "dyn_sweep.json")
    if not (gfiles["zeroshot"] and os.path.exists(dyn)):
        print("skip goal maps (no data yet)")
        return
    d = json.load(open(dyn))
    G = np.array(d["goals"])
    key = lambda g: (round(g[0], 3), round(g[1], 3))
    simok = [q for q in d["sweeps"]["ours (sim, zero-shot)"]["com_x"] if abs(q["v"]) < 1e-9][0]["ok_goal"]

    def agg(files, robots=("real_r1", "real_s1", "real_r4m", "real_r5")):
        acc = {}
        for f in files:
            if not any(f.endswith(f"_{rb}.json") for rb in robots):
                continue
            for r in json.load(open(f)):
                a = acc.setdefault(key(r["goal"]), [])
                a.append(r)
        okm = np.array([np.mean([ok(r) for r in acc.get(key(g), [])]) if acc.get(key(g)) else np.nan for g in G])
        exm = np.array([np.mean([r["ex"] for r in acc.get(key(g), []) if not r["fell"]]) if any(not r["fell"] for r in acc.get(key(g), [])) else np.nan for g in G])
        return okm, exm
    z_ok, z_ex = agg(gfiles["zeroshot"])
    a_ok, a_ex = agg(gfiles["after24"]) if gfiles["after24"] else (np.full(len(G), np.nan),) * 2
    train = np.array([(0.45, 0), (0.60, 0), (0.50, 0.15), (0.60, 0.10), (0.50, 0.05), (0.575, 0.15)])
    test = np.array([(0.4875, 0), (0.6125, 0), (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035), (0.6375, 0.09)])
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=True)
    panels = [(axs[0, 0], np.array(simok, float), "nominal GPU sim (zero-shot)", "ok"), (axs[0, 1], z_ok, "4 CPU robots, zero-shot", "ok"),
              (axs[0, 2], a_ok, "4 CPU robots, after 24 real jumps", "ok"),
              (axs[1, 1], z_ex, "landing x error, zero-shot", "ex"), (axs[1, 2], a_ex, "landing x error, after 24 jumps", "ex"),
              (axs[1, 0], a_ok - z_ok, "success change, after 24 - zero-shot", "diff")]
    for ax, v, title, kind in panels:
        if kind == "ok":
            sc = ax.scatter(G[:, 0], G[:, 1], c=100 * v, cmap=SEQ, vmin=0, vmax=100, s=150, marker="s", edgecolors=SURF, lw=0.5)
            lab = "success (%)"
        elif kind == "ex":
            sc = ax.scatter(G[:, 0], G[:, 1], c=100 * v, cmap=DIV, norm=TwoSlopeNorm(0, -15, 15), s=150, marker="s", edgecolors=SURF, lw=0.5)
            lab = "mean x error (cm), - = short"
        else:
            sc = ax.scatter(G[:, 0], G[:, 1], c=100 * v, cmap=DIV, norm=TwoSlopeNorm(0, -60, 60), s=150, marker="s", edgecolors=SURF, lw=0.5)
            lab = "change (points)"
        ax.scatter(train[:, 0], train[:, 1], marker="x", s=55, c=INK, lw=1.5, label="hardware-stage training goals", zorder=3)
        ax.scatter(test[:, 0], test[:, 1], marker="o", s=70, facecolors="none", edgecolors=INK, lw=1.2, label="reserved test goals", zorder=3)
        ax.set_title(title)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(lab, fontsize=8)
        style(ax, grid_y=False)
    for ax in axs[1]:
        ax.set_xlabel("goal distance x (m)")
    for ax in axs[:, 0]:
        ax.set_ylabel("box height h (m)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h[:2], l[:2], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Goal generalization over the 2D goal plane (76 valid goals; robots r1, s1, r4m, r5, 2 episodes per goal)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "05_goal_plane_maps.png")


# -- 6. real-jump budget --------------------------------------------------------------------------------------------
def _jilc_land(z, base):
    gx, gh = float(base.split("_g")[1].split("_")[0]), float(base.split("_g")[1].split("_")[1])
    xr, lx = np.asarray(z["x_ref"], float), np.asarray(z["log_X"], float)
    target = xr[-1].copy()
    target[:2] = xr[0, :2] + np.array([gx, gh])
    target[2] = 0.0
    e = lx[-1] - target
    return dict(robot="real_" + base.split("_real_")[1].split("_s")[0], ex=float(e[0]), ez=float(e[1]), fell=bool(z["log_fell"]))


def jilc_at(k, robots=ROBOTS):
    """per-goal ILC (jilc_v3: scratch on each test goal): what it flies after k trials (trial k+1, or its last)."""
    rs = []
    for d in sorted(glob.glob(os.path.join(P, "jilc_v3", "scratch_g*"))):
        tr = sorted(glob.glob(os.path.join(d, "trial_*.npz")))
        if tr:
            r = _jilc_land(np.load(tr[min(k + 1, len(tr)) - 1]), os.path.basename(d))
            if r["robot"] in robots:
                rs.append(r)
    return rs


def jilc_eval(k, robots=ROBOTS):
    """the per-goal input after k trials, re-flown with our evaluation protocol (jilc_eval/b<k>)."""
    rs = []
    for d in sorted(glob.glob(os.path.join(P, "jilc_eval", f"b{k}", "retarget_g*"))):
        tr = sorted(glob.glob(os.path.join(d, "trial_*.npz")))
        if tr:
            r = _jilc_land(np.load(tr[0]), os.path.basename(d))
            if r["robot"] in robots:
                rs.append(r)
    return rs



def fig_budget():
    pts = [("fixed 6 goals", [(0, "hw6g16_v10s2it20", "start"), (12, "igfix_b12", "final"), (24, "igfix_b24", "final"),
                              (24, "sens_none", "final"), (24, "gate_loose", "final"), (48, "hw6g_v10s2it20", "final"),
                              (96, "hw6g16_v10s2it20", "final")]),
           ("information-gain goals", [(12, "ig_b12", "final"), (24, "ig_b24", "final")]),
           ("random goals", [(12, "igrnd_b12", "final"), (24, "igrnd_b24", "final")]),
           ("Abbeel-style direction", [(24, "hw_abbeel", "final")]),
           ("our learner + DR (A), our hardware stage", [(0, "drA_hw24", "start"), (24, "drA_hw24", "final")]),
           ("our learner + DR (B), our hardware stage", [(0, "drB_hw24", "start"), (24, "drB_hw24", "final")])]
    SLOT = [0, 6, 7, 2, 3, 3]                      # categorical slot per series (the DR pair shares slot 3)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for k, (lab, ps) in enumerate(pts):
        xs, ys, los, his = [], [], [], []
        pooled = {}
        for (n, run, tag) in ps:                     # runs at the same budget pooled (the 24-jump stage ran 3 times)
            pooled.setdefault(n, []).extend(load(run, tag))
        for n, rs in sorted(pooled.items()):
            if not rs:
                continue
            p, lo, hi = wilson(sum(ok(r) for r in rs), len(rs))
            xs.append(n + (k - 1) * 0.9)
            ys.append(100 * p)
            los.append(100 * (p - lo))
            his.append(100 * (hi - p))
        ax.errorbar(xs, ys, yerr=[los, his], color=CAT[SLOT[k]], marker=MARK[k], ms=6, mec=SURF, lw=0, elinewidth=1,
                    capsize=2, label=lab, zorder=3)
        if k == 0 or k >= 4:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], color=CAT[SLOT[k]], lw=1.5 if k == 0 else 1.0,
                    ls="-" if k != 5 else ":", zorder=2)
    # per-goal ILC (JumpILC) trained on the test goals themselves: trial k+1 per goal, and re-flown with our protocol
    jt = [(8 * kk, jilc_at(kk)) for kk in range(13)]
    jt = [(n, rs) for n, rs in jt if rs]
    if jt:
        ax.plot([n for n, _ in jt], [100 * np.mean([ok(r) for r in rs]) for _, rs in jt], color=CAT[1], lw=1.2, ls="--",
                label="per-goal ILC on the test goals (trial k+1)", zorder=2)
    je = [(8 * kk, jilc_eval(kk)) for kk in (3, 12)]
    je = [(n, rs) for n, rs in je if rs]
    if je:
        ps_ = [wilson(sum(ok(r) for r in rs), len(rs)) for _, rs in je]
        ax.errorbar([n for n, _ in je], [100 * q[0] for q in ps_], yerr=[[100 * (q[0] - q[1]) for q in ps_],
                    [100 * (q[2] - q[0]) for q in ps_]], color=CAT[1], marker="s", ms=6, mec=SURF, lw=0, elinewidth=1,
                    capsize=2, label="per-goal ILC, re-flown with our protocol", zorder=3)
    for lab, run, tag, c in (("PPO + DR", "base/eval_ppo_plain", "start", 4), ("RMA", "base/eval_rma", "start", 5)):
        rs = load(run, tag)
        if rs:
            v = 100 * np.mean([ok(r) for r in rs])
            ax.axhline(v, color=CAT[c], lw=1, ls="--")
            ax.text(97, v + 0.6, f"{lab} (zero-shot, 77-102k DR sim jumps)", ha="right", fontsize=7, color=INK2)
    ax.set_xlabel("real jumps per robot")
    ax.set_ylabel("success on the reserved test goals (%)")
    ax.set_xlim(-4, 100)
    ax.set_ylim(0, 55)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_title("Hardware-stage budget (4 CPU robots; ours from 1,920 nominal sim jumps; 24 = 3 runs pooled)")
    save(fig, "06_real_jump_budget.png")


# -- 7. success vs tolerance -----------------------------------------------------------------------------------------
def fig_tolerance():
    import sys
    sys.path.insert(0, P)
    import success_metrics as sm
    rows = []
    for lab, run, tag in METHODS:
        rs = sm.load(run, tag)
        if rs:
            rows.append((lab, sm.summary(rs)))
    sm.plot(rows, os.path.join(OUT, "07_success_vs_tolerance.png"))


# -- 8. sim training -------------------------------------------------------------------------------------------------
def fig_sim_training():
    e = json.load(open(os.path.join(P, "v10_s2", "evals.json")))
    it = np.array([x["it"] for x in e])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, (key, lab) in enumerate((("clean", "nominal starts"), ("perturbed", "perturbed starts (sigma_q 0.08)"))):
        ax.plot(96 * it, [100 * x[key]["ok"] for x in e], color=CAT[k], marker=MARK[k], lw=2, ms=5, mec=SURF, label=lab + ", success")
        ax.plot(96 * it, [100 * x[key]["fell"] for x in e], color=CAT[k], lw=1.2, ls="--", label=lab + ", falls")
    ax.axvline(96 * 20, color=INK2, lw=1, ls=":")
    ax.text(96 * 20 + 120, 3, "deployed checkpoint\n(it 20, 1,920 jumps)", fontsize=7.5, color=INK2, va="bottom")
    ax.set_xlabel("nominal sim jumps (96 per ILC iteration)")
    ax.set_ylabel("% of the 76-goal plane")
    ax.set_ylim(0, 100)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Sim stage: one ILC step per jump, regressed into the network (nominal GPU sim)")
    save(fig, "08_sim_training.png")


# == the perturbation axes, separated =================================================================================
# robot set for the axis figures: r1, s1, r4m (r4's physics, good mocap), r5 -- the constant-perturbation sweep's set
AX_RB = ["real_r1", "real_s1", "real_r4m", "real_r5"]
# where each robot's runs live: zero-shot / after the nominal 24-jump stage
SRC_ZS = {"real_r1": "hw6g16_v10s2it20", "real_s1": "hw6g16_v10s2it20", "real_r5": "hw6g16_v10s2it20", "real_r4m": "base6_r4m",
          "real_r4": "hw6g16_v10s2it20"}
SRC_24 = {"real_r1": "gate_loose", "real_s1": "gate_loose", "real_r5": "gate_loose", "real_r4m": "base6_r4m", "real_r4": "gate_loose"}
SRC_96 = {"real_r1": "hw6g16_v10s2it20", "real_s1": "hw6g16_v10s2it20", "real_r5": "hw6g16_v10s2it20", "real_r4": "hw6g16_v10s2it20"}
# physics: motor strength / mass (the thrust-to-weight deficit the short landings come from); r4m = r4's physics
STW = {"real_r1": 0.929 / 0.969, "real_s1": 1.0 / 1.05, "real_r4m": 0.864 / 0.982, "real_r4": 0.864 / 0.982, "real_r5": 0.943 / 1.14}
STANCE = {"noseup": 0.224, "nosedn": 0.224, "crouch": 0.316, "tall": 0.316}       # |planar joint offset| (rad)
SENSE_NAME = {"none": "nominal sensing", "delay10": "10 ms actuation delay", "mocapbad": "bad mocap (120 Hz, 15 ms)"}
TRAIN = np.array([(0.45, 0), (0.60, 0), (0.50, 0.15), (0.60, 0.10), (0.50, 0.05), (0.575, 0.15)])


def ld(src, rb, tag, rob=False):
    run = src.get(rb) if isinstance(src, dict) else src
    if run is None:
        return []
    return load(run, tag, rob=rob, robots=[rb])


def by_cond(rs, p):
    return [r for r in rs if r.get("cond", "").endswith("+" + p)]


def stage_sets(p=None):
    """per stage the jumps (r1, s1, r4m, r5; test goals) under condition p (None: nominal starts):
    zero-shot, after the nominal 24-jump stage, after a 24-jump stage trained under p (tp_<p>)."""
    out = {}
    for lab, src, tag in (("zero-shot", SRC_ZS, "start"), ("24 jumps, nominal training", SRC_24, "final")):
        rs = []
        for rb in AX_RB:
            rs += ld(src, rb, tag, rob=p is not None) if p is None else by_cond(ld(src, rb, tag, rob=True), p)
        out[lab] = rs
    if p is not None:
        out["24 jumps, trained under this condition"] = [r for rb in AX_RB for r in by_cond(load(f"tp_{p}", "final", rob=True, robots=[rb]), p)]
    return out


def dot(ax, x, rs, c, m, lab=None, dx=0.0):
    if not rs:
        return
    p, lo, hi = wilson(sum(ok(r) for r in rs), len(rs))
    ax.errorbar(x + dx, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color=CAT[c], marker=MARK[c], ms=6,
                mec=SURF, lw=0, elinewidth=1, capsize=2, label=lab, zorder=3)


def nearest_train(g):
    return float(np.min(np.linalg.norm(TRAIN - np.asarray(g)[None], axis=1)))


def grid_jumps(stage, robots=AX_RB):
    rs = []
    for rb in robots:
        f = os.path.join(P, "figdata", f"grid_{stage}_{rb}.json")
        if os.path.exists(f):
            rs += [dict(r, robot=rb) for r in json.load(open(f))]
    return rs


# -- 9. success along each axis, each with its own x -------------------------------------------------------------
def fig_axes():
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True, gridspec_kw=dict(width_ratios=[1.2, 1, 1.5, 1.3]))
    stages = [("zero-shot", 0), ("24 jumps, nominal training", 1), ("96 jumps, nominal training", 2), ("24 jumps, trained under this condition", 6)]
    # (a) environmental: physics -- strength-to-weight per robot (nominal starts and sensing)
    ax = axs[0]
    order = sorted(AX_RB, key=lambda rb: -STW[rb])          # strongest first
    for k, (lab, c) in enumerate(stages[:3]):
        src, tag = {0: (SRC_ZS, "start"), 1: (SRC_24, "final"), 2: (SRC_96, "final")}[c]
        for i, rb in enumerate(order):
            dot(ax, i, ld(src, rb, tag), c, c, None, dx=(k - 1) * 0.15)
    ax.set_xticks(range(len(order)), [f"{rb.replace('real_', '')}\n{STW[rb]:.2f}" for rb in order], fontsize=8)
    ax.set_xlabel("robot, strength-to-weight (motor / mass; nominal 1.00)")
    ax.set_ylabel("success on the test goals (%)")
    ax.set_title("Environment: physics (each robot)")
    # (b) environmental: sensing / actuation
    ax = axs[1]
    for i, p in enumerate(["none", "delay10", "mocapbad"]):
        sets = stage_sets(None if p == "none" else p)
        for lab, c in stages:
            dot(ax, i, sets.get(lab.replace("96 jumps", "x"), []), c, c, None, dx=(stages.index((lab, c)) - 1.5) * 0.12)
    ax.set_xticks(range(3), ["nominal\nsensing", "10 ms\nactuation delay", "bad mocap\n120 Hz, 15 ms"], fontsize=7.5)
    ax.set_title("Environment: sensing / actuation")
    # (c) starting condition
    ax = axs[2]
    conds = ["nominal", "noseup", "nosedn", "crouch", "tall", "blk15", "blk2"]
    for i, p in enumerate(conds):
        sets = stage_sets(None if p == "nominal" else p)
        for k, (lab, c) in enumerate(stages):
            dot(ax, i, sets.get(lab, []), c, c, None, dx=(k - 1.5) * 0.12)
    short = dict(noseup="nose\nup", nosedn="nose\ndown", crouch="crouched", tall="tall", blk15="block\n1.5 cm", blk2="block\n2 cm")
    ax.set_xticks(range(len(conds)), ["nominal"] + [f"{short[p]}\n{STANCE[p]:.2f} rad" if p in STANCE else short[p] for p in conds[1:]], fontsize=7.5)
    ax.axvline(4.5, color=GRID, lw=1)
    ax.text(5.5, 67, "terrain under feet", ha="center", fontsize=7.5, color=INK2)
    ax.text(2.5, 67, "stance posture", ha="center", fontsize=7.5, color=INK2)
    ax.set_title("Starting condition (same robots)")
    # (d) goal: distance from the nearest hardware-stage training goal (76-goal grid, CPU robots)
    ax = axs[3]
    bins = [(-0.001, 0.001, "trained\ngoal"), (0.001, 0.03, "< 3 cm"), (0.03, 0.06, "3-6 cm"), (0.06, 1.0, "> 6 cm")]
    for k, (lab, c, stage) in enumerate((("zero-shot", 0, "zeroshot"), ("24 jumps, nominal training", 1, "after24"))):
        rs = grid_jumps(stage)
        for i, (lo, hi, _) in enumerate(bins):
            dot(ax, i, [r for r in rs if lo < nearest_train(r["goal"]) <= hi or (lo < 0 and nearest_train(r["goal"]) < 1e-3)], c, c, None, dx=(k - 0.5) * 0.15)
    ax.set_xticks(range(len(bins)), [b[2] for b in bins], fontsize=7.5)
    ax.set_xlabel("distance in (x, h) from the nearest training goal")
    ax.set_title("Goal (76-goal grid, 2 episodes per goal)")
    for ax in axs:
        style(ax)
        ax.set_ylim(0, 70)
    handles = [plt.Line2D([], [], color=CAT[c], marker=MARK[c], lw=0, ms=6, mec=SURF) for _, c in stages]
    fig.legend(handles, [s[0] for s in stages], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Success along each perturbation axis separately (robots r1, s1, r4m, r5; 95% intervals; the GPU sim always nominal)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "09_success_per_axis.png")


# -- 10. what the hardware stage absorbs, per axis --------------------------------------------------------------------
def dprop(a, b):
    """after - before and its 95% normal interval (difference of proportions)."""
    if not a or not b:
        return np.nan, np.nan
    p1, p0 = np.mean([ok(r) for r in a]), np.mean([ok(r) for r in b])
    se = np.sqrt(p1 * (1 - p1) / len(a) + p0 * (1 - p0) / len(b))
    return p1 - p0, 1.96 * se


def fig_absorb():
    rows = []           # (axis, label, [(stage label, colour, delta, ci)])
    for rb in AX_RB:
        rows.append(("environment: physics", rb.replace("real_", ""),
                     [("nominal training", 1, *dprop(ld(SRC_24, rb, "final"), ld(SRC_ZS, rb, "start")))]))
    for p in PERT_SENSE + PERT_START:
        s = stage_sets(p)
        ax_name = "environment: sensing" if p in PERT_SENSE else "starting condition"
        rows.append((ax_name, PERT_NAME[p], [("nominal training", 1, *dprop(s["24 jumps, nominal training"], s["zero-shot"])),
                                              ("trained under it", 6, *dprop(s["24 jumps, trained under this condition"], s["zero-shot"]))]))
    zs, af = grid_jumps("zeroshot"), grid_jumps("after24")
    for lab, sel in (("trained goals", lambda g: nearest_train(g) < 1e-3), ("unseen, < 6 cm", lambda g: 1e-3 < nearest_train(g) <= 0.06),
                     ("unseen, > 6 cm", lambda g: nearest_train(g) > 0.06)):
        rows.append(("goal", lab, [("nominal training", 1, *dprop([r for r in af if sel(r["goal"])], [r for r in zs if sel(r["goal"])]))]))
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(rows) + 1.6))
    ys = np.arange(len(rows))[::-1]
    prev = None
    for y, (axn, lab, ds) in zip(ys, rows):
        for k, (sl, c, d, ci) in enumerate(ds):
            if np.isfinite(d):
                ax.errorbar(100 * d, y + (k - 0.5) * 0.25 * (len(ds) > 1), xerr=100 * ci, color=CAT[c], marker=MARK[c], ms=6,
                            mec=SURF, lw=0, elinewidth=1, capsize=2, zorder=3)
        if axn != prev:
            ax.axhline(y + 0.5, color=GRID, lw=1)
            ax.text(-58, y + 0.15, axn, fontsize=8.5, color=INK, weight="bold", va="bottom")
            prev = axn
    ax.set_yticks(ys, [r[1] for r in rows], fontsize=8)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-60, 60)
    ax.set_xlabel("change in success from the 24-jump hardware stage (points; 95% interval)")
    handles = [plt.Line2D([], [], color=CAT[c], marker=MARK[c], lw=0, ms=6, mec=SURF) for c in (1, 6)]
    ax.legend(handles, ["stage trained from the nominal start", "stage trained under that condition"], frameon=False, fontsize=8, loc="lower right")
    style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_title("What the hardware stage absorbs, per perturbation axis (r1, s1, r4m, r5; vs zero-shot)")
    save(fig, "10_what_the_stage_absorbs.png")


# -- 11. trained-on x tested-on --------------------------------------------------------------------------------------
def fig_matrix():
    conds = ["nominal"] + PERT_SENSE + PERT_START
    M = np.full((len(conds), len(conds)), np.nan)
    for i, tr in enumerate(conds):
        for j, te in enumerate(conds):
            if tr == "nominal":
                rs = [r for rb in AX_RB for r in (ld(SRC_24, rb, "final") if te == "nominal" else by_cond(ld(SRC_24, rb, "final", rob=True), te))]
            else:
                rs = [r for rb in AX_RB for r in (load(f"tp_{tr}", "final", robots=[rb]) if te == "nominal" else by_cond(load(f"tp_{tr}", "final", rob=True, robots=[rb]), te))]
            if rs:
                M[i, j] = np.mean([ok(r) for r in rs])
    if np.isnan(M[1:]).all():
        print("skip trained x tested matrix (no constant-perturbation runs yet)")
        return
    fig, ax = plt.subplots(figsize=(9.5, 7))
    im = ax.imshow(100 * M, cmap=SEQ, vmin=0, vmax=60)
    for i in range(len(conds)):
        for j in range(len(conds)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=8, color=INK if M[i, j] < 0.35 else "#ffffff",
                        weight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.5))
    names = ["nominal"] + [PERT_NAME[c] for c in conds[1:]]
    ax.set_xticks(range(len(conds)), names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(conds)), names, fontsize=8)
    for v in (0.5, 2.5):
        ax.axvline(v, color=SURF, lw=3)
        ax.axhline(v, color=SURF, lw=3)
    ax.set_xlabel("tested on")
    ax.set_ylabel("hardware stage trained under (24 jumps)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("success (%)")
    ax.set_title("Trained-on x tested-on (r1, s1, r4m, r5; test goals). Groups: nominal | sensing | starting condition")
    save(fig, "11_trained_x_tested.png")


# -- 12. interactions ---------------------------------------------------------------------------------------------
def fig_interactions():
    conds = ["nominal", "delay10", "mocapbad"] + PERT_START
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 0.7]))
    for ax, (title, src, tag) in zip(axs[:2], (("robot x condition, zero-shot", SRC_ZS, "start"), ("robot x condition, after 24 jumps (nominal training)", SRC_24, "final"))):
        M = np.full((len(AX_RB), len(conds)), np.nan)
        for i, rb in enumerate(AX_RB):
            for j, p in enumerate(conds):
                rs = ld(src, rb, tag) if p == "nominal" else by_cond(ld(src, rb, tag, rob=True), p)
                if rs:
                    M[i, j] = np.mean([ok(r) for r in rs])
        ax.imshow(100 * M, cmap=SEQ, vmin=0, vmax=70, aspect="auto")
        for i in range(len(AX_RB)):
            for j in range(len(conds)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color=INK if M[i, j] < 0.4 else "#ffffff")
        ax.set_xticks(range(len(conds)), ["nominal"] + [PERT_NAME[c] for c in conds[1:]], rotation=35, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(AX_RB)), [f"{rb.replace('real_', '')} (s/w {STW[rb]:.2f})" for rb in AX_RB], fontsize=8)
        for v in (0.5, 2.5):
            ax.axvline(v, color=SURF, lw=3)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title)
    # robot x goal distance (grid)
    ax = axs[2]
    bins = [(-1, 1e-3, "trained"), (1e-3, 0.06, "< 6 cm"), (0.06, 1, "> 6 cm")]
    M = np.full((len(AX_RB), 2 * len(bins)), np.nan)
    for i, rb in enumerate(AX_RB):
        for s_, stage in enumerate(("zeroshot", "after24")):
            rs = grid_jumps(stage, robots=[rb])
            for j, (lo, hi, _) in enumerate(bins):
                sel = [r for r in rs if lo < nearest_train(r["goal"]) <= hi]
                if sel:
                    M[i, s_ * len(bins) + j] = np.mean([ok(r) for r in sel])
    ax.imshow(100 * M, cmap=SEQ, vmin=0, vmax=70, aspect="auto")
    for i in range(len(AX_RB)):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color=INK if M[i, j] < 0.4 else "#ffffff")
    ax.set_xticks(range(M.shape[1]), [f"{'0-shot' if k < len(bins) else 'after'}\n{b[2]}" for k, b in enumerate(bins * 2)], fontsize=7)
    ax.set_yticks(range(len(AX_RB)), [rb.replace("real_", "") for rb in AX_RB], fontsize=8)
    ax.axvline(len(bins) - 0.5, color=SURF, lw=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("robot x goal distance (76-goal grid)")
    fig.suptitle("Interactions between the axes: environment (rows) x starting condition / sensing / goal (success %)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "12_axis_interactions.png")


if __name__ == "__main__":
    for f in (fig_methods, fig_per_pert, fig_robots, fig_dyn_sweeps, fig_goal_maps, fig_budget, fig_tolerance, fig_sim_training,
              fig_axes, fig_absorb, fig_matrix, fig_interactions):
        try:
            f()
        except Exception as ex:                      # one figure's missing data must not stop the rest
            print(f"{f.__name__} failed: {type(ex).__name__}: {ex}")
