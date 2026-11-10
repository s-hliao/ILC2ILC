"""car_fig1_data.py RUNS_DIR OUT.json (2026-11-06, figure 1): the car's simulation-training curves from ilc2real_fix/runs/fig1
(fig1_eval.json + each run's log.jsonl) -> budget_figs2.py's data format ("sim" part).
x = simulated hours = simulated laps x 5.12 s / 3600 (the mean lap time of the 6 training plans at their plan speed), counting the
shared BC clone's laps (runs/freeBCN_s0) for every arm, RMA's phase-2 laps, and FADA's distillation laps.
(2026-11-08) FADA = fadaR1 ONLY (FIGURE_PLAN_1106.md: the in-regime FADA on RMA's phase-2 data, H 1; user decision 2026-11-07): scored by
the identical fig1_eval.py on Lambda (fig1/fig1_eval_fadaR1.json, it 40 / 75 / 150); its laps = its teacher rma_s<s>'s + its own
distillation laps (fadaR1_logs/s<s>.log sim_laps, linear between the logged iterations). The old fada_d* entries are ignored. y = the zero-shot score per panel:
mean over (seed, car), 95 % cluster bootstrap over (seed, car)."""
import glob
import json
import os
import re
import sys

import numpy as np

R, out = sys.argv[1], sys.argv[2]
LAP_S = 5.12
E = json.load(open(f"{R}/fig1/fig1_eval.json"))
E = {k: v for k, v in E.items() if "/fada_d" not in k}
E.update(json.load(open(f"{R}/fig1/fig1_eval_fadaR1.json")))
rng = np.random.default_rng(0)


def last_laps(log, upto_it=None):
    """the simulated laps in a run's log at its checkpoint upto_it (ppo_dr: events with it == upto_it - 1 or a ckpt event with it ==
    upto_it; sim_train: the eval event of upto_it), else at the end."""
    best = None
    for ln in open(log):
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if "sim_laps" not in e:
            continue
        if upto_it is None:
            best = e["sim_laps"]
        elif e.get("event") == "ckpt" and e.get("it") == upto_it:
            return e["sim_laps"]
        elif e.get("it") is not None and e.get("event") in ("ppo", "eval", "ilc") and e["it"] in (upto_it - 1, upto_it):
            best = e["sim_laps"]
    return best


clone = last_laps(f"{R}/freeBCN_s0/log.jsonl")
pts = {}                                                   # arm -> list of (laps, seed, policy file)
for f in E:
    m = re.search(r"fig1/(ppodr|ppo|ours|oursdr)_s(\d)/policy_(it(\d+)|final)\.npz", f)
    if m:
        arm = {"ppodr": "PPO + DR", "ppo": "PPO", "ours": "DistILC", "oursdr": "DistILC + DR"}[m.group(1)]
        d = os.path.dirname(f)
        it = int(m.group(4)) if m.group(4) else None
        laps = last_laps(f"{R}/{d.split('runs/', 1)[-1]}/log.jsonl" if not os.path.isabs(d) else f"{d}/log.jsonl", it)
        pts.setdefault(arm, []).append((clone + laps, int(m.group(2)), f))
        continue
    m = re.search(r"fig1/rma_s(\d)(_ck(\d+))?/policy_final\.npz", f)
    if m:
        s, k = int(m.group(1)), m.group(3)
        if k:                                             # teacher laps at it k + this phase-2's own laps
            laps = last_laps(f"{R}/fig1/rma_s{s}/log.jsonl", int(k)) + last_laps(f"{R}/fig1/rma_s{s}_ck{k}/log.jsonl")
        else:
            laps = last_laps(f"{R}/fig1/rma_s{s}/log.jsonl")
        pts.setdefault("RMA", []).append((clone + laps, s, f))
        continue
    m = re.search(r"fig1/fadaR1_s(\d)/it(\d+)_policy\.npz", f)
    if m:
        s, k = int(m.group(1)), int(m.group(2))
        lg = [json.loads(ln) for ln in open(f"{R}/fadaR1_logs/s{s}.log") if ln.startswith('{"it"')]
        own = float(np.interp(k, [0] + [e["it"] for e in lg], [0.0] + [e["sim_laps"] for e in lg]))
        pts.setdefault("FADA", []).append((clone + last_laps(f"{R}/fig1/rma_s{s}/log.jsonl") + own, s, f))
c0 = [k for k in E if k.endswith("freeBCN_s0/policy_final.npz")]
D = {"panels": [["A", "A: inside DR", 12, "cars"], ["B", "B: extrapolated", 12, "cars"], ["C", "C: outside DR class", 12, "cars"],
                ["plastic", "plastic tire", 12, "cars"], ["rubber", "rubber tire", 2, "cars"]], "sim": {}}
for arm, P in pts.items():
    D["sim"][arm] = {}
    for pan in [p[0] for p in D["panels"]]:
        rows = []
        groups = {}                                       # checkpoints of the different seeds pooled per checkpoint rank
        for laps, s, f in sorted(P):
            groups.setdefault(s, []).append((laps, f))
        nck = min(len(v) for v in groups.values())
        series = [[] for _ in range(nck)]
        for s, v in groups.items():
            for i, (laps, f) in enumerate(v[:nck]):
                series[i].append((laps, s, E[f][pan]))
        if c0 and arm != "FADA":                          # every arm's curve starts at the shared clone (FADA: its points sit at its teacher's cost + its own, no line from the clone)
            series.insert(0, [(clone, 0, E[c0[0]][pan])])
        for ser in series:
            cells = [(s, c, v["score"]) for _, s, d in ser for c, v in d.items()]
            y = np.array([v for _, _, v in cells])
            bs = [y[rng.integers(0, len(y), len(y))].mean() for _ in range(2000)]
            rows.append([float(np.mean([l for l, _, _ in ser])) * LAP_S / 3600, float(y.mean()),
                         float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        D["sim"][arm][pan] = rows
json.dump(D, open(out, "w"), indent=1)
print({a: len(v["A"]) for a, v in D["sim"].items()})
