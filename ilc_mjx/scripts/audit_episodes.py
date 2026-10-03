#!/usr/bin/env python3
"""audit_episodes.py TRUTH.npz BANK.json: the base jumps of audit_truth.py as the trainer's episodes
(dilc_execute.EpisodeMaker.make with their FD Jacobians: obs, act, A, B, gn -- normalized as the critic sees
them), written next to the truth as TRUTH_ep.npz, for audit_critic.py (torch, the dilc env)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402

from ilc_mjx import host_path  # noqa: E402
from dilc_execute import EpisodeMaker, GoalBank  # noqa: E402

t = np.load(sys.argv[1])
bank = json.load(open(sys.argv[2]))
gb = GoalBank(dict(bank, plans=[dict(p, path=host_path(p["path"])) for p in bank["plans"]]))
mk = EpisodeMaker(gb, "/home/henry/mujoco_menagerie")
eps = []
for j in range(len(t["goals"])):
    g = t["goals"][j]
    rf = gb.reference(g)[0]
    eps.append(mk.make(g, rf, t["X"][j], t["U"][j], t["X"][j], bool(t["fell"][j]), jac=(t["A"][j], t["B"][j])))
out = {k: np.stack([e[k] for e in eps]) for k in ("obs", "act", "A", "B", "gn", "rc")}
srb = [mk.make(t["goals"][j], gb.reference(t["goals"][j])[0], t["X"][j], t["U"][j], t["X"][j],
               bool(t["fell"][j])) for j in range(len(t["goals"]))]
out.update(A_srb=np.stack([e["A"] for e in srb]), B_srb=np.stack([e["B"] for e in srb]))
dst = sys.argv[1].replace(".npz", "_ep.npz")
np.savez(dst, **out)
print(f"-> {dst}: obs {out['obs'].shape}, max |obs - flown obs| {np.abs(out['obs'] - t['obs']).max():.1e}")
