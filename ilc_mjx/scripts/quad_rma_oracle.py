#!/usr/bin/env python3
"""
quad_rma_oracle.py --teacher fig4/rmat_s<s>/it300 --robots ... --out fig4/eval/rmaOracle_s<s> (2026-11-07, paper/FIGURE_PLAN_1106.md
"RMA with the ORACLE latent", written before any result): the figure-4 RMA teacher's exported actor (it<k>/policy/policy.npz: input
[o (31), z (8)], its latent encoder E already merged -- checked: W0[:, 31:] == W_teacher[:, 31:] @ E, b0 == b_teacher + W_teacher[:, 31:] @ bE)
with each robot's TRUE normalized dynamics z = (JumpEnv.dyn_from_cond(robot's condition) - DYN_CENTER) / DYN_SCALE folded into the first
layer's bias (exactly quad_real_ppo.py's load_actor). Zero-shot, no real data, no stage. Writes OUT/<robot>/policy (deploy format, member
'policy'); the evaluation is fig4_eval.sh (figure 4's zero-shot scoring) on each robot.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "/home/henry/ilc_ws/src/ilc_quad/scripts")
ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True)
ap.add_argument("--robots", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from ilc_mjx.jump import JumpEnv  # noqa: E402
import dilc_execute as dx  # noqa: E402

OD = 31
src = os.path.join(a.teacher, "policy")
d = dict(np.load(os.path.join(src, "policy.npz")))
assert d["W0"].shape[1] == OD + 8, d["W0"].shape
for r in a.robots:
    raw = np.asarray(JumpEnv.dyn_from_cond(None, dx.NAMED[r]), float)
    z = (raw - JumpEnv.DYN_CENTER) / JumpEnv.DYN_SCALE
    w = {k: np.asarray(v) for k, v in d.items()}
    w["b0"] = (w["b0"].astype(np.float64) + w["W0"][:, OD:].astype(np.float64) @ z).astype(np.float32)
    w["W0"] = w["W0"][:, :OD].astype(np.float32)
    pd_ = os.path.join(a.out, r, "policy")
    os.makedirs(pd_, exist_ok=True)
    np.savez(os.path.join(pd_, "policy.npz"), **w)
    shutil.copy(os.path.join(src, "bank.json"), pd_)
    m = json.load(open(os.path.join(src, "manifest.json")))
    m.pop("dyn_dim", None)
    json.dump(m, open(os.path.join(pd_, "manifest.json"), "w"))
    json.dump(dict(robot=r, raw=raw.tolist(), z=z.tolist(), teacher=os.path.abspath(a.teacher)), open(os.path.join(a.out, r, "oracle.json"), "w"))
    print(r, "raw", np.round(raw, 3).tolist(), "z", np.round(z, 2).tolist(), flush=True)
