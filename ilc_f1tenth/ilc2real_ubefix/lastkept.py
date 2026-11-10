#!/usr/bin/env python3
"""lastkept.py STAGE_DIR OUT_DIR (2026-10-30): a CANDIDATE return rule for the car stage, for evaluation only (adopting it needs the
user's OK): per car, return the network of the LAST iteration whose trial was flown and kept, instead of the final update (which
is computed after the last trial and never flown). Uses only the stage's own trial record (summary.json 'rollbacks': iteration k's
trial rolled back -> that network was not kept), never an evaluation score. Needs a stage run with --save-its 1
(STAGE_DIR/its/it<k>_<car>_policy.npz = the network flown in iteration k, k = 0 .. iters - 1).
-> OUT_DIR/<car>_policy.npz (+ args.json copied, and lastkept.json: car -> k), for eval_free.py."""
import json
import os
import shutil
import sys

src, out = sys.argv[1:3]
S = json.load(open(os.path.join(src, "summary.json")))
iters = json.load(open(os.path.join(src, "args.json")))["iters"]
os.makedirs(out, exist_ok=True)
pick = {}
for c in S["rollbacks"]:                  # every car of the stage (summary "cars" is empty for --backend sim)
    bad = {r["it"] for r in S["rollbacks"].get(c, [])}
    k = max(k for k in range(iters) if k not in bad or k == 0)
    shutil.copy(os.path.join(src, "its", f"it{k}_{c}_policy.npz"), os.path.join(out, f"{c}_policy.npz"))
    pick[c] = k
shutil.copy(os.path.join(src, "args.json"), os.path.join(out, "args.json"))
json.dump(pick, open(os.path.join(out, "lastkept.json"), "w"), indent=1)
print(out, pick)
