#!/usr/bin/env python3
"""train_like.py SCRIPT SRC_RUN OUT GPU [key=value ...] (2026-10-31, the nominal-mu-0.6 rebuild on plans_nom06): rerun SCRIPT
(sim_train.py / ppo_dr.py) with SRC_RUN's exact saved arguments (runs/<SRC_RUN>/args.json), changing only --out, --gpu and the
given keys (plans=a,b,c lists, init=path, ...). The environment (F1T_MU, F1T_PLANS, F1T_DR_MU) comes from the caller."""
import json, subprocess, sys
script, src, out, gpu = sys.argv[1:5]
a = json.load(open(f"runs/{src}/args.json"))
a.update(out=f"runs/{out}", gpu=gpu)
for kv in sys.argv[5:]:
    k, v = kv.split("=", 1)
    a[k] = v.split(",") if k in ("plans", "eval_plans") else (None if v == "None" else v)
cmd = [sys.executable, "-u", script]
for k, v in a.items():
    if v is None or v is False:
        continue
    f = f"--{k.replace('_', '-')}"
    cmd += [f] if v is True else ([f] + [str(x) for x in v] if isinstance(v, list) else [f, str(v)])
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
