#!/usr/bin/env python3
"""rma_matched.py SEED GPU (2026-10-31; user: "what experiment would show our method is better than rma on cars" -> RMA at a matched
simulation budget, as the pogo's low-budget panel): rmaV_s<SEED>'s exact arguments (runs/rmaV_s<SEED>/args.json) with --iters 10
(teacher, ~5.6k sim laps incl. the BC start) and --adapt-iters 25 (phase 2, ~6.4k laps): ~12k laps, at or above our network's 8.3k
(freeDR5V_s<k>) -- the baseline gets the larger budget. -> runs/rmaV10_s<SEED>/."""
import json, subprocess, sys
s, gpu = sys.argv[1:3]
a = json.load(open(f"runs/rmaV_s{s}/args.json"))
a.update(out=f"runs/rmaV10_s{s}", iters=10, adapt_iters=25, gpu=gpu)
cmd = [sys.executable, "-u", "ppo_dr.py"]
for k, v in a.items():
    if v is None or v is False:
        continue
    flag = f"--{k.replace('_', '-')}"
    if v is True:
        cmd.append(flag)
    elif isinstance(v, list):
        cmd += [flag] + [str(x) for x in v]
    else:
        cmd += [flag, str(v)]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
