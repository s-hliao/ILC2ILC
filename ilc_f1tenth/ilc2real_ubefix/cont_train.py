#!/usr/bin/env python3
"""cont_train.py RUN SEED GPU (2026-10-31; user: "figure out why our ppo is failing, maybe you just have to train longer"):
continue runs/<RUN>_s<SEED> (freePPOV = PPO + DR, rmaV = RMA) for 300 more PPO iterations from its final actor, critic and
exploration level (ppo_dr.py --init + --resume), with the run's own arguments otherwise; RMA then refits its phase 2
(adaptation module, 150 iterations) on the continued teacher. Both baselines get the same extra budget.
-> runs/<RUN>600_s<SEED>/."""
import json, subprocess, sys
run, s, gpu = sys.argv[1:4]
src = f"runs/{run}_s{s}"
a = json.load(open(f"{src}/args.json"))
a.update(out=f"runs/{run}600_s{s}", init=f"{src}/policy_final.npz", resume=f"{src}/agent_final.npz", iters=300, gpu=gpu)
cmd = [sys.executable, "-u", "ppo_dr.py"]
for k, v in a.items():
    if v is None or v is False:
        continue
    f = f"--{k.replace('_', '-')}"
    cmd += [f] if v is True else ([f] + [str(x) for x in v] if isinstance(v, list) else [f, str(v)])
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
