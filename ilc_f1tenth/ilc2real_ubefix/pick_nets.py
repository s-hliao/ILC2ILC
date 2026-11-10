#!/usr/bin/env python3
"""pick_nets.py: eval_axes.py --net specs (NAME=SPEC) for the car's picked baselines (src/paper/baselines.json: the best
non-oracle / oracle RMA and FADA): rma_nonoracle=..., rma_oracle=..., fada_nonoracle=..., fada_oracle=... A pick whose arm
holds per-car adapted networks is 'arm:runs/hw/<arm>'; a zero-shot pick is 'zs:<its policy>' (its args.json)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BJ = json.load(open(os.path.join(HERE, '..', '..', 'paper', 'baselines.json')))['car']
out = []
for role in ('rma_nonoracle', 'rma_oracle', 'fada_nonoracle', 'fada_oracle'):
    b = BJ[role]['best']
    arm = b['source'] if b['kind'] == 'point' else f"{b['source']}_ep{b['best_budget']}"
    d = os.path.join(HERE, 'runs', 'hw', arm)
    if os.path.exists(os.path.join(d, 'real_nom_policy.npz')):
        out.append(f'{role}=arm:runs/hw/{arm}')
    else:
        out.append(f"{role}=zs:{json.load(open(os.path.join(d, 'args.json')))['policy']}")
print(' '.join(out))
