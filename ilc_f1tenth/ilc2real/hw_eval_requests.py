#!/usr/bin/env python3
"""
hw_eval_requests.py --session-dir DIR --policy NPZ --tag NAME [--plan P] [--laps L] [--runs N]: queue zero-shot (or any
fixed-network) evaluation flights for the REAL car's net_runner (ilc_f1tenth_hw), the hardware counterpart of eval_free.py
(2026-10-30): N requests of L chained laps on plan P, the policy file as is -- an MLP, an RMA policy_final.npz (rma_adapt.npz
must sit next to it) or a FADA idm.npz / <car>_policy.npz; net_runner loads them through policy_runtime.load_policy, the
same loader the simulated evaluations use. hw_stage.py --backend manual sends MLPs only, so the baselines go through here.
    --score: score every finished request of this tag (rc.lap_metrics: rms e_y, pace, laps, crash -> eval_free's nerr)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--session-dir', required=True)
ap.add_argument('--policy', default='')
ap.add_argument('--tag', required=True, help='a name for these flights (e.g. rma_s0_plastic); kept in each request')
ap.add_argument('--plan', default='mocap_square2fast_b0_free_v08')
ap.add_argument('--laps', type=float, default=5.0)
ap.add_argument('--runs', type=int, default=4)
ap.add_argument('--launch', type=float, nargs=2, default=[1.0, 8.0], metavar=('V_HAND', 'CURRENT'))
ap.add_argument('--score', action='store_true')
a = ap.parse_args()
sd = os.path.expanduser(a.session_dir)
os.makedirs(sd, exist_ok=True)

if a.score:
    import bank
    import real_car as rc
    rows = []
    for f in sorted(os.listdir(sd)):
        if not (f.startswith('req_') and f.endswith('.json')):
            continue
        k = f[4:-5]
        q = json.load(open(os.path.join(sd, f)))
        if q.get('tag') != a.tag or not os.path.exists(os.path.join(sd, f'res_{k}.npz')):
            continue
        d = np.load(os.path.join(sd, f'res_{k}.npz'), allow_pickle=True)
        r = {n: (d[n].item() if d[n].ndim == 0 else d[n]) for n in d.files}
        lm = rc.lap_metrics(r, bank.load_bank([q['plan']])[0])
        nerr = 5.0 if lm.get('crash') else max(lm['rms_ey'] / 0.10, (1 - lm['pace']) / 0.1)
        rows.append(dict(req=k, plan=q['plan'], rms_ey=lm.get('rms_ey'), pace=lm.get('pace'), laps=lm.get('laps'),
                         crash=lm.get('crash', ''), nerr=nerr))
        print(json.dumps(rows[-1]))
    if rows:
        print(f"{a.tag}: {len(rows)} flights, nerr {np.mean([r['nerr'] for r in rows]):.2f}")
    sys.exit(0)

assert a.policy and os.path.exists(a.policy), a.policy
k0 = len([f for f in os.listdir(sd) if f.startswith('req_') and f.endswith('.json')])
for i in range(a.runs):
    k = f'{k0 + i:05d}'
    tmp = os.path.join(sd, f'.req_{k}.json')
    json.dump(dict(plan=a.plan, policy=os.path.abspath(a.policy), laps=float(a.laps), tag=a.tag,
                   launch=dict(v_hand=float(a.launch[0]), current=float(a.launch[1]))), open(tmp, 'w'), indent=1)
    os.replace(tmp, os.path.join(sd, f'req_{k}.json'))
print(f"queued {a.runs} x {a.laps} laps of {a.plan} with {a.policy} as {a.tag}: req_{k0:05d} .. req_{k0 + a.runs - 1:05d}")
