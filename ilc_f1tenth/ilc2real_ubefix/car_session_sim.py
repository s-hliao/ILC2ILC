#!/usr/bin/env python3
"""
car_session_sim.py --session-dir DIR --car real_mu: a desk stand-in for the real car's net_runner (ilc_f1tenth_hw), for
testing hw_stage.py --backend manual without hardware. Watches DIR for req_<k>.json (as net_runner does), flies each request
on a SIMULATED car with real_car.run (the request's plan and policy npz, a slow rolling start: launch v0 0.6 m/s, the
request's v_hand / current), and writes res_<k>.npz / res_<k>.json in net_runner's format (policy_runtime.Controller.record:
the 'true' entries are the measured ones -- the real car has no ground truth; crash = the room's envelope, --safety-ey).
    source env_tight.sh && python car_session_sim.py --session-dir /tmp/sess --car real_mu
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--session-dir', required=True)
ap.add_argument('--car', default='real_nom')
ap.add_argument('--safety-ey', type=float, default=0.5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--once', action='store_true', help='serve the pending requests and exit')
a = ap.parse_args()
os.environ['F1T_SAFETY_EY'] = str(a.safety_ey)          # real_car.run's envelope = net_runner's safety_ey
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank                                      # noqa: E402
import policy_runtime as prt                     # noqa: E402
import real_car as rc                            # noqa: E402

PER_STEP = ('x', 'obs', 'mu', 'a', 'err', 'idx', 'applied')


def serve(k, req):
    plan = bank.load_plan(req['plan'])
    plan['name'] = req['plan']
    pol = prt.load_policy(req['policy'])
    la = req.get('launch') or dict(v_hand=1.0, current=8.0)
    t0 = time.time()
    r = rc.run(a.car, plan, pol, laps=float(req.get('laps', 2.05)), s0=0.0, pert=np.zeros(5), seed=a.seed + int(k),
               launch=dict(v0=0.6, v_hand=float(la['v_hand']), current=float(la['current'])))
    # net_runner's record: the measured state stands in for the truth; the failure checks on the measured state
    rec = {n: np.asarray(r[n]) for n in PER_STEP}
    rec.update(x_true=rec['x'], obs_true=rec['obs'], err_true=rec['err'], x_end=rec['x'][-1], idx_end=r['idx_end'],
               progress=r['progress'], laps=r['laps'], failed=r['failed'], crash=r['crash'],
               peak_ey=float(np.max(np.abs(rec['err'][:, 0]))) if len(rec['err']) else 0.0, max_ey=a.safety_ey,
               steps=len(rec['x']), car='real', handover=r['handover'], plan=req['plan'], seconds=time.time() - t0)
    sd = a.session_dir
    np.savez(os.path.join(sd, f'res_{k}.npz'), **{n: np.asarray(v) for n, v in rec.items() if v is not None})
    summ = dict(plan=req['plan'], laps=float(rec['laps']), failed=bool(rec['failed']), crash=rec['crash'],
                steps=int(rec['steps']), peak_ey=float(rec['peak_ey']), handover=int(rec['handover']),
                seconds=rec['seconds'], sim_car=a.car)
    tmp = os.path.join(sd, f'.res_{k}.json')
    json.dump(summ, open(tmp, 'w'), indent=1)
    os.replace(tmp, os.path.join(sd, f'res_{k}.json'))
    print(json.dumps(dict(trial=k, **summ)), flush=True)


def main():
    os.makedirs(a.session_dir, exist_ok=True)
    print(f'car_session_sim: {a.session_dir}, car {a.car}, envelope {a.safety_ey} m', flush=True)
    while True:
        busy = False
        for f in sorted(glob.glob(os.path.join(a.session_dir, 'req_*.json'))):
            k = os.path.basename(f)[4:-5]
            if os.path.exists(os.path.join(a.session_dir, f'res_{k}.json')):
                continue
            serve(k, json.load(open(f)))
            busy = True
        if a.once and not busy:
            return
        if not busy:
            time.sleep(0.5)


if __name__ == '__main__':
    main()
