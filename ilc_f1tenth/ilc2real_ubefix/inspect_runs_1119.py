"""inspect_runs_1119.py (2026-11-19): replay saved stage networks on a stand-in car and break the lateral error down by phase.

Phases: launch (before the hand-over to the network), the first 1 s after it, lap 1, lap 2. Also the pace (mean |V| over the
plan's |V| at the same arc length) and the time beyond 0.5 / 1.0 m. Several launch currents can be compared.

    python inspect_runs_1119.py --car ax:mu=0.4 --plan mocap_fig8R115_b0_free_x095 --run runs/screen/HW1116L_... --currents 8 4
"""
import argparse, glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank                 # noqa: E402
import real_car as rc       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--car', required=True)
ap.add_argument('--plan', required=True)
ap.add_argument('--run', default='', help='stage dir with its/it<k>_*_policy.npz; empty: --policy only')
ap.add_argument('--policy', default='runs/freePPON_s0/policy_final.npz')
ap.add_argument('--its', default='', help='comma list of iterations (default all)')
ap.add_argument('--currents', type=float, nargs='+', default=[8.0])
ap.add_argument('--v-hand', type=float, default=1.0)
ap.add_argument('--laps', type=float, default=2.0)
ap.add_argument('--seed', type=int, default=701)
ap.add_argument('--rma', default='', help='RMA base dir (rma_adapt.npz): the replayed actors fly with its frozen adaptation module (2026-11-20)')
a = ap.parse_args()

plan = bank.load_bank([a.plan])[0]
L = float(plan['L']) if 'L' in plan else float(np.asarray(plan['s'])[-1])


def load(path):
    d = np.load(path)
    n = len([k for k in d.files if k.startswith('W')])
    return [(np.asarray(d[f'W{i}']), np.asarray(d[f'b{i}'])) for i in range(n)]


pols = []
RMA = None
if a.rma:                                            # as hw_stage.load_policy: phi + history length; the stage trains the actor only
    e = np.load(os.path.join(a.rma, 'rma_adapt.npz'))
    RMA = dict(adapt=[(np.asarray(e[f'W{i}']), np.asarray(e[f'b{i}'])) for i in range(len([k for k in e.files if k.startswith('W')]))],
               hist=int(e['hist']))
if a.run:
    fs = sorted(glob.glob(os.path.join(a.run, 'its', 'it*_policy.npz')), key=lambda f: int(re.findall(r'/it(\d+)_', f)[0]))
    keep = None if not a.its else {int(x) for x in a.its.split(',')}
    pols = [(f'it{re.findall(r"/it(\d+)_", f)[0]}', load(f)) for f in fs if keep is None or int(re.findall(r'/it(\d+)_', f)[0]) in keep]
else:
    pols = [('zs', load(a.policy))]
rms = lambda q: float(np.sqrt(np.mean(np.square(q)))) if len(q) else float('nan')
print(f'car {a.car}  plan {a.plan}  L {L:.2f} m')
for cur in a.currents:
    for name, pol in pols:
        r = rc.run(a.car, plan, policy=dict(RMA, actor=pol) if RMA else pol, laps=a.laps, seed=a.seed, launch=dict(v0=0.6, v_hand=a.v_hand, current=cur))
        e = np.asarray(r['err_true'])
        ey = e[:, 0]
        kh = int(r['handover']) if r.get('handover') is not None else 0
        prog = np.cumsum(np.r_[0, np.abs(np.diff(np.unwrap(np.asarray(r['idx']) * 2 * np.pi / len(plan['x']))))]) / (2 * np.pi)
        lap1 = (np.arange(len(ey)) >= kh) & (prog < 1.0)
        lap2 = (prog >= 1.0)
        z = np.asarray(plan['z'])[np.asarray(r['idx'])]
        x = np.asarray(r['x_true'])
        v, vr = np.hypot(x[:, 3], x[:, 4]), np.hypot(z[:, 2], z[:, 3])
        post = np.arange(len(ey)) >= kh
        print(f' launch {cur:4.1f} A  {name:4s}  launch rms {100 * rms(ey[:kh]):5.1f} max {100 * np.abs(ey[:kh]).max() if kh else 0:5.1f} | '
              f'first 1 s after {100 * rms(ey[kh:kh + 40]):5.1f} | lap1 {100 * rms(ey[lap1]):5.1f} lap2 {100 * rms(ey[lap2]):5.1f} | '
              f'peak {100 * np.abs(ey).max():5.1f} cm | pace {np.mean(v[post]) / np.mean(vr[post]):.2f} | >0.5 m {np.mean(np.abs(ey) > 0.5) * len(ey) * rc.DT:4.1f} s'
              f' >1.0 m {np.mean(np.abs(ey) > 1.0) * len(ey) * rc.DT:4.1f} s | laps {r["laps"]:.2f} {r["crash"] or ""}', flush=True)
