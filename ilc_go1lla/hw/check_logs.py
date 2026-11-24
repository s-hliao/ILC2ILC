"""Check walker_node.py walk logs before trusting them (2026-11-21; run on the first real walks):
  fields / lengths, policy period jitter, IMU gyro vs mocap angular velocity (axes / signs / scale), the torque convention
  (tauEst vs the commanded PD: its spring part should match after + KD dq), and that the stage's conversion and G go through.
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu ~/venvs/pg0/bin/python hw/check_logs.py DIR_OR_FILES... [--G]
"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
import walker_core as C   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('paths', nargs='+')
ap.add_argument('--G', action='store_true', help='also build the stage rows / G on each walk (needs JAX)')
ap.add_argument('--params', default='runs/walker_nom_s0/params.pkl')
a = ap.parse_args()
files = []
for p in a.paths:
    files += sorted(glob.glob(os.path.join(p, '*.npz'))) if os.path.isdir(p) else [p]
files = [f for f in files if not os.path.basename(f).startswith(('VOID_',)) and not f.endswith('.part.npz')]
if a.G:
    from ilc_go1lla import walkenv as WE, srb as S
    nom = WE.Walker(a.params, stand_at_goal=True)
for f in files:
    z = np.load(f); meta = json.loads(str(z['meta']))
    n = len(z['t'])
    msg = [f'{os.path.basename(f)}: {n} steps, status {meta.get("status", "?")}, abort {meta.get("abort")}']
    if 't_wall' in z:
        dt = np.diff(z['t_wall']) * 1000
        msg.append(f'period {dt.mean():.1f} ms (p99 {np.percentile(dt, 99):.1f}, max {dt.max():.1f})')
    if 'w_mocap' in z:
        g, wm = z['v'][:, 3:6].copy(), z['w_mocap']
        tau_f = float(meta.get('mocap_tau', 0.02))                 # the mocap twist is EMA-filtered: filter the gyro alike
        for k in range(1, len(g)):
            g[k] = g[k - 1] + C.DT / (tau_f + C.DT) * (g[k] - g[k - 1])
        ok = np.all(np.isfinite(wm), 1)
        corr = [np.corrcoef(g[ok, i], wm[ok, i])[0, 1] if np.std(wm[ok, i]) > 1e-3 else np.nan for i in range(3)]
        scale = [np.dot(g[ok, i], wm[ok, i]) / max(np.dot(wm[ok, i], wm[ok, i]), 1e-9) for i in range(3)]
        msg.append('gyro vs mocap w: corr ' + ' '.join(f'{c:+.2f}' for c in corr) + ', scale ' + ' '.join(f'{s:+.2f}' for s in scale)
                   + ('  !!! check the IMU axes' if np.nanmin(corr) < 0.7 else ''))
    # torque convention: spring part of the PD from the command vs the measured torque (after the step)
    q_tgt = C.HOME + C.ACTION_SCALE * z['a']
    tau_cmd = np.clip(C.KP * (q_tgt - z['q_after']), -C.TAU_MAX, C.TAU_MAX)
    tq = z['tau_now']; qd = z['qd']
    for tag, tau in (('raw', np.r_[tq[1:], tq[-1:]]), ('+KD dq', np.r_[(tq + C.KD * qd)[1:], (tq + C.KD * qd)[-1:]])):
        msg.append(f'tau {tag} vs Kp(q*-q): rms {np.sqrt(np.mean((tau - tau_cmd) ** 2)):.2f} N m')
    msg.append(f'meta tau_includes_kd {meta.get("tau_includes_kd")}')
    print('\n    '.join(msg))
    if a.G:
        sys.argv = ['x']   # stage_goal_hw's helpers are not importable without its argparse; redo the conversion here
        tau_act = tq + (C.KD * qd if meta.get('tau_includes_kd', True) else 0.0)
        out = dict(p=z['p'], quat=z['quat'], qj=z['qj'], v=z['v'], qd=qd, cmd=z['cmd'], a=z['a'], contact=z['contact'],
                   obs_state=z['obs_state'], tau=np.r_[tau_act[1:], tau_act[-1:]], q_after=z['q_after'])
        out = {k: np.asarray(v, np.float32) for k, v in out.items()}
        g = z['goal'].astype(np.float32)
        nu = ((min(len(out['p']), int(C.handover_time(g) / C.DT)) - 1) // 10) * 10 + 1
        rows, G = S.make_G(nom, {k: v[:nu] for k, v in out.items()}, g, W=10)
        G = np.asarray(G)
        print(f'    G {G.shape}, finite {np.isfinite(G).all()}, |G| {np.abs(G).max():.2e}; rows rms {np.sqrt(np.mean(np.asarray(rows) ** 2)):.2f}')
