"""beh_eval2.py OUT.json PLAN ITS RUN_DIR... (per-run practised PLAN, ITS e.g. 0,1,3; practice-only protocol 2026-11-08)
Original: beh_eval.py OUT.json RUN_DIR... (2026-11-08, user: grip on rubber / drift on plastic must EMERGE in sim): for each SB stage run, its it0
and it3 per-car policies flown on the protocol's practised and held-out plans (fig2 protocol: 4 starts, seed 701, 5 laps, F1T_DEV_TIME):
score, rms e_y, pace AND the behaviour -- body sideslip beta = atan2(vy, vx) from x_true: median |beta|, fraction of time |beta| > 10 deg,
and the same restricted to the corners (|plan curvature| > 0.4)."""
import os, sys, json, glob, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); os.chdir(D)
os.environ.update(F1T_DEV_TIME='1', F1T_MU='0.6', F1T_PLANS=D + '/plans_nom06', F1T_TRACKS=D + '/../lifted_linear_tire_20261009/tracks_tight',
                  F1T_SAFETY_EY='0.5', NUMBA_NUM_THREADS='1', OMP_NUM_THREADS='1', JAX_PLATFORMS='cpu', F1T_SPIN_RHO='3')
import bank, policy_runtime as pr, real_car as rc
PROTO = {'plastic': ('mocap_square2fast_b0_free_a45', 'mocap_peanutM_b0_free_a33'), 'rubber': ('mocap_fig8R100_b0_free_x120', 'mocap_peanutM_b0_free_x120')}


def main():
    out = sys.argv[1]; res = json.load(open(out)) if os.path.exists(out) else {}
    jobs, meta = [], []
    PN = sys.argv[2]; ITS = [int(i) for i in sys.argv[3].split(',')]
    plans = {p['name']: p for p in bank.load_bank([PN])}
    for d in sys.argv[4:]:
        prot = 'plastic' if '_plastic_' in d else 'rubber'
        for f in sorted(glob.glob(os.path.join(d, 'its', 'it0_*_policy.npz'))):
            car = os.path.basename(f)[4:-11]
            for it in ITS:
                pf = os.path.join(d, 'its', f'it{it}_{car}_policy.npz')
                if not os.path.exists(pf):
                    continue
                pol = pr.load_policy(pf)
                for role, pn in (('practised', PN),):
                    L = float(plans[pn]['length']); perts = np.random.default_rng(701).normal(size=(4, 5)) * 0.5
                    for r in range(4):
                        jobs.append(dict(car_name=car, plan=plans[pn], policy=pol, laps=5, s0=r / 4 * L, pert=perts[r], seed=701 + r))
                        meta.append((d, prot, car, it, role, pn))
    rs = rc.run_many(jobs, procs=int(os.environ.get('F1T_PROCS', '40')))
    acc = {}
    for (d, prot, car, it, role, pn), r in zip(meta, rs):
        p = bank.load_bank([pn])[0]; lm = rc.lap_metrics(r, p)
        x = np.asarray(r['x_true']); b = np.degrees(np.abs(np.arctan2(x[:, 4], np.maximum(x[:, 3], 0.05))))
        k = np.abs(np.asarray(p['kappa_s'])[np.asarray(r['idx'])]) > 0.4
        n = min(len(k), len(x)); k = k[:n]; x = x[:n]; b = b[:n]
        yr, dl, vx = x[:, 5], x[:, 8], np.maximum(x[:, 3], 0.3)
        kin = np.arctan(0.3302 * yr / vx)                      # kinematic (no-slip) steering for the measured yaw rate
        kc = k & (np.abs(yr) > 0.5)
        cs = float(np.mean(np.sign(dl[kc]) != np.sign(yr[kc]))) if kc.any() else np.nan
        sr = float(np.median(dl[kc] / kin[kc])) if kc.any() else np.nan
        vv = np.hypot(x[:, 3], x[:, 4]); cm = (vv > 1.0) & (np.abs(yr) > 0.5)
        bsg = np.degrees(np.arctan2(x[:, 4], np.maximum(x[:, 3], 0.05))) * np.sign(yr)      # signed: < 0 = rear out (drift)
        bkn = np.degrees(np.arctan(0.17145 * np.abs(yr) / np.maximum(vv, 0.3)))             # kinematic (grip) value
        sg = dict(sb_p50=float(np.median(bsg[cm])), sb_p10=float(np.percentile(bsg[cm], 10)), rear3=float(np.mean(bsg[cm] < -3)),
                  rear10=float(np.mean(bsg[cm] < -10)), beyond_kin=float(np.median(bkn[cm] - bsg[cm]))) if cm.any() else {}
        acc.setdefault((os.path.basename(d), prot, role, it), []).append(dict(
            score=5.0 if lm.get('failed') else max(lm['rms_ey'] / 0.1, (1 - lm['pace']) / 0.1), rms_ey=lm.get('rms_ey', np.nan), pace=lm.get('pace', np.nan),
            beta_med=float(np.median(b)), beta_p90=float(np.percentile(b, 90)), beta_gt10=float(np.mean(b > 10)), beta_gt5=float(np.mean(b > 5)), beta_med_corner=float(np.median(b[k])) if k.any() else np.nan,
            beta_gt10_corner=float(np.mean(b[k] > 10)) if k.any() else np.nan, countersteer=cs, steer_over_kin=sr, **sg))
    for (run, prot, role, it), v in sorted(acc.items()):
        res.setdefault(run, {}).setdefault(role, {})[str(it)] = {q: float(np.nanmean([e[q] for e in v])) for q in v[0]}
    json.dump(res, open(out, 'w'), indent=1)
    for run in res:
        for role in res[run]:
            for it in sorted(res[run][role], key=int)[1:]:
                a, b = res[run][role]['0'], res[run][role][it]
                print(f"{run:34s} it{it} score {a['score']:.2f}->{b['score']:.2f} | rms_ey {a['rms_ey']:.3f}->{b['rms_ey']:.3f} pace {a['pace']:.2f}->{b['pace']:.2f}"
                      f" | SIGNED beta p50 {a['sb_p50']:.1f}->{b['sb_p50']:.1f} p10 {a['sb_p10']:.1f}->{b['sb_p10']:.1f}, beyond kinematic {a['beyond_kin']:.1f}->{b['beyond_kin']:.1f},"
                      f" rear-out <-3 {100 * a['rear3']:.0f}->{100 * b['rear3']:.0f}% <-10 {100 * a['rear10']:.0f}->{100 * b['rear10']:.0f}%"
                      f" | |beta| median {a['beta_med']:.1f}->{b['beta_med']:.1f} deg, corners {a['beta_med_corner']:.1f}->{b['beta_med_corner']:.1f},"
                      f" countersteer {100 * a['countersteer']:.0f}->{100 * b['countersteer']:.0f}%, steer/kin {a['steer_over_kin']:.2f}->{b['steer_over_kin']:.2f},"
                      f" p90 {a['beta_p90']:.1f}->{b['beta_p90']:.1f}, >5 deg {100 * a['beta_gt5']:.0f}->{100 * b['beta_gt5']:.0f}%, >10 deg {100 * a['beta_gt10']:.0f}->{100 * b['beta_gt10']:.0f}%", flush=True)


if __name__ == '__main__':
    main()
