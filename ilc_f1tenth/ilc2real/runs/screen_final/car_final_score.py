"""car_final_score.py RUBBER_ROOT PLASTIC_ROOT: the publishable car table. Per network: % nerr change of the stage (it 3, 6 real laps) vs its
own zero-shot (it 0), held-out peanut CW + practised track, eval seed 701. CI: bootstrap over RUNS (network seed x stage seed; each run's
cars averaged), so a run's cars are not counted as independent; also the cell count better."""
import sys, json, glob, numpy as np
rng = np.random.default_rng(0)
def table(root, pat, f, plans, lab):
    for net, nn in (('freePPON', 'PPO+DR + stage'), ('rmaN', 'RMA + stage')):
        fs = sorted(glob.glob(f'{root}/{pat}_{net}_s*/{f}'))
        E = [json.load(open(x)) for x in fs]
        for pn in plans:
            R, cells = [], []
            for e in E:
                if pn not in e or '3' not in e[pn]: continue
                d = [100 * (e[pn]['3'][c] / e[pn]['0'][c] - 1) for c in e[pn]['0']]
                R.append(np.mean(d)); cells += d
            if not R: continue
            R = np.array(R); bs = R[rng.integers(0, len(R), (4000, len(R)))].mean(1)
            print(f'{lab:8s} {nn:15s} {pn.replace("mocap_", "").replace("_b0_free", ""):18s} {R.mean():+6.1f} % [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]  runs better {int((R < 0).sum())}/{len(R)}  cells better {int((np.array(cells) < 0).sum())}/{len(cells)}')
table(sys.argv[1], 'SL50x120*', 'screen_eval.json', ['mocap_peanutM_b0_free_x120', 'mocap_fig8R100_b0_free_x120'], 'rubber')
table(sys.argv[2], 'PLSL*', 'screen_eval_701.json', ['mocap_peanutM_b0_free_a33', 'mocap_square2fast_b0_free_a45'], 'plastic')
