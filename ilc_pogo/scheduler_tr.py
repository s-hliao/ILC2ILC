"""
scheduler_tr.py (2026-10-17, machine ube): does ONE adaptive step rule replace the hand-set step knobs? Validation goals
(POGO_EVAL_SET unset), 36 trials (6 goals x 6 iterations, evaluated also after 12 / 24), the three starting networks x 3
seeds. Outputs runs/v4/<arm>_<net>_s<k>/.
  batch 'tr1' (2 x 2 factorial at the current default size):
     fix10      the current default: beta 1, cap 0.10, rollback 0.7, fixed refit (800 steps)
     fix10_fit  + refit the newest trials to convergence (realized >= 0.95 or plateau; age decay 0)
     tr10       adaptive trust region (rho thresholds 0.25 / 0.75, x0.5 / x2, r0 0.10), fixed refit
     tr10_fit   adaptive trust region + refit to convergence
  batch 'tr2' (sensitivity to the one remaining number, the initial size):
     fix025 / fix05 / fix20   fixed cap 0.025 / 0.05 / 0.20 (rollback 0.7)
     tr025 / tr40             adaptive, r0 0.025 / 0.40 (fixed refit; batch 1 showed refit-to-convergence with age decay
                              0 starts faster but ends 8-12 points lower)
    POGO_FIX_STAGE=tr1 POGO_STATE=scheduler_tr1 POGO_GPUS=0,1,2 POGO_CPU_JOBS=12 nohup python scheduler.py --jobs scheduler_tr &
"""
import os

STAGE = set(os.environ.get('POGO_FIX_STAGE', 'tr1').split(','))
NETS = {'nom': 'runs/nom_s{s}/policy_pick.npz', 'drLK20': 'runs/drBlsLK20_s{s}/policy_pick.npz',
        'ppo_dr': 'runs/ppo_dr_s{s}/policy_final.npz'}
FIT = '--fit-realize 0.95 --age-decay 0'      # refit the newest step to convergence
ARMS = {
    'tr1': {'fix10': '--cap 0.10', 'fix10_fit': f'--cap 0.10 {FIT}', 'tr10': '--tr 1 --cap 0.10',
            'tr10_fit': f'--tr 1 --cap 0.10 {FIT}'},
    'tr2': {'fix025': '--cap 0.025', 'fix05': '--cap 0.05', 'fix20': '--cap 0.20',
            'tr025': '--tr 1 --cap 0.025', 'tr40': '--tr 1 --cap 0.40'},     # batch 1: refit-to-convergence hurts
}


def add_jobs(S):
    for st in sorted(STAGE):
        for arm, args in ARMS.get(st, {}).items():
            for net, pat in NETS.items():
                for s in (0, 1, 2):
                    name = f'{arm}_{net}_s{s}'
                    pol = pat.format(s=s)
                    S.job(f'hw:{name}', f'{S.ENV} {S.PY} -u hw_stage.py --policy {pol} --out runs/v4/{name} --iters 6 '
                          f'--eval-at 2 4 {args} --gpu {{gpu}} > logs/v4_{name}.log 2>&1', f'runs/v4/{name}/summary.json',
                          [pol], 'cpu')
