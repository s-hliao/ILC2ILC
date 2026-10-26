"""
scheduler_servo.py (2026-10-24): the whole pogo matrix RETRAINED under the servo leg (common.POGO_LEG=servo: the leg
angle is a state driven by a hip servo, it no longer teleports to the command). Run from its own working directory so
nothing of the instant-leg runs/ is touched: log/pogo_servo/ holds symlinks to these scripts plus fresh runs/ logs/.
Recipes = the frozen ones of the third test pass (NOTES "Third frozen test pass"), validation goals (seed 301):
  bc            the Raibert-cloned start (sim_train --iters 0)
  nom_s<k>      ours, nominal sim stage (40 its) -> sim_validate pick
  drNomInit_s<k> ours + DR from the nominal pick (--dr 1 --accept 2 --fbk-refresh 20, 40 its) -> pick
  ppo_dr_s<k>   PPO + DR;  rmaR1full_s<k>  RMA (on-policy phase 2);  fadaF2_s<k>  FADA (apex target, 24 trials)
  ppo_s<k>_ep*  PPO on the sticks (privileged critic), 24 .. 384 trials
  hw arms       <net>_s<k>_zs (zero-shot) and <net>_s<k>_PF (our hardware stage, 36 trials, fixed perturbed starts)
    cd log/pogo_servo && POGO_LEG=servo POGO_STATE=scheduler_servo POGO_GPUS=0,1,2 POGO_GPU_JOBS=6 POGO_CPU_JOBS=20 \
        nohup python scheduler.py --jobs scheduler_servo > logs/scheduler_servo.out 2>&1 &
"""
import common

assert common.LEG == 'servo', 'run with POGO_LEG=servo'
BC = 'runs/bc/policy_bc.npz'
PF = '--train-pert 1.0 --pert-fixed 1 --iters 6 --eval-at 2 4'


def add_jobs(S):
    def hw(name, policy, args, needs=()):
        S.job(f'hw:{name}', f'{S.ENV} {S.PY} -u hw_stage.py --policy {policy} --out runs/v3/{name} {args} --gpu {{gpu}} '
              f'> logs/v3_{name}.log 2>&1', f'runs/v3/{name}/summary.json',
              list(needs) + ([policy] if policy != 'expert' else []), 'cpu')

    def sim(name, args, needs):
        S.job(name, f'{S.ENV} {S.PY} -u sim_train.py --out runs/{name} {args} --gpu {{gpu}} > logs/{name}.log 2>&1',
              f'runs/{name}/policy_final.npz', needs, 'gpu')
        S.job(f'val:{name}', f'{S.ENV} CUDA_VISIBLE_DEVICES={{gpu}} {S.PY} -u sim_validate.py runs/{name} '
              f'> logs/val_{name}.log 2>&1', f'runs/{name}/policy_pick.npz', [f'runs/{name}/policy_final.npz'], 'gpu')

    S.job('bc', f'{S.ENV} {S.PY} -u sim_train.py --out runs/bc --iters 0 --seed 0 --gpu {{gpu}} > logs/bc.log 2>&1', BC, [], 'gpu')
    hw('expert_zs', 'expert', '--iters 0')
    for s in (0, 1, 2):
        nom, dr = f'runs/nom_s{s}/policy_pick.npz', f'runs/drNomInit_s{s}/policy_pick.npz'
        ppo, rma = f'runs/ppo_dr_s{s}/policy_final.npz', f'runs/rmaR1full_s{s}'
        sim(f'nom_s{s}', f'--init {BC} --iters 40 --seed {s}', [BC])
        sim(f'drNomInit_s{s}', f'--init {nom} --dr 1.0 --iters 40 --accept 2 --fbk-refresh 20 --seed {s}', [nom])
        S.job(f'ppo_dr_s{s}', f'{S.ENV} {S.PY} -u ppo_dr.py --out runs/ppo_dr_s{s} --mode ppo --dr 1.0 --init {BC} --seed {s} '
              f'--gpu {{gpu}} > logs/ppo_dr_s{s}.log 2>&1', ppo, [BC], 'gpu')
        S.job(f'rmaR1full_s{s}', f'{S.ENV} {S.PY} -u ppo_dr.py --out {rma} --mode rma --dr 1.0 --init {BC} --seed {s} '
              f'--adapt-iters 200 --p2-onpolicy 1 --gpu {{gpu}} > logs/rmaR1full_s{s}.log 2>&1', f'{rma}/rma_adapt.npz', [BC], 'gpu')
        for net, pol in (('nom', nom), ('drNomInit', dr), ('ppo_dr', ppo)):
            hw(f'{net}_s{s}_zs', pol, '--iters 0')
            hw(f'{net}_s{s}_PF', pol, PF)
        hw(f'rmaR1_s{s}_zs', f'{rma}/policy_final.npz', '--iters 0', [f'{rma}/rma_adapt.npz'])
        S.job(f'fadaF2_s{s}', f'{S.ENV} {S.PY} -u fada.py --out runs/v3/fadaF2_s{s} --init {ppo} --target apex --seed {s} '
              f'--iters 4 --gpu {{gpu}} > logs/v3_fadaF2_s{s}.log 2>&1', f'runs/v3/fadaF2_s{s}_hw24/summary.json', [ppo], 'cpu')
        S.job(f'ppo_real_s{s}', f'{S.ENV} {S.PY} -u ppo_real.py --mode ppo --ppo-dir runs/ppo_dr_s{s} --rma-dir {rma} --seed {s} '
              f'--checkpoints 24 48 96 192 384 --out runs/v3 --tag _s{s} --gpu {{gpu}} > logs/v3_ppo_real_s{s}.log 2>&1',
              f'runs/v3/ppo_s{s}_ep384/summary.json', [f'runs/ppo_dr_s{s}/agent_final.npz', f'{rma}/rma_adapt.npz'], 'cpu')
