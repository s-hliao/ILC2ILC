"""
scheduler_fix.py: the 2026-10-17 round (FADA / RMA fixes, budget, seeds; plan in ~/.claude/plans). Every real-stick
evaluation scores the VALIDATION set (POGO_EVAL_SET unset = 'val'); the frozen test pass is reeval.py at the end.
Outputs runs/v3/<arm>/summary.json; sim stages runs/<name>/.
    POGO_STATE=scheduler_fix POGO_GPUS=1,2,3 nohup python scheduler.py --jobs scheduler_fix > logs/scheduler_fix.out 2>&1 &
"""
import os

STAGE = set(os.environ.get('POGO_FIX_STAGE', 'rma').split(','))   # exact stage names
RMA_T = 'runs/rma_s0/policy_final.npz'
BC_RMA = 'runs/dbg_nom/policy_bc.npz'


def add_jobs(S):
    def hw(name, policy, args='', needs=()):
        S.job(f'hw:{name}', f'{S.ENV} {S.PY} -u hw_stage.py --policy {policy} --out runs/v3/{name} {args} --gpu {{gpu}} '
              f'> logs/v3_{name}.log 2>&1', f'runs/v3/{name}/summary.json',
              list(needs) + ([policy] if policy != 'expert' else []), 'cpu')

    def rma_p2(name, args, teacher=RMA_T):
        S.job(name, f'{S.ENV} {S.PY} -u ppo_dr.py --out runs/{name} --mode rma --adapt-only --teacher {teacher} {args} '
              f'--gpu {{gpu}} > logs/{name}.log 2>&1', f'runs/{name}/rma_adapt.npz', [teacher], 'gpu')
        hw(f'{name}_zs', f'runs/{name}/policy_final.npz', '--iters 0', [f'runs/{name}/rma_adapt.npz'])

    if 'rma' in STAGE:
        # reference arms on the validation set
        hw('ppo_dr_s0_zs', 'runs/ppo_dr_s0/policy_final.npz', '--iters 0')
        hw('rma_s0_zs', RMA_T, '--iters 0', ['runs/rma_s0/rma_adapt.npz'])
        P2 = '--adapt-iters 200'
        rma_p2('rmaR1_s0', f'{P2} --p2-onpolicy 1')
        rma_p2('rmaR2_s0', f'{P2} --p2-onpolicy 1 --p2-am 10')
        rma_p2('rmaR3_s0', f'{P2} --p2-onpolicy 1 --p2-am 10 --p2-dr 1.5')
        rma_p2('rmaR3b_s0', f'{P2} --p2-onpolicy 1 --p2-am 100 --p2-dr 1.5')
        rma_p2('rmaR3c_s0', f'{P2} --p2-onpolicy 1 --p2-am 10 --p2-dr 2.0')
        # R4: a teacher trained with z-noise (robust to the estimate), then the R3 phase 2
        S.job('rmaZn_s0', f'{S.ENV} {S.PY} -u ppo_dr.py --out runs/rmaZn_s0 --mode rma --init {BC_RMA} --z-noise 0.3 '
              f'{P2} --p2-onpolicy 1 --p2-am 10 --p2-dr 1.5 --gpu {{gpu}} > logs/rmaZn_s0.log 2>&1',
              'runs/rmaZn_s0/rma_adapt.npz', [BC_RMA], 'gpu')
        hw('rmaZn_s0_zs', 'runs/rmaZn_s0/policy_final.npz', '--iters 0', ['runs/rmaZn_s0/rma_adapt.npz'])

    if 'fada' in STAGE:
        SRC = 'runs/ppo_dr_s0/policy_final.npz'
        for nm, args in (('fadaF1', '--target hstep'), ('fadaF1n', '--target hstep --real-noise 0.05'),
                         ('fadaF1H25n', '--target hstep --H 25 --real-noise 0.05'), ('fadaF2', '--target apex'),
                         ('fadaF2n', '--target apex --real-noise 0.05'), ('fadaF2n10', '--target apex --real-noise 0.1')):
            S.job(nm, f'{S.ENV} {S.PY} -u fada.py --out runs/v3/{nm}_s0 --init {SRC} {args} --iters 4 --gpu {{gpu}} '
                  f'> logs/v3_{nm}_s0.log 2>&1', f'runs/v3/{nm}_s0_hw24/summary.json', [SRC], 'cpu')

    if 'budget' in STAGE:
        # the budget curve along one run per seed: 6 goals x 1 trial per iteration, evaluated after 1..16 iterations
        SRCS = {'nom': 'runs/nom_s{s}/policy_pick.npz', 'drLK20': 'runs/drBlsLK20_s{s}/policy_pick.npz',
                'ppo_dr': 'runs/ppo_dr_s{s}/policy_final.npz'}
        for nm, pat in SRCS.items():
            for s in (0, 1, 2):
                hw(f'{nm}_s{s}_curve', pat.format(s=s), '--iters 16 --eval-at 1 2 3 4 6 8 12')

    if 'fada2' in STAGE:
        SRC = 'runs/ppo_dr_s0/policy_final.npz'
        for nm, args in (('fadaF2c', '--target apex --idm-clean 1 --real-noise 0.05'),
                         ('fadaF2c05', '--target apex --idm-clean 1 --a-off 0.05 --real-noise 0.05'),
                         ('fadaF1c05', '--target hstep --idm-clean 1 --a-off 0.05 --real-noise 0.05')):
            S.job(nm, f'{S.ENV} {S.PY} -u fada.py --out runs/v3/{nm}_s0 --init {SRC} {args} --iters 4 --gpu {{gpu}} '
                  f'> logs/v3_{nm}_s0.log 2>&1', f'runs/v3/{nm}_s0_hw24/summary.json', [SRC], 'cpu')

    if 'seeds' in STAGE:
        # chosen variants (validation set): RMA = on-policy phase 2 (R1, the DAgger fix; R2/R3 tie at 100 %);
        # FADA = next-apex target, no real-trial noise (F2: 71 -> 96 %)
        for s in (1, 2):
            S.job(f'rmaR1full_s{s}', f'{S.ENV} {S.PY} -u ppo_dr.py --out runs/rmaR1full_s{s} --mode rma --init {BC_RMA} '
                  f'--seed {s} --adapt-iters 200 --p2-onpolicy 1 --gpu {{gpu}} > logs/rmaR1full_s{s}.log 2>&1',
                  f'runs/rmaR1full_s{s}/rma_adapt.npz', [BC_RMA], 'gpu')
            hw(f'rmaR1_s{s}_zs', f'runs/rmaR1full_s{s}/policy_final.npz', '--iters 0', [f'runs/rmaR1full_s{s}/rma_adapt.npz'])
            src = f'runs/ppo_dr_s{s}/policy_final.npz'
            S.job(f'fadaF2_s{s}', f'{S.ENV} {S.PY} -u fada.py --out runs/v3/fadaF2_s{s} --init {src} --target apex --seed {s} '
                  f'--iters 4 --gpu {{gpu}} > logs/v3_fadaF2_s{s}.log 2>&1', f'runs/v3/fadaF2_s{s}_hw24/summary.json', [src], 'cpu')
        # zero-shot references, 3 seeds
        hw('expert_zs', 'expert', '--iters 0')
        for s in (0, 1, 2):
            hw(f'nom_s{s}_zs', f'runs/nom_s{s}/policy_pick.npz', '--iters 0')
            hw(f'drLK20_s{s}_zs', f'runs/drBlsLK20_s{s}/policy_pick.npz', '--iters 0')
            if s:
                hw(f'ppo_dr_s{s}_zs', f'runs/ppo_dr_s{s}/policy_final.npz', '--iters 0')
            # PPO on the robots (privileged critic), from PPO+DR seed s
            S.job(f'ppo_real_s{s}', f'{S.ENV} {S.PY} -u ppo_real.py --mode ppo --ppo-dir runs/ppo_dr_s{s} --seed {s} '
                  f'--checkpoints 24 48 96 192 384 --out runs/v3 --tag _s{s} --gpu {{gpu}} > logs/v3_ppo_real_s{s}.log 2>&1',
                  f'runs/v3/ppo_s{s}_ep384/summary.json', [f'runs/ppo_dr_s{s}/agent_final.npz'], 'cpu')
        S.job('rma_oracle_s0', f'{S.ENV} {S.PY} -u ppo_real.py --mode rma_oracle --seed 0 --checkpoints 24 --out runs/v3 '
              f'--tag _s0 --gpu {{gpu}} > logs/v3_rma_oracle_s0.log 2>&1', 'runs/v3/rma_oracle_s0_ep24/summary.json',
              ['runs/rma_s0/agent_final.npz'], 'cpu')
        # per-goal ILC: deterministic (fixed drop, deterministic robots), one run
        S.job('goalilc', f'{S.ENV} {S.PY} -u goal_ilc.py --out runs/v3/goalilc --gpu {{gpu}} > logs/v3_goalilc.log 2>&1',
              'runs/v3/goalilc_b8/summary.json', [], 'cpu')

    if 'budgetB' in STAGE:
        # the shape of a trial at the knee (~24 trials): fewer goals per iteration (same total trials), and shorter /
        # longer trials (same count, different real seconds); eval along the run for the curve
        SRCB = {'nom': 'runs/nom_s{s}/policy_pick.npz', 'ppo_dr': 'runs/ppo_dr_s{s}/policy_final.npz',
                'drLK20': 'runs/drBlsLK20_s{s}/policy_pick.npz'}
        G3 = '--goals 0.0 0.58 1.1 0.72 1.4 0.65'
        G2 = '--goals 0.0 0.58 1.1 0.72'
        for nm, pat in SRCB.items():
            for s in (0, 1, 2):
                pol = pat.format(s=s)
                hw(f'{nm}_s{s}_g3', pol, f'{G3} --iters 16 --eval-at 2 4 6 8 12')
                hw(f'{nm}_s{s}_g2', pol, f'{G2} --iters 24 --eval-at 3 6 9 12 18')
                hw(f'{nm}_s{s}_T150', pol, '--T 150 --iters 8 --eval-at 1 2 3 4 6')

    if 'drfix' in STAGE:
        # ours + DR: the stage realizes ~30 % of its predicted change on the DR nets (45 % on the nominal ones) -> is it
        # the refit (800 Adam steps, lr 1e-4) under-applying the GN step? (validation goals, 36 trials)
        for s in (0, 2):
            pol = f'runs/drBlsLK20_s{s}/policy_pick.npz'
            for nm, args in (('base', ''), ('st3k', '--steps 3000'), ('lr3', '--lr 3e-4'), ('st3klr3', '--steps 3000 --lr 3e-4')):
                hw(f'drLK20_s{s}_fix_{nm}', pol, f'--iters 6 --eval-at 2 4 {args}')
        hw('nom_s2_fix_base', 'runs/nom_s2/policy_pick.npz', '--iters 6 --eval-at 2 4')

    if 'drgrid' in STAGE:
        # ours + DR: does denser goal coverage fix the DR net's goal generalization? A 3 x 3 grid over the goal box
        # (not chosen from the validation goals) x 4 iterations = 36 trials, the same budget as 6 goals x 6 iterations
        G9 = '--goals ' + ' '.join(f'{v} {h}' for v in (0.0, 0.75, 1.5) for h in (0.55, 0.65, 0.75))
        for s in (0, 1, 2):
            hw(f'drLK20_s{s}_grid9', f'runs/drBlsLK20_s{s}/policy_pick.npz', f'{G9} --iters 4 --eval-at 2')
        hw('nom_s2_grid9', 'runs/nom_s2/policy_pick.npz', f'{G9} --iters 4 --eval-at 2')
        hw('ppo_dr_s2_grid9', 'runs/ppo_dr_s2/policy_final.npz', f'{G9} --iters 4 --eval-at 2')

    if 'drnom' in STAGE:
        # ours + DR with the DR sim stage started from the NOMINAL sim network instead of BC (ube 2026-10-17 side result:
        # 94.3 vs 91.5 % at 36 trials): the drBlsLK20 sim recipe (line search, lagged feedback), 40 iterations, the
        # sim-only stopping rule, then zero-shot and the default hardware stage (36 trials) on the validation goals
        for s in (0, 1, 2):
            n, init = f'drNomInit_s{s}', f'runs/nom_s{s}/policy_pick.npz'
            S.job(n, f'{S.ENV} {S.PY} -u sim_train.py --out runs/{n} --init {init} --dr 1.0 --iters 40 --accept 2 '
                  f'--fbk-refresh 20 --seed {s} --gpu {{gpu}} > logs/{n}.log 2>&1', f'runs/{n}/policy_final.npz', [init], 'gpu')
            S.job(f'val:{n}', f'{S.ENV} CUDA_VISIBLE_DEVICES={{gpu}} {S.PY} -u sim_validate.py runs/{n} > logs/val_{n}.log 2>&1',
                  f'runs/{n}/policy_pick.npz', [f'runs/{n}/policy_final.npz'], 'gpu')
            hw(f'drNomInit_s{s}_zs', f'runs/{n}/policy_pick.npz', '--iters 0')
            hw(f'drNomInit_s{s}_curve', f'runs/{n}/policy_pick.npz', '--iters 8 --eval-at 2 4 6')

    if 'horizon' in STAGE:
        # 2026-10-18: the hardware stage trains on 3 s trials (~6 hops) from the goal's UNPERTURBED apex but is scored on
        # 10 s runs from perturbed drops; the offset stick's failures are a slow drift after hop ~6 and a second (period-2)
        # gait reached only from perturbed starts -- neither visible to training. Longer trials and perturbed training
        # starts, on the validation goals; T1000P at 12 trials = 120 s of hopping ~ the default's 36 x 3 s = 108 s.
        NETS = {'nom': 'runs/nom_s{s}/policy_pick.npz', 'drNomInit': 'runs/drNomInit_s{s}/policy_pick.npz'}
        ARMS_H = {'T500': '--T 500 --iters 6 --eval-at 2 4', 'P': '--train-pert 1.0 --iters 6 --eval-at 2 4',
                  'T500P': '--T 500 --train-pert 1.0 --iters 6 --eval-at 2 4',
                  'T1000P': '--T 1000 --train-pert 1.0 --iters 6 --eval-at 1 2 4'}
        for net, pat in NETS.items():
            for arm, args in ARMS_H.items():
                for s in (0, 1, 2):
                    hw(f'{net}_s{s}_{arm}', pat.format(s=s), args)

    if 'pfix' in STAGE:
        # 2026-10-18: the 'horizon' perturbed arms redraw the training starts every iteration, so rollback / new-fall
        # compare the new and the kept network on different starts (T1000P: mass / spring stuck at 62 % from it4 on,
        # every later step rejected). --pert-fixed 1: one perturbed start per goal, kept across iterations (the ILC
        # repeated-initial-condition assumption); otherwise the 'horizon' arms unchanged.
        NETS = {'nom': 'runs/nom_s{s}/policy_pick.npz', 'drNomInit': 'runs/drNomInit_s{s}/policy_pick.npz'}
        ARMS_F = {'PF': '--train-pert 1.0 --pert-fixed 1 --iters 6 --eval-at 2 4',
                  'T600PF': '--T 600 --train-pert 1.0 --pert-fixed 1 --iters 6 --eval-at 2 4',
                  'T1000PF': '--T 1000 --train-pert 1.0 --pert-fixed 1 --iters 6 --eval-at 2 4'}
        for net, pat in NETS.items():
            for arm, args in ARMS_F.items():
                for s in (0, 1, 2):
                    hw(f'{net}_s{s}_{arm}', pat.format(s=s), args)
    if 'pfix2' in STAGE:
        # the fixed perturbed-start protocol for the other arm that uses our hardware stage (fairness), on validation
        for arm, args in {'PF': '--train-pert 1.0 --pert-fixed 1 --iters 6 --eval-at 2 4',
                          'T600PF': '--T 600 --train-pert 1.0 --pert-fixed 1 --iters 6 --eval-at 2 4'}.items():
            for s in (0, 1, 2):
                hw(f'ppo_dr_s{s}_{arm}', f'runs/ppo_dr_s{s}/policy_final.npz', args)
    if 'drnomT' in STAGE:
        # ours + DR, sim side of the horizon finding: the DR sim stage from the nominal net (drNomInit's recipe) on 6 s
        # training rollouts instead of 3 s (slow drifts after ~6 hops become visible to the sim stage), then the sim-only
        # stopping rule; the hardware stage follows in 'drnomT_hw' with the protocol the 'horizon' stage picks
        for s in (0, 1, 2):
            n, init = f'drNomInitT600_s{s}', f'runs/nom_s{s}/policy_pick.npz'
            S.job(n, f'{S.ENV} {S.PY} -u sim_train.py --out runs/{n} --init {init} --dr 1.0 --iters 40 --accept 2 '
                  f'--fbk-refresh 20 --T 600 --seed {s} --gpu {{gpu}} > logs/{n}.log 2>&1', f'runs/{n}/policy_final.npz', [init], 'gpu')
            S.job(f'val:{n}', f'{S.ENV} CUDA_VISIBLE_DEVICES={{gpu}} {S.PY} -u sim_validate.py runs/{n} > logs/val_{n}.log 2>&1',
                  f'runs/{n}/policy_pick.npz', [f'runs/{n}/policy_final.npz'], 'gpu')
            hw(f'drNomInitT600_s{s}_zs', f'runs/{n}/policy_pick.npz', '--iters 0')
