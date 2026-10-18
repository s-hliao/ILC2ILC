"""
scheduler_unmod.py (2026-10-18, ube): the unmodelled-dynamics robot suite (real_pogo.UNMODELLED: thrust dead zone +
backlash, sensor bias, thrust lag, compliant ground, leg mass; none in the nominal model or the DR family). Validation
goals only (POGO_EVAL_SET=val), 3 seeds, every arm on the 5 robots. Outputs runs/v5/<arm>/summary.json.
  nomzs / nomhw36       ours nominal (nom_s<k>/policy_pick): zero-shot / + the default hardware stage, 36 trials
                        (6 goals x 6 iterations, also evaluated after 12 / 24)
  ppodrhw36             PPO+DR (ppo_dr_s<k>) + our hardware stage, 36 trials
  rmaR1_s<k>_zs         the fixed RMA (on-policy phase 2): rmaR1_s0, rmaR1full_s1, rmaR1full_s2
  fadaF2_s<k>_zs/_hw24  the fixed FADA (next-apex target), zero-shot and after 24 real trials
  ppo_unm_s<k>_ep0/_ep48  PPO on the robots (privileged critic) from PPO+DR seed k: ep0 = PPO+DR zero-shot, ep48
    POGO_EVAL_SET=val POGO_STATE=scheduler_unmod POGO_GPUS=0,1,2,3 POGO_CPU_JOBS=20 nohup python scheduler.py --jobs scheduler_unmod &
The reference rows on real_nom come from the existing runs (runs/v4/fix10_*, runs/v3/rmaR1_*, fadaF2_*, ppo_s*_ep48):
same goals, drops and seeds.
"""
U = '--robots real_dead real_bias real_lag real_ground real_legm'
RMA = {0: 'runs/rmaR1_s0', 1: 'runs/rmaR1full_s1', 2: 'runs/rmaR1full_s2'}


def add_jobs(S):
    def hw(name, policy, args, needs=()):
        S.job(f'hw:{name}', f'{S.ENV} {S.PY} -u hw_stage.py --policy {policy} --out runs/v5/{name} {U} {args} '
              f'--gpu {{gpu}} > logs/v5_{name}.log 2>&1', f'runs/v5/{name}/summary.json', [policy] + list(needs), 'cpu')

    for s in (0, 1, 2):
        nom, ppo = f'runs/nom_s{s}/policy_pick.npz', f'runs/ppo_dr_s{s}/policy_final.npz'
        hw(f'nomhw36_s{s}', nom, '--iters 6 --eval-at 2 4')
        hw(f'ppodrhw36_s{s}', ppo, '--iters 6 --eval-at 2 4')
        hw(f'rmaR1_s{s}_zs', f'{RMA[s]}/policy_final.npz', '--iters 0', [f'{RMA[s]}/rma_adapt.npz'])
        hw(f'nomzs_s{s}', nom, '--iters 0')
        S.job(f'fadaF2_s{s}', f'{S.ENV} {S.PY} -u fada.py --out runs/v5/fadaF2_s{s} --init {ppo} --target apex --seed {s} '
              f'--iters 4 {U} --gpu {{gpu}} > logs/v5_fadaF2_s{s}.log 2>&1', f'runs/v5/fadaF2_s{s}_hw24/summary.json', [ppo], 'cpu')
        S.job(f'ppo_unm_s{s}', f'{S.ENV} {S.PY} -u ppo_real.py --mode ppo --ppo-dir runs/ppo_dr_s{s} --seed {s} '
              f'--checkpoints 48 --out runs/v5 --tag _unm_s{s} {U} --gpu {{gpu}} > logs/v5_ppo_unm_s{s}.log 2>&1',
              f'runs/v5/ppo_unm_s{s}_ep48/summary.json', [f'runs/ppo_dr_s{s}/agent_final.npz'], 'cpu')
