#!/usr/bin/env python3
"""
scheduler.py: the full rerun of the F1TENTH ILC2Real matrix (LLA-MPC Fiala car: Fiala TO + Fiala nominal sim /
linearization, multi-body "real" cars), as a dependency scheduler. Each job: a shell command, the files it needs,
the file that marks it done, and a resource class -- 'gpu' (sim stages / RL training: at most GPU_JOBS at a time,
spread over the 4 GPUs), 'cpu' (hardware-stage arms and CPU-heavy evaluations: at most CPU_JOBS at a time, each
F1T_PROCS workers), 'host' (light, single process). A job starts when its needs exist and a slot is free; a failed
job is retried once. Re-runnable: jobs whose done-file exists are skipped.
State: logs/scheduler.log (events), logs/scheduler_status.json (per job: waiting / running / done / failed).
    nohup python scheduler.py > logs/scheduler.out 2>&1 &
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = os.path.expanduser('~/miniconda3/envs/f1t/bin/python')
PD = os.environ.get('F1T_PLANS') or 'plans'          # the plan bank (per friction: F1T_MU / F1T_PLANS)
GPU_JOBS, CPU_JOBS, CPU_PROCS = 8, 2, 8

PL8 = ' '.join(f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (0, 10, 18, 25))
PL14 = PL8 + ' ' + ' '.join(f'mocap_{t}_b{b}_mu80' for t in ('square2fast', 'figfast') for b in (10, 18, 25))
V3 = '--anchor-w 1.0 --beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5'
BCW = '--bc-iters 10 --iters 0 --a-off 0.3 --pert 2.0 --explore-frac 0.8'

JOBS = []


def job(name, cmd, done, needs=(), kind='gpu'):
    JOBS.append(dict(name=name, cmd=cmd, done=done, needs=list(needs), kind=kind))


def sim(name, args, needs=()):
    job(name, f'{PY} -u sim_train.py --out runs/{name} {args} --gpu {{gpu}} > logs/{name}.log 2>&1',
        f'runs/{name}/policy_final.npz', needs, 'gpu')


def arm(name, policy, args='', needs=()):
    job(f'hw:{name}', f'F1T_PROCS={CPU_PROCS} {PY} -u hw_stage.py --policy {policy} --out runs/hw/{name} {args} '
        f'--gpu {{gpu}} > logs/hw_{name}.log 2>&1', f'runs/hw/{name}/summary.json',
        list(needs) + ([policy] if policy != 'lqr' else []), 'cpu')


# ---- plans: the drift homotopy (incl. the held-out 14 / 21 deg), conservative variants (the goal fallback), LQR
# gains cached once (no write races later)
for t in ('mocap_square2fast', 'mocap_figfast'):
    job(f'to_{t}', f'{PY} -u trajopt_drift.py {t} --steps 0 10 14 18 21 --beta 25 --out {PD} > logs/to_{t}.log 2>&1',
        f'{PD}/{t}_b21.npz', [], 'host')
for t in ('mocap_square2fast', 'mocap_figfast'):
    job(f'to_mu80_{t}', f'{PY} -u trajopt_drift.py {t} --steps 0 10 18 --beta 25 --mu-plan 0.8 --tag _mu80 --out {PD} '
        f'> logs/to_mu80_{t}.log 2>&1', f'{PD}/{t}_b25_mu80.npz', [f'{PD}/{t}_b21.npz'], 'host')
job('lqr_cache', f'{PY} -u cache_lqr.py > logs/lqr_cache.log 2>&1', f'{PD}/lqr_cache.done',
    [f'{PD}/mocap_square2fast_b25_mu80.npz', f'{PD}/mocap_figfast_b25_mu80.npz'], 'host')
LQ = [f'{PD}/lqr_cache.done']

# ---- sim stages (Fiala nominal sim; DR variants on randomized Fiala)
sim('bcB', f'--plans {PL8} {BCW} --seed 0', LQ)
sim('v1nom_s0', f'--plans {PL8} --bc-iters 6 --iters 20 --eval-every 10 --seed 0', LQ)            # narrow v1 recipe
for s in (0, 1, 2):
    sim(f'v3nom_s{s}', f'--plans {PL8} --init runs/bcB/policy_final.npz {V3} --iters 30 --seed {s}', ['runs/bcB/policy_final.npz'])
sim('v3olG_s0', f'--plans {PL8} --init runs/bcB/policy_final.npz {V3} --iters 30 --seed 0 --no-feedback-jac', ['runs/bcB/policy_final.npz'])
sim('v3noexp_s0', f'--plans {PL8} --init runs/bcB/policy_final.npz {V3} --iters 30 --seed 0 --explore-frac 0 --chain 0',
    ['runs/bcB/policy_final.npz'])
for s in (0, 1):
    sim(f'v3drB_s{s}', f'--plans {PL8} --dr 1.0 --bc-iters 10 {V3} --iters 30 --seed {s}', LQ)
    sim(f'v3drA_s{s}', f'--plans {PL8} --init runs/v3nom_s{s}/policy_final.npz {V3} --dr 1.0 --iters 15 --seed {s}',
        [f'runs/v3nom_s{s}/policy_final.npz'])
    sim(f'bc6_s{s}', f'--plans {PL14} {BCW} --seed {s}', LQ)
    sim(f'v6nom_s{s}', f'--plans {PL14} --init runs/bc6_s{s}/policy_final.npz {V3} --iters 30 --seed {s}',
        [f'runs/bc6_s{s}/policy_final.npz'])
# RL baselines (warm start: the narrow v1 LQR clone, as last night) and FADA
job('ppo_dr_s0', f'{PY} -u ppo_dr.py --plans {PL8} --out runs/ppo_dr_s0 --init runs/v1nom_s0/policy_bc.npz --mode ppo '
    f'--gpu {{gpu}} > logs/ppo_dr_s0.log 2>&1', 'runs/ppo_dr_s0/policy_final.npz', ['runs/v1nom_s0/policy_bc.npz'])
job('rma_s0', f'{PY} -u ppo_dr.py --plans {PL8} --out runs/rma_s0 --init runs/v1nom_s0/policy_bc.npz --mode rma '
    f'--gpu {{gpu}} > logs/rma_s0.log 2>&1', 'runs/rma_s0/rma_adapt.npz', ['runs/v1nom_s0/policy_bc.npz'])
job('fada', f'F1T_PROCS=6 {PY} -u fada_car.py --out runs/hw/fada_hw24 --init runs/bcB/policy_final.npz --gpu {{gpu}} '
    f'> logs/fada.log 2>&1', 'runs/hw/fada_hw24/summary.json', ['runs/bcB/policy_final.npz'], 'cpu')
for nm, dr in (('v5ppoNom_s0', ''), ('v5ppoDR_s0', '--dr 1.0')):
    sim(nm, f'--plans {PL8} --init runs/ppo_dr_s0/policy_final.npz {V3} --iters 20 --seed 0 {dr}', ['runs/ppo_dr_s0/policy_final.npz'])

# ---- hardware-stage / evaluation arms on the multi-body cars
arm('lqr_zs', 'lqr', '--iters 0', LQ)
job('trackilc_big', f'F1T_PROCS={CPU_PROCS} {PY} -u track_ilc.py --out runs/trackilc_big --budgets 3 4 12 --eval-runs 12 '
    f'--eval-seed 1701 --gpu {{gpu}} > logs/trackilc_big.log 2>&1', 'runs/trackilc_big/summary.json', LQ, 'cpu')
P = lambda n: f'runs/{n}/policy_final.npz'
# every hardware-stage arm flies CHAINED 2-lap trials (hw_stage.py's default; user, 2026-10-09 evening: "only do
# 2 lap chaining"): iterations x 6 goals x 2 laps, so --iters 2 / 4 / 8 = 24 / 48 / 96 real laps per car. The 1-lap
# arms are in runs_1lap/.
arm('bcB_zs', P('bcB'), '--iters 0')
arm('bcB_hw24b', P('bcB'), '--iters 2')
arm('v1narrow_hw24', P('v1nom_s0'), '--iters 2')
for s in (0, 1, 2):
    arm(f'v3nom_s{s}_zs', P(f'v3nom_s{s}'), '--iters 0')
    arm(f'v3nom_s{s}_hw24', P(f'v3nom_s{s}'), '--iters 2')
arm('v3nom_s0_hw24_s2', P('v3nom_s0'), '--iters 2 --seed 9002')
arm('v3nom_s0_hw24vs', P('v3nom_s0'), '--iters 2 --train-starts 0.5')
arm('v3nom_s0_hw48', P('v3nom_s0'), '--iters 4')
arm('v3nom_s0_hw96', P('v3nom_s0'), '--iters 8')
arm('v3nom_s0_hw96_cTR', P('v3nom_s0'), '--iters 8 --anchor-n0 1e9')
arm('v3nom_s0_hw24_openG', P('v3nom_s0'), '--iters 2 --jac open')
arm('v3nom_s0_hw24_flip', P('v3nom_s0'), '--iters 2 --jac flip')
arm('v3nom_s0_hw24_noTR', P('v3nom_s0'), '--iters 2 --anchor-w 0')
arm('v3nom_s0_hw24_noRB', P('v3nom_s0'), '--iters 2 --rollback 0')
arm('v3nom_s0_hw24_b25only', P('v3nom_s0'), '--iters 6 --goals mocap_square2fast_b25 mocap_figfast_b25')
for n in ('v3olG_s0', 'v3noexp_s0', 'v3drA_s0', 'v3drA_s1', 'v3drB_s0', 'v3drB_s1', 'v5ppoNom_s0', 'v5ppoDR_s0'):
    arm(f'{n}_zs', P(n), '--iters 0')
    arm(f'{n}_hw24', P(n), '--iters 2')
arm('ppo_dr_zs', P('ppo_dr_s0'), '--iters 0')
arm('ppo_dr_hw24', P('ppo_dr_s0'), '--iters 2')
arm('ppo_dr_hw24_s2', P('ppo_dr_s0'), '--iters 2 --seed 9002')
for k, sd in ((1, 9001), (2, 9002), (3, 9003)):
    arm(f'ppo_dr_hw48' + ('' if k == 1 else f'_s{k}'), P('ppo_dr_s0'), f'--iters 4 --seed {sd}')
arm('ppo_dr_hw96_cTR', P('ppo_dr_s0'), '--iters 8 --anchor-n0 1e9')
arm('rma_zs', P('rma_s0'), '--iters 0', ['runs/rma_s0/rma_adapt.npz'])
for s in (0, 1):
    arm(f'v6nom_s{s}_zs', P(f'v6nom_s{s}'), '--iters 0')
    arm(f'v6nom_s{s}_hw24', P(f'v6nom_s{s}'), '--iters 2')
    arm(f'v6nom_s{s}_hw24fb', P(f'v6nom_s{s}'), '--iters 2 --fallback-suffix _mu80 --fallback-after 1')
    arm(f'v6nom_s{s}_hw24fb3', P(f'v6nom_s{s}'), '--iters 2 --fallback-suffix _mu80 --fallback-after 2')
arm('v6nom_s0_hw48fb3', P('v6nom_s0'), '--iters 4 --fallback-suffix _mu80 --fallback-after 2')

# ---- final: x12 re-evaluation of every arm, long chains, table, figures, GIFs (after all arms)
ARMS = [j['name'][3:] for j in JOBS if j['name'].startswith('hw:')] + ['fada_hw24', 'fada_hw24_zs']
ALL_DONE = [j['done'] for j in JOBS]
job('reeval', f'F1T_PROCS=20 {PY} -u reeval.py {" ".join(ARMS)} > logs/reeval.log 2>&1', 'logs/reeval.done', ALL_DONE, 'cpu')
JOBS[-1]['cmd'] += ' && touch logs/reeval.done'
job('long', f'for a in v3drB_s0_hw24 v3nom_s0_hw24 ppo_dr_hw48 v6nom_s1_hw24fb v3nom_s0_zs; do F1T_PROCS=12 {PY} -u long_eval.py $a --laps 20 '
    f'> logs/long_$a.log 2>&1; done && touch logs/long.done', 'logs/long.done', ['logs/reeval.done'], 'cpu')
job('final', f'{PY} table.py --big > results_table.txt && {PY} figures.py > logs/figures.log 2>&1 && touch logs/final.done',
    'logs/final.done', ['logs/long.done'], 'host')


def main():
    status = {j['name']: 'done' if os.path.exists(j['done']) else 'waiting' for j in JOBS}
    tries = {j['name']: 0 for j in JOBS}
    running = {}                                   # name -> (Popen, kind, gpu)
    gpu_load = [0, 0, 0, 0]
    log = open('logs/scheduler.log', 'a')

    def say(msg):
        log.write(time.strftime('%H:%M:%S ') + msg + '\n')
        log.flush()
    say(f'start: {len(JOBS)} jobs, {sum(v == "done" for v in status.values())} already done')
    while True:
        for name, (pr, kind, g) in list(running.items()):
            rc = pr.poll()
            if rc is None:
                continue
            del running[name]
            if g is not None:
                gpu_load[g] -= 1
            jb = next(j for j in JOBS if j['name'] == name)
            if rc == 0 and os.path.exists(jb['done']):
                status[name] = 'done'
                say(f'done   {name}')
            else:
                tries[name] += 1
                status[name] = 'waiting' if tries[name] < 2 else 'failed'
                say(f'FAILED {name} (exit {rc}, try {tries[name]}){" -> retry" if tries[name] < 2 else ""}')
        n_gpu = sum(1 for v in running.values() if v[1] == 'gpu')
        n_cpu = sum(1 for v in running.values() if v[1] == 'cpu')
        for jb in JOBS:
            nm = jb['name']
            if status[nm] != 'waiting' or not all(os.path.exists(f) for f in jb['needs']):
                continue
            if jb['kind'] == 'gpu' and n_gpu >= GPU_JOBS:
                continue
            if jb['kind'] == 'cpu' and n_cpu >= CPU_JOBS:
                continue
            g = int(min(range(4), key=lambda i: gpu_load[i]))
            if jb['kind'] in ('gpu', 'cpu'):
                gpu_load[g] += 1
            else:
                g = None
            if nm.startswith('hw:') and tries[nm] > 0:          # a half-written arm from a failed try
                out = jb['done'].rsplit('/', 1)[0]
                if out.startswith('runs/hw/') and os.path.isdir(out):
                    subprocess.call(['rm', '-rf', out])
            pr = subprocess.Popen(jb['cmd'].replace('{gpu}', str(0 if g is None else g)), shell=True)
            running[nm] = (pr, jb['kind'], g)
            status[nm] = 'running'
            n_gpu += jb['kind'] == 'gpu'
            n_cpu += jb['kind'] == 'cpu'
            say(f'start  {nm} (gpu {g})')
        json.dump(dict(time=time.strftime('%H:%M:%S'), counts={k: sum(v == k for v in status.values())
                                                               for k in ('waiting', 'running', 'done', 'failed')},
                       status=status), open('logs/scheduler_status.json', 'w'), indent=1)
        if not running and all(v in ('done', 'failed') or not all(os.path.exists(f) for f in
                                   next(j for j in JOBS if j['name'] == k)['needs']) for k, v in status.items()):
            if all(v in ('done', 'failed') for v in status.values()) or not running:
                blocked = [k for k, v in status.items() if v == 'waiting']
                say(f'idle: nothing runnable; {len(blocked)} blocked: {blocked[:10]}')
                if all(v in ('done', 'failed') for v in status.values()):
                    say('ALL_DONE')
                    return
                time.sleep(60)
        time.sleep(15)


if __name__ == '__main__':
    if '--list' in sys.argv:
        for j in JOBS:
            print(j['kind'], j['name'], '| needs', j['needs'], '| done', j['done'])
        sys.exit()
    main()
