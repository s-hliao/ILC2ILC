#!/usr/bin/env python3
"""
scheduler.py: the pogo-stick DistILC matrix as a dependency scheduler (the car's scheduler.py). Each job: a shell
command, the files it needs, the file that marks it done, a resource class -- 'gpu' (sim stages: at most GPU_JOBS,
spread over GPUs), 'cpu' (hardware-stage / evaluation arms on the numpy robots: at most CPU_JOBS, one thread each).
A failed job is retried once; jobs whose done-file exists are skipped (re-runnable).
State: logs/scheduler.log, logs/scheduler_status.json.     nohup python scheduler.py > logs/scheduler.out 2>&1 &
Other job lists: --list FILE.py (a module defining JOBS via the same job()/sim()/arm() helpers, e.g. scheduler_axes).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = os.path.expanduser('~/miniconda3/envs/pogo/bin/python')
ENV = 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false'
GPUS = [int(g) for g in os.environ.get('POGO_GPUS', '0,1,2,3').split(',')]
GPU_JOBS, CPU_JOBS = int(os.environ.get('POGO_GPU_JOBS', 8)), int(os.environ.get('POGO_CPU_JOBS', 14))
STATE = os.environ.get('POGO_STATE', 'scheduler')

JOBS = []
P = lambda n: f'runs/{n}/policy_final.npz'


def job(name, cmd, done, needs=(), kind='gpu'):
    JOBS.append(dict(name=name, cmd=cmd, done=done, needs=list(needs), kind=kind))


def sim(name, args, needs=()):
    job(name, f'{ENV} {PY} -u sim_train.py --out runs/{name} {args} --gpu {{gpu}} > logs/{name}.log 2>&1',
        P(name), needs, 'gpu')


def arm(name, policy, args='', needs=()):
    job(f'hw:{name}', f'{ENV} {PY} -u hw_stage.py --policy {policy} --out runs/hw/{name} {args} --gpu {{gpu}} '
        f'> logs/hw_{name}.log 2>&1', f'runs/hw/{name}/summary.json',
        list(needs) + ([policy] if policy != 'expert' else []), 'cpu')


def validate(name, needs):
    """the sim-only stopping rule (sim_validate.py) -> runs/<name>/policy_pick.npz"""
    job(f'val:{name}', f'{ENV} CUDA_VISIBLE_DEVICES={{gpu}} {PY} -u sim_validate.py runs/{name} > logs/val_{name}.log 2>&1',
        PK(name), needs, 'gpu')


PK = lambda n: f'runs/{n}/policy_pick.npz'
PPO = 'runs/ppo_dr_s0/policy_final.npz'


def main_jobs():
    """v2 (2026-10-13): every sim stage goes through the sim-only stopping rule (policy_pick); the hardware stage runs
    its new defaults (full GN step, no trust region; hw_stage.py). The old defaults are the _gentle / _TR ablations."""
    BC = 'runs/bc/policy_bc.npz'
    for s in (0, 1, 2):
        sim(f'nom_s{s}', f'--init {BC} --iters 40 --seed {s}', [BC])
    sim('olG_s0', f'--init {BC} --iters 40 --seed 0 --no-feedback-jac', [BC])
    sim('noexp_s0', f'--init {BC} --iters 40 --seed 0 --explore-frac 0 --chain 0', [BC])
    for s in (0, 1):
        sim(f'drB_s{s}', f'--dr 1.0 --bc-iters 8 --iters 40 --seed {s}')
        sim(f'drA_s{s}', f'--init {PK(f"nom_s{s}")} --dr 1.0 --iters 20 --seed {s}', [PK(f'nom_s{s}')])
    for nm, dr in (('ppoNom_s0', ''), ('ppoDR_s0', '--dr 1.0')):
        sim(nm, f'--init {PPO} --iters 20 --seed 0 {dr}', [PPO])
    SIMS = ['nom_s0', 'nom_s1', 'nom_s2', 'olG_s0', 'noexp_s0', 'drA_s0', 'drA_s1', 'drB_s0', 'drB_s1', 'ppoNom_s0', 'ppoDR_s0']
    for n in SIMS:
        validate(n, [P(n)])

    arm('expert_zs', 'expert', '--iters 0')
    arm('bc_zs', BC, '--iters 0')
    arm('nom_s0_final_zs', P('nom_s0'), '--iters 0', [P('nom_s0')])                 # without the stopping rule
    arm('nom_s0_final_hw24', P('nom_s0'), '--iters 4', [P('nom_s0')])
    for n in SIMS:
        arm(f'{n}_zs', PK(n), '--iters 0')
        arm(f'{n}_hw24', PK(n), '--iters 4')
    for n, pol in (('nom_s0', PK('nom_s0')), ('drB_s0', PK('drB_s0')), ('ppo_dr', PPO)):
        for it, b in ((1, 6), (2, 12), (8, 48)):
            arm(f'{n}_hw{b}', pol, f'--iters {it}')
    N0 = PK('nom_s0')
    arm('nom_s0_hw24_gentle', N0, '--iters 4 --beta 0.5 --cap 0.05 --delta 0.02 --anchor-w 1')   # the v1 defaults
    arm('nom_s0_hw24_TR', N0, '--iters 4 --anchor-w 1')
    arm('nom_s0_hw24_openG', N0, '--iters 4 --jac open')
    arm('nom_s0_hw24_flip', N0, '--iters 4 --jac flip')
    arm('nom_s0_hw24_noRB', N0, '--iters 4 --rollback 0 --crash-rollback 0')
    arm('nom_s0_hw24_pert', N0, '--iters 4 --seed 9002 --train-pert 0.3')
    arm('nom_s0_hw24_rep2', N0, '--iters 4 --seed 9003')
    arm('ppo_dr_zs', PPO, '--iters 0')
    arm('ppo_dr_hw24', PPO, '--iters 4')
    arm('rma_zs', 'runs/rma_s0/policy_final.npz', '--iters 0', ['runs/rma_s0/rma_adapt.npz'])
    job('fada', f'{ENV} {PY} -u fada.py --out runs/hw/fada --init {N0} --iters 4 --gpu {{gpu}} > logs/fada.log 2>&1',
        'runs/hw/fada_hw24/summary.json', [N0], 'cpu')


def run():
    status = {j['name']: 'done' if os.path.exists(j['done']) else 'waiting' for j in JOBS}
    tries = {j['name']: 0 for j in JOBS}
    running = {}
    gpu_load = {g: 0 for g in GPUS}
    log = open(f'logs/{STATE}.log', 'a')

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
            gpu_load[g] -= 1
            jb = next(j for j in JOBS if j['name'] == name)
            if rc == 0 and os.path.exists(jb['done']):
                status[name] = 'done'
                say(f'done   {name}')
            else:
                tries[name] += 1
                status[name] = 'waiting' if tries[name] < 2 else 'failed'
                say(f'FAILED {name} (exit {rc}, try {tries[name]})')
        n = {k: sum(1 for v in running.values() if v[1] == k) for k in ('gpu', 'cpu')}
        for jb in JOBS:
            nm = jb['name']
            if status[nm] != 'waiting' or not all(os.path.exists(f) for f in jb['needs']):
                continue
            if n[jb['kind']] >= (GPU_JOBS if jb['kind'] == 'gpu' else CPU_JOBS):
                continue
            g = min(GPUS, key=lambda i: gpu_load[i])
            gpu_load[g] += 1
            pr = subprocess.Popen(jb['cmd'].replace('{gpu}', str(g)), shell=True)
            running[nm] = (pr, jb['kind'], g)
            status[nm] = 'running'
            n[jb['kind']] += 1
            say(f'start  {nm} (gpu {g})')
        json.dump(dict(time=time.strftime('%H:%M:%S'), counts={k: sum(v == k for v in status.values())
                                                               for k in ('waiting', 'running', 'done', 'failed')},
                       status=status), open(f'logs/{STATE}_status.json', 'w'), indent=1)
        if not running:
            runnable = [k for k, v in status.items() if v == 'waiting'
                        and all(os.path.exists(f) for f in next(j for j in JOBS if j['name'] == k)['needs'])]
            if not runnable:
                blocked = [k for k, v in status.items() if v == 'waiting']
                if not blocked:
                    say('ALL_DONE')
                    return
                say(f'idle: {len(blocked)} blocked: {blocked[:12]}')
                time.sleep(60)
        time.sleep(5)


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--jobs':
        import importlib
        importlib.import_module(sys.argv[2]).add_jobs(sys.modules[__name__])
    else:
        main_jobs()
    if '--print' in sys.argv:
        for j in JOBS:
            print(j['kind'], j['name'], '| needs', j['needs'])
        sys.exit()
    run()
