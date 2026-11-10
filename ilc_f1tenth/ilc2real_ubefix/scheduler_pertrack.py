#!/usr/bin/env python3
"""
scheduler_pertrack.py (generated from scheduler.py): the per-track ablation of the F1TENTH ILC2Real matrix (LLA-MPC Fiala car: Fiala TO + Fiala nominal sim /
linearization, multi-body "real" cars), as a dependency scheduler. Each job: a shell command, the files it needs,
the file that marks it done, and a resource class -- 'gpu' (sim stages / RL training: at most GPU_JOBS at a time,
spread over the 4 GPUs), 'cpu' (hardware-stage arms and CPU-heavy evaluations: at most CPU_JOBS at a time, each
F1T_PROCS workers), 'host' (light, single process). A job starts when its needs exist and a slot is free; a failed
job is retried once. Re-runnable: jobs whose done-file exists are skipped.
State: logs/scheduler_pertrack.log (events), logs/scheduler_pertrack_status.json (per job: waiting / running / done / failed).
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
GPU_JOBS, CPU_JOBS, CPU_PROCS = 4, 3, 8

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


# ---- the per-track ablation (user, 2026-10-10: "why aren't we running training on the different tracks
# independently?" -> "queue the per-track ablation"). The method keeps ONE goal-conditioned network for both tracks;
# here each track gets its OWN network, trained the same way on that track's 4 plans only (the sim stage, the BC
# warm start for the nominal learner, and the hardware stage on that track's 3 goals), evaluated on that track's 4
# eval plans with the shared networks' exact starts (reeval.py honours eval_plans). Budgets (user: at most 10 real
# laps per car): the track's b25 plan, one chained 2-lap trial per iteration, 1 / 2 iterations = 2 / 4 laps per
# network = 4 / 8 laps per car, against the shared networks' lap4 / lap8 arms (scheduler_laps.py: the two b25 plans
# alternated). merge_pertrack.py combines each square / figfast pair into runs/hw/pt_<net>_s<seed>_lap<laps per car>.
LQ = [f'{PD}/lqr_cache.done']
TR = dict(sq='square2fast', fig='figfast')
plans = lambda t, bs: ' '.join(f'mocap_{TR[t]}_b{b}' for b in bs)
P = lambda n: f'runs/{n}/policy_final.npz'
ARMS = []
for t in TR:
    PL = plans(t, (0, 10, 18, 25))
    sim(f'bcB_{t}', f'--plans {PL} {BCW} --seed 0', LQ)
    for s in (0, 1):
        sim(f'v3nom_{t}_s{s}', f'--plans {PL} --init {P(f"bcB_{t}")} {V3} --iters 30 --seed {s}', [P(f'bcB_{t}')])
        sim(f'v3drB_{t}_s{s}', f'--plans {PL} --dr 1.0 --bc-iters 10 {V3} --iters 30 --seed {s}', LQ)
        for net in (f'v3nom_{t}_s{s}', f'v3drB_{t}_s{s}'):
            hw = f'--goals mocap_{TR[t]}_b25 --goal-cycle 1 --anchor-plans {PL} --eval-plans {plans(t, (25, 21, 14, 0))}'
            for b, it in (('zs', 0), ('lap2', 1), ('lap4', 2)):     # 2 / 4 laps per network = 4 / 8 per car
                arm(f'{net}_{b}', P(net), f'--iters {it} {hw}')
                ARMS.append(f'{net}_{b}')
# the SHARED network's matching arms (scheduler_laps.py's recipe: the two b25 plans alternated, one chained 2-lap
# trial per iteration): seed 1 of DR (B) was not in the lap ablation
G2 = '--goals mocap_square2fast_b25 mocap_figfast_b25'
for k in (2, 4):
    arm(f'v3drB_s1_lap{2 * k}', 'runs/v3drB_s1/policy_final.npz', f'--iters {k} {G2} --goal-cycle 1')
    ARMS.append(f'v3drB_s1_lap{2 * k}')
job('reeval_pertrack', f'F1T_PROCS=24 {PY} -u reeval.py {" ".join(ARMS)} > logs/reeval_pertrack.log 2>&1 && '
    f'touch logs/reeval_pertrack.done', 'logs/reeval_pertrack.done', [f'runs/hw/{x}/summary.json' for x in ARMS], 'cpu')
MERGED = []
for kind in ('v3nom', 'v3drB'):
    for s in (0, 1):
        for b, tot in (('zs', 'zs'), ('lap2', 'lap4'), ('lap4', 'lap8')):
            MERGED.append(f'pt_{kind}_s{s}_{tot}')                  # named by the laps per CAR
            job(f'merge:{MERGED[-1]}', f'{PY} merge_pertrack.py {MERGED[-1]} {kind}_sq_s{s}_{b} {kind}_fig_s{s}_{b} '
                f'>> logs/merge_pertrack.log 2>&1', f'runs/hw/{MERGED[-1]}/eval_big.json', ['logs/reeval_pertrack.done'],
                'host')


def main():
    status = {j['name']: 'done' if os.path.exists(j['done']) else 'waiting' for j in JOBS}
    tries = {j['name']: 0 for j in JOBS}
    running = {}                                   # name -> (Popen, kind, gpu)
    gpu_load = [0, 0, 0, 0]
    log = open('logs/scheduler_pertrack.log', 'a')

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
                       status=status), open('logs/scheduler_pertrack_status.json', 'w'), indent=1)
        if not running and all(v in ('done', 'failed') or not all(os.path.exists(f) for f in
                                   next(j for j in JOBS if j['name'] == k)['needs']) for k, v in status.items()):
            if all(v in ('done', 'failed') for v in status.values()) or not running:
                blocked = [k for k, v in status.items() if v == 'waiting']
                say(f'idle: nothing runnable; {len(blocked)} blocked: {blocked[:10]}')
                if all(v in ('done', 'failed') for v in status.values()):
                    say('ALL_DONE')
                    open('logs/scheduler_pertrack.alldone', 'w').close()
                    return
                time.sleep(60)
        time.sleep(15)


if __name__ == '__main__':
    if '--list' in sys.argv:
        for j in JOBS:
            print(j['kind'], j['name'], '| needs', j['needs'], '| done', j['done'])
        sys.exit()
    main()
