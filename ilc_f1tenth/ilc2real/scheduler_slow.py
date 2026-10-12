#!/usr/bin/env python3
"""
scheduler_slow.py (generated from scheduler_lowmu.py): SLOWER drift plans (vscale 0.9 / 0.8 / 0.7) for our headline car recipe + equal-speed baselines, on the F1TENTH ILC2Real matrix (LLA-MPC Fiala car: Fiala TO + Fiala nominal sim /
linearization, multi-body "real" cars), as a dependency scheduler. Each job: a shell command, the files it needs,
the file that marks it done, and a resource class -- 'gpu' (sim stages / RL training: at most GPU_JOBS at a time,
spread over the 4 GPUs), 'cpu' (hardware-stage arms and CPU-heavy evaluations: at most CPU_JOBS at a time, each
F1T_PROCS workers), 'host' (light, single process). A job starts when its needs exist and a slot is free; a failed
job is retried once. Re-runnable: jobs whose done-file exists are skipped.
State: logs/scheduler_slow.log (events), logs/scheduler_slow_status.json (per job: waiting / running / done / failed).
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
GPU_JOBS, CPU_JOBS, CPU_PROCS = 3, 3, 8

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


P = lambda n: f'runs/{n}/policy_final.npz'
# ---- SLOWER PLANS (user, 2026-10-11: "for drifting, it is ok to decrease speed to get better performance on our
# method"). The whole speed target of the drift TO scaled by 0.9 / 0.8 / 0.7 (trajopt_drift --vscale), one plan bank
# per scale (plans_mu02_tight_v90 ...; same tracks, beta homotopy, tight-track env). Success stays relative to the plan
# flown (pace >= 90 % of the SLOW plan), so the absolute speed drops with the scale. Per bank:
#   ours: the headline recipe (DR (B) sim stage, task objective) retrained on the slow bank, zs / 2 / 10 / 24 laps;
#   the full-speed headline network flown on the slow plans (no retraining) zs;
#   PPO+DR, RMA, plan LQR zero-shot on the same slow plans (equal-speed comparison).
TASK = '--err-scale 0.05 1e9 1e9 1e9 1e9 0.3'
G2 = '--goals mocap_square2fast_b25 mocap_figfast_b25'
for v in (90, 80, 70):
    bk = f'{HERE}/plans_mu02_tight_v{v}'
    E = f'F1T_PLANS={bk}'
    for t in ('square2fast', 'figfast'):
        job(f'to_v{v}_{t}', f'{E} {PY} -u trajopt_drift.py mocap_{t} --steps 0 10 14 18 21 --beta 25 --vscale {v / 100} '
            f'--out {bk} > logs/to_v{v}_{t}.log 2>&1', f'{bk}/mocap_{t}_b25.npz', (), 'host')
    job(f'lqr_v{v}', f'{E} {PY} -u cache_lqr.py > logs/lqr_v{v}.log 2>&1', f'{bk}/lqr_cache.done',
        [f'{bk}/mocap_{t}_b25.npz' for t in ('square2fast', 'figfast')], 'host')
    LQv = [f'{bk}/lqr_cache.done']
    net = f'v3drBt_v{v}_s0'
    job(net, f'{E} {PY} -u sim_train.py --out runs/{net} --plans {PL8} --dr 1.0 --bc-iters 10 {V3} --iters 30 --seed 0 '
        f'{TASK} --gpu {{gpu}} > logs/{net}.log 2>&1', P(net), LQv, 'gpu')
    arms = {f'{net}_zs': (P(net), '--iters 0'),
            f'{net}_lap2': (P(net), f'--iters 1 {G2} --goal-cycle 1'),
            f'{net}_lap10': (P(net), f'--iters 5 {G2} --goal-cycle 1'),
            f'{net}_hw24': (P(net), '--iters 2'),
            f'v3drBt_s0_on_v{v}_zs': (P('v3drBt_s0'), '--iters 0'),
            f'ppo_dr_on_v{v}_zs': (P('ppo_dr_s0'), '--iters 0'),
            f'rma_on_v{v}_zs': (P('rma_s0'), '--iters 0'),
            f'lqr_on_v{v}_zs': ('lqr', '--iters 0')}
    for nm, (pol, args) in arms.items():
        job(f'hw:{nm}', f'{E} F1T_PROCS={CPU_PROCS} {PY} -u hw_stage.py --policy {pol} --out runs/hw/{nm} {args} '
            f'--gpu {{gpu}} > logs/hw_{nm}.log 2>&1', f'runs/hw/{nm}/summary.json',
            LQv + ([pol] if pol != 'lqr' else []), 'cpu')
    job(f'reeval_v{v}', f'{E} F1T_PROCS={CPU_PROCS} {PY} -u reeval.py {" ".join(arms)} > logs/reeval_v{v}.log 2>&1 && '
        f'touch logs/reeval_v{v}.done', f'logs/reeval_v{v}.done', [f'runs/hw/{nm}/summary.json' for nm in arms], 'cpu')


def main():
    status = {j['name']: 'done' if os.path.exists(j['done']) else 'waiting' for j in JOBS}
    tries = {j['name']: 0 for j in JOBS}
    running = {}                                   # name -> (Popen, kind, gpu)
    gpu_load = [0, 0, 0, 0]
    log = open('logs/scheduler_slow.log', 'a')

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
                       status=status), open('logs/scheduler_slow_status.json', 'w'), indent=1)
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
