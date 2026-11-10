#!/usr/bin/env python3
"""
scheduler_lowmu.py (generated from scheduler.py): the low-friction small-budget study of the F1TENTH ILC2Real matrix (LLA-MPC Fiala car: Fiala TO + Fiala nominal sim /
linearization, multi-body "real" cars), as a dependency scheduler. Each job: a shell command, the files it needs,
the file that marks it done, and a resource class -- 'gpu' (sim stages / RL training: at most GPU_JOBS at a time,
spread over the 4 GPUs), 'cpu' (hardware-stage arms and CPU-heavy evaluations: at most CPU_JOBS at a time, each
F1T_PROCS workers), 'host' (light, single process). A job starts when its needs exist and a slot is free; a failed
job is retried once. Re-runnable: jobs whose done-file exists are skipped.
State: logs/scheduler_lowmu.log (events), logs/scheduler_lowmu_status.json (per job: waiting / running / done / failed).
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
GPU_JOBS, CPU_JOBS, CPU_PROCS = 2, 1, 6

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


# ---- lap-budget ablation (user, 2026-10-09: the car's real budget is small, ~4-10 laps): the hardware stage on the
# two full-drift plans (one per track) for 1..5 iterations = 2..10 laps per car, one iteration over the 6 goals (6
# laps), bigger steps at 6 laps; the per-track ILC on the same two plans at 1..5 laps per plan. Needs the main queue's
# sim-stage networks.
G2 = '--goals mocap_square2fast_b25 mocap_figfast_b25'
P = lambda n: f'runs/{n}/policy_final.npz'
# ---- the low-friction car, at the small budgets (user, 2026-10-09 evening: "try the local pace target, 2-10 lap budget
# with low-friction car"): chained 2-lap trials, one per update alternating the two b25 plans (--goal-cycle 1), 2..10
# laps per car, on all five cars (real_mu the target, the others must not regress):
#   pt90 / pt92  a LOWER global pace target, 90 / 92 % of the plan's speed (user: a lower target, not a local one):
#         at mu x 0.8 the path holds only near sqrt(0.8) ~ 89-90 % pace; (none: the lap ablation's / here for drBt)
for net in ('v3drB_s0', 'v3drBt_s0'):
    for k in (1, 2, 3, 4, 5):
        if net == 'v3drBt_s0':
            arm(f'{net}_lap{2 * k}', P(net), f'--iters {k} {G2} --goal-cycle 1')
        arm(f'{net}_lap{2 * k}_pt90', P(net), f'--iters {k} {G2} --goal-cycle 1 --pace-target 0.90')
        arm(f'{net}_lap{2 * k}_pt92', P(net), f'--iters {k} {G2} --goal-cycle 1 --pace-target 0.92')
# FADA without the actuator states in its target (the one-step inverse was reading the action off them: the IDM's
# steering output ~20x more sensitive to the requested steering than to the path) and H periods ahead
for H in (8, 4):
    job(f'fada_v2_h{H}', f'F1T_PROCS={CPU_PROCS} {PY} -u fada_car.py --out runs/hw/fada_v2_h{H}_hw24 --dims 0 1 2 3 4 '
        f'--horizon {H} --gpu {{gpu}} > logs/fada_v2_h{H}.log 2>&1', f'runs/hw/fada_v2_h{H}_hw24/summary.json',
        ['runs/bcB/policy_final.npz'], 'cpu')
ARMS = [j['name'][3:] for j in JOBS if j['name'].startswith('hw:')] + [f'fada_v2_h{H}_hw24' for H in (8, 4)] + \
       [f'fada_v2_h{H}_hw24_zs' for H in (8, 4)]
job('reeval_lowmu', f'F1T_PROCS={CPU_PROCS} {PY} -u reeval.py {" ".join(ARMS)} --runs 6 > logs/reeval_lowmu.log 2>&1 && '
    f'touch logs/reeval_lowmu.done', 'logs/reeval_lowmu.done', [j['done'] for j in JOBS], 'cpu')


def main():
    status = {j['name']: 'done' if os.path.exists(j['done']) else 'waiting' for j in JOBS}
    tries = {j['name']: 0 for j in JOBS}
    running = {}                                   # name -> (Popen, kind, gpu)
    gpu_load = [0, 0, 0, 0]
    log = open('logs/scheduler_lowmu.log', 'a')

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
                       status=status), open('logs/scheduler_lowmu_status.json', 'w'), indent=1)
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
