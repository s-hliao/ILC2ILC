#!/usr/bin/env python3
"""
scheduler_match.py (generated from scheduler.py): the real-data matching study of the F1TENTH ILC2Real matrix (LLA-MPC Fiala car: Fiala TO + Fiala nominal sim /
linearization, multi-body "real" cars), as a dependency scheduler. Each job: a shell command, the files it needs,
the file that marks it done, and a resource class -- 'gpu' (sim stages / RL training: at most GPU_JOBS at a time,
spread over the 4 GPUs), 'cpu' (hardware-stage arms and CPU-heavy evaluations: at most CPU_JOBS at a time, each
F1T_PROCS workers), 'host' (light, single process). A job starts when its needs exist and a slot is free; a failed
job is retried once. Re-runnable: jobs whose done-file exists are skipped.
State: logs/scheduler_match.log (events), logs/scheduler_match_status.json (per job: waiting / running / done / failed).
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
GPU_JOBS, CPU_JOBS, CPU_PROCS = 2, 1, 8

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


# ---- the data-matching study (user, 2026-10-09): the baselines trained ON THE REAL CARS with privileged data, from
# their sim-trained networks, up to ~2000 episodes (laps) per car -- how much real data does each need to match our
# best model after the hardware stage (ours + DR + 24 laps)? Checkpoints runs/hw/<arm>_ep<K>, K = 0 .. 2016.
CK = [24, 48, 96, 192, 384, 768, 1536, 2016]
done_of = lambda arm: f'runs/hw/{arm}_ep{CK[-1]}/summary.json'
job('real:ppo', f'F1T_PROCS={CPU_PROCS} {PY} -u ppo_real_car.py --init runs/ppo_dr_s0/policy_final.npz --arm ppo_real '
    f'--gpu {{gpu}} > logs/real_ppo.log 2>&1', done_of('ppo_real'), ['runs/ppo_dr_s0/policy_final.npz'], 'cpu')
job('real:rma', f'F1T_PROCS={CPU_PROCS} {PY} -u ppo_real_car.py --init runs/rma_s0/policy_final.npz --arm rma_real '
    f'--gpu {{gpu}} > logs/real_rma.log 2>&1', done_of('rma_real'), ['runs/rma_s0/policy_final.npz'], 'cpu')
for lam in ('0.5', '0.8', '0.95'):
    tag = 'fada_real_l' + lam.replace('.', '')
    job(f'real:{tag}', f'F1T_PROCS={CPU_PROCS} {PY} -u fada_real_car.py --idm runs/hw/fada_hw24/idm.npz --arm {tag} '
        f'--lam {lam} --gpu {{gpu}} > logs/real_{tag}.log 2>&1', done_of(tag), ['runs/hw/fada_hw24/idm.npz'], 'cpu')
# the oracle planner's source: the best of our learner + DR after the hardware stage (1-lap or chained trials), by
# its own evaluation's full-drift success
DRA = ['v3drA_s0_hw24', 'v3drB_s0_hw24', 'v3drA_s0_c2_hw24', 'v3drB_s0_c2_hw24']
job('pick_best', f"{PY} -c \"import json,numpy as np; B=['mocap_square2fast_b25','mocap_figfast_b25']; "
    f"s={{a: np.mean([v[p]['success'] for v in json.load(open('runs/hw/'+a+'/summary.json'))['eval'].values() for p in B]) "
    f"for a in {DRA}}}; best=max(s, key=s.get); open('runs/best_dr_arm.txt','w').write(best); print(s, best)\" "
    f"> logs/pick_best.log 2>&1", 'runs/best_dr_arm.txt', [f'runs/hw/{x}/summary.json' for x in DRA], 'host')
job('real:fada_real_oracle', f'F1T_PROCS={CPU_PROCS} {PY} -u fada_real_car.py --idm runs/hw/fada_hw24/idm.npz '
    f'--arm fada_real_oracle --oracle-arm $(cat runs/best_dr_arm.txt) --gpu {{gpu}} > logs/real_fada_oracle.log 2>&1',
    done_of('fada_real_oracle'), ['runs/hw/fada_hw24/idm.npz', 'runs/best_dr_arm.txt'], 'cpu')
ARMS = [f'{arm}_ep{k}' for arm in ('ppo_real', 'rma_real', 'fada_real_l05', 'fada_real_l08', 'fada_real_l095',
                                   'fada_real_oracle') for k in [0] + CK]
job('reeval_match', f'F1T_PROCS=12 {PY} -u reeval.py {" ".join(ARMS)} --runs 6 > logs/reeval_match.log 2>&1 && '
    f'touch logs/reeval_match.done', 'logs/reeval_match.done', [j['done'] for j in JOBS], 'cpu')


def main():
    status = {j['name']: 'done' if os.path.exists(j['done']) else 'waiting' for j in JOBS}
    tries = {j['name']: 0 for j in JOBS}
    running = {}                                   # name -> (Popen, kind, gpu)
    gpu_load = [0, 0, 0, 0]
    log = open('logs/scheduler_match.log', 'a')

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
                       status=status), open('logs/scheduler_match_status.json', 'w'), indent=1)
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
