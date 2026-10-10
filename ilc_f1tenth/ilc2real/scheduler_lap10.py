#!/usr/bin/env python3
"""
scheduler_lap10.py (generated from scheduler.py): the small-budget (2 / 10 real laps) figure set of the F1TENTH ILC2Real matrix (LLA-MPC Fiala car: Fiala TO + Fiala nominal sim /
linearization, multi-body "real" cars), as a dependency scheduler. Each job: a shell command, the files it needs,
the file that marks it done, and a resource class -- 'gpu' (sim stages / RL training: at most GPU_JOBS at a time,
spread over the 4 GPUs), 'cpu' (hardware-stage arms and CPU-heavy evaluations: at most CPU_JOBS at a time, each
F1T_PROCS workers), 'host' (light, single process). A job starts when its needs exist and a slot is free; a failed
job is retried once. Re-runnable: jobs whose done-file exists are skipped.
State: logs/scheduler_lap10.log (events), logs/scheduler_lap10_status.json (per job: waiting / running / done / failed).
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


# ---- the small-budget figure set (user, 2026-10-10: "replace the 24 lap graphs" -- the car's real budget is 2-10
# laps; scored by the strict band AND by full completion of the 5-lap chained test). Every hardware stage here: the two
# b25 plans alternated, one chained 2-lap trial per iteration (scheduler_laps.py's recipe): 1 / 5 iterations = 2 / 10
# laps per car. Our headline network: our learner + DR (B) with the task objective (v3drBt_s0). Starts after the
# per-track queue (the CPU cap).
LQ = [f'{PD}/lqr_cache.done']
P = lambda n: f'runs/{n}/policy_final.npz'
G2 = '--goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1'
WAIT = ['logs/scheduler_pertrack.alldone']
ARMS = []
for net in ('v3drA_s0', 'v3drA_s1', 'v3drB_s1', 'v3nom_s2', 'v6nom_s0'):
    for L in (2, 10):
        arm(f'{net}_lap{L}', P(net), f'--iters {L // 2} {G2}', WAIT)
        ARMS.append(f'{net}_lap{L}')
for L in (2, 10):                                    # PPO+DR, a second hardware-stage seed
    arm(f'ppo_dr_lap{L}_s2', P('ppo_dr_s0'), f'--iters {L // 2} {G2} --seed 9002', WAIT)
    ARMS.append(f'ppo_dr_lap{L}_s2')
# the perturbation suite with the 10-lap networks (eval_axes.py; tags *_l10)
SWEEP = ['mass=0.8', 'mass=0.9', 'mass=1.1', 'mass=1.4', 'mu=0.7', 'mu=0.9', 'mu=1.1', 'ky=0.7', 'ky=0.85', 'ky=1.2',
         'kt=0.7', 'kt=0.85', 'kt=1.15', 'tau=0.06', 'tau=0.08', 'tau=0.1', 'd_delay=1', 'd_delay=2', 'i_delay=1',
         'i_delay=2', 'd_off=0.03', 'd_off=-0.05']
SENSE = ['mocapbad', 'latency2', 'lowrate', 'ekflag', 'noisy3']
REAL = ['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag']
EV4 = '--plans mocap_square2fast_b25 mocap_figfast_b25 mocap_square2fast_b0 mocap_figfast_b0'
NETS = ('ours_drBt=arm:runs/hw/v3drBt_s0_lap10 ours_drBt_zs=zs:runs/v3drBt_s0/policy_final.npz '
        'ours_drB=arm:runs/hw/v3drB_s0_lap10 ours_drB_zs=zs:runs/v3drB_s0/policy_final.npz '
        'ours_nom=arm:runs/hw/v3nom_s0_lap10 ours_nom_zs=zs:runs/v3nom_s0/policy_final.npz '
        'ppo_dr_hw=arm:runs/hw/ppo_dr_lap10 ppo_dr_zs=zs:runs/ppo_dr_s0/policy_final.npz '
        'rma_zs=zs:runs/rma_s0/policy_final.npz lqr=lqr')
NEED = WAIT + [f'runs/hw/{a}/summary.json' for a in ('v3drBt_s0_lap10', 'v3drB_s0_lap10', 'v3nom_s0_lap10', 'ppo_dr_lap10')]
ev = lambda tag, cars, extra='': job(f'ax:{tag}', f'F1T_PROCS={CPU_PROCS} {PY} -u eval_axes.py --tag {tag} --cars {cars} '
                                     f'--net {NETS} {extra} > logs/axes_{tag}.log 2>&1', f'runs/axes/{tag}.json', NEED, 'cpu')
ev('sweep_l10', ' '.join(f"'ax:{x}'" for x in SWEEP), EV4)
ev('sense_l10', ' '.join(f"'{c}+{s}'" for c in REAL for s in SENSE), EV4)
ev('start_direct_l10', ' '.join(REAL), f'{EV4} --laps 3 --launch 0.6 0.6 0')
ev('start_launch_l10', ' '.join(REAL), f'{EV4} --laps 3 --launch 0.6 1.0 15')
ev('heldout_l10', ' '.join(REAL), '--plans mocap_squareH_b25 mocap_squareH_b21 mocap_squareH_b14 mocap_squareH_b0')
# ours (task-objective DR network) adapted with 10 laps ON each perturbed car, then trained x tested
ADAPT = [f'ax:{x}' for x in SWEEP] + [f'real_nom+{s}' for s in SENSE]
tag_of = lambda car: car.replace('ax:', 'ax_').replace('real_nom+', 'sense_').replace('=', '').replace(',', '_')
for car in ADAPT:
    arm(f'{tag_of(car)}_drBt_lap10', P('v3drBt_s0'), f"--iters 5 {G2} --cars '{car}'", WAIT)
    ARMS.append(f'{tag_of(car)}_drBt_lap10')
job('reeval_lap10', f'F1T_PROCS={CPU_PROCS} {PY} -u reeval.py {" ".join(ARMS)} > logs/reeval_lap10.log 2>&1 && '
    f'touch logs/reeval_lap10.done', 'logs/reeval_lap10.done', [f'runs/hw/{x}/summary.json' for x in ARMS], 'cpu')
TP = ['ax:mass=1.4', 'ax:mu=0.7', 'ax:ky=0.7', 'ax:kt=0.7', 'ax:d_delay=2', 'ax:i_delay=2', 'real_nom+mocapbad',
      'real_nom+latency2']
TPN = ' '.join(f"{tag_of(c)}=file:runs/hw/{tag_of(c)}_drBt_lap10/{c}_policy.npz" for c in TP)
job('ax:tp_matrix_l10', f"F1T_PROCS={CPU_PROCS} {PY} -u eval_axes.py --tag tp_matrix_l10 --cars real_nom "
    + ' '.join(f"'{c}'" for c in TP) + f" --net 'nominal_adapted=arm:runs/hw/v3drBt_s0_lap10' "
    + ' '.join(f"'{n}'" for n in TPN.split()) + f" {EV4} > logs/axes_tp_matrix_l10.log 2>&1",
    'runs/axes/tp_matrix_l10.json', [f'runs/hw/{tag_of(c)}_drBt_lap10/summary.json' for c in TP], 'cpu')
# 20-lap continuous chains of the 10-lap networks (and zero-shot)
LONG = ['v3drBt_s0_lap10', 'v3drB_s0_lap10', 'v3nom_s0_lap10', 'ppo_dr_lap10', 'v3drBt_s0_zs']
job('long_l10', f'for a in {" ".join(LONG)}; do [ -f runs/hw/$a/long_eval.json ] || F1T_PROCS={CPU_PROCS} {PY} -u '
    f'long_eval.py $a --laps 20 > logs/long_$a.log 2>&1; done && touch logs/long_l10.done', 'logs/long_l10.done',
    ['logs/reeval_lap10.done'], 'cpu')


def main():
    status = {j['name']: 'done' if os.path.exists(j['done']) else 'waiting' for j in JOBS}
    tries = {j['name']: 0 for j in JOBS}
    running = {}                                   # name -> (Popen, kind, gpu)
    gpu_load = [0, 0, 0, 0]
    log = open('logs/scheduler_lap10.log', 'a')

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
                       status=status), open('logs/scheduler_lap10_status.json', 'w'), indent=1)
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
