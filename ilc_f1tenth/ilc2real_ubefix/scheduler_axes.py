#!/usr/bin/env python3
"""
scheduler_axes.py (generated from scheduler.py): the perturbation suite of the F1TENTH ILC2Real matrix (LLA-MPC Fiala car: Fiala TO + Fiala nominal sim /
linearization, multi-body "real" cars), as a dependency scheduler. Each job: a shell command, the files it needs,
the file that marks it done, and a resource class -- 'gpu' (sim stages / RL training: at most GPU_JOBS at a time,
spread over the 4 GPUs), 'cpu' (hardware-stage arms and CPU-heavy evaluations: at most CPU_JOBS at a time, each
F1T_PROCS workers), 'host' (light, single process). A job starts when its needs exist and a slot is free; a failed
job is retried once. Re-runnable: jobs whose done-file exists are skipped.
State: logs/scheduler_axes.log (events), logs/scheduler_axes_status.json (per job: waiting / running / done / failed).
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
P = lambda n: f'runs/{n}/policy_final.npz'
# ---- the car's perturbation suite (user, 2026-10-09 evening: the quadruped's perturbation set, for the car) -----------
#   sweeps   single-axis "real" cars (mb_car.car_config "ax:"): one parameter at a time, at several severities
#   sense    sensing / estimation perturbations (mb_car.SENSE) on each of the five real cars
#   start    slow rolling starts (0.6 m/s, zero current): the network from the start, or after a launch to 1.0 m/s
#   heldout  a held-out TRACK (make_heldout_track.py: mocap_squareH), never trained or adapted on
#   adapted  our learner + DR (B)'s hardware stage (chained 2-lap trials, 24 laps) ON each sweep / sensing car; the
#            networks adapted under one perturbation are then tested on the others (trained x tested, as tp_*)
# every evaluation: eval_axes.py (6 runs x 5 chained laps, the b25 and b0 plans of both tracks; the held-out track's own)
SWEEP = ['mass=0.8', 'mass=0.9', 'mass=1.1', 'mass=1.4', 'mu=0.7', 'mu=0.9', 'mu=1.1', 'ky=0.7', 'ky=0.85', 'ky=1.2',
         'kt=0.7', 'kt=0.85', 'kt=1.15', 'tau=0.06', 'tau=0.08', 'tau=0.1', 'd_delay=1', 'd_delay=2', 'i_delay=1',
         'i_delay=2', 'd_off=0.03', 'd_off=-0.05']
SENSE = ['mocapbad', 'latency2', 'lowrate', 'ekflag', 'noisy3']
REAL = ['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag']
EV4 = '--plans mocap_square2fast_b25 mocap_figfast_b25 mocap_square2fast_b0 mocap_figfast_b0'
NETS = ('ours_drB=arm:runs/hw/v3drB_s0_hw24 ours_drB_zs=zs:runs/v3drB_s0/policy_final.npz '
        'ours_nom=arm:runs/hw/v3nom_s0_hw24 ours_nom_zs=zs:runs/v3nom_s0/policy_final.npz '
        'ppo_dr_hw=arm:runs/hw/ppo_dr_hw24 ppo_dr_zs=zs:runs/ppo_dr_s0/policy_final.npz '
        'rma_zs=zs:runs/rma_s0/policy_final.npz lqr=lqr')
NEED = ['runs/hw/v3drB_s0_hw24/summary.json', 'runs/hw/v3nom_s0_hw24/summary.json', 'runs/hw/ppo_dr_hw24/summary.json']
ev = lambda tag, cars, extra='': job(f'ax:{tag}', f'F1T_PROCS={CPU_PROCS} {PY} -u eval_axes.py --tag {tag} --cars {cars} '
                                     f'--net {NETS} {extra} > logs/axes_{tag}.log 2>&1', f'runs/axes/{tag}.json', NEED, 'cpu')
job('to_heldout', f'{PY} make_heldout_track.py > logs/to_heldout.log 2>&1 && {PY} -u trajopt_drift.py mocap_squareH '
    f'--steps 0 10 14 18 21 --beta 25 --out {PD} >> logs/to_heldout.log 2>&1', f'{PD}/mocap_squareH_b21.npz', [], 'host')
ev('sweep', ' '.join(f"'ax:{x}'" for x in SWEEP), EV4)
ev('sense', ' '.join(f"'{c}+{s}'" for c in REAL for s in SENSE), EV4)
ev('start_direct', ' '.join(REAL), f'{EV4} --laps 3 --launch 0.6 0.6 0')
ev('start_launch', ' '.join(REAL), f'{EV4} --laps 3 --launch 0.6 1.0 15')
job('ax:heldout', f'F1T_PROCS={CPU_PROCS} {PY} -u eval_axes.py --tag heldout --cars {" ".join(REAL)} --net {NETS} '
    f'--plans mocap_squareH_b25 mocap_squareH_b21 mocap_squareH_b14 mocap_squareH_b0 > logs/axes_heldout.log 2>&1',
    'runs/axes/heldout.json', NEED + [f'{PD}/mocap_squareH_b21.npz'], 'cpu')
# our learner + DR (B)'s hardware stage ON each sweep / sensing car (one car each), then reeval'd like every arm
ADAPT = [f'ax:{x}' for x in SWEEP] + [f'real_nom+{s}' for s in SENSE]
tag_of = lambda car: car.replace('ax:', 'ax_').replace('real_nom+', 'sense_').replace('=', '').replace(',', '_')
for car in ADAPT:
    arm(f'{tag_of(car)}_drB_hw24', P('v3drB_s0'), f"--iters 2 --cars '{car}'")
ARMS = [j['name'][3:] for j in JOBS if j['name'].startswith('hw:')]
job('reeval_axes', f'F1T_PROCS={CPU_PROCS} {PY} -u reeval.py {" ".join(ARMS)} > logs/reeval_axes.log 2>&1 && '
    f'touch logs/reeval_axes.done', 'logs/reeval_axes.done', [f'runs/hw/{x}/summary.json' for x in ARMS], 'cpu')
# trained x tested: each network adapted under one perturbation, on the nominal car and every other one
TP = ['ax:mass=1.4', 'ax:mu=0.7', 'ax:ky=0.7', 'ax:kt=0.7', 'ax:d_delay=2', 'ax:i_delay=2', 'real_nom+mocapbad',
      'real_nom+latency2']
TPN = ' '.join(f"{tag_of(c)}=file:runs/hw/{tag_of(c)}_drB_hw24/{c}_policy.npz" for c in TP)
job('ax:tp_matrix', f"F1T_PROCS={CPU_PROCS} {PY} -u eval_axes.py --tag tp_matrix --cars real_nom "
    + ' '.join(f"'{c}'" for c in TP) + f" --net 'nominal_adapted=arm:runs/hw/v3drB_s0_hw24' "
    + ' '.join(f"'{n}'" for n in TPN.split()) + f" {EV4} > logs/axes_tp_matrix.log 2>&1", 'runs/axes/tp_matrix.json',
    [f'runs/hw/{tag_of(c)}_drB_hw24/summary.json' for c in TP], 'cpu')


def main():
    status = {j['name']: 'done' if os.path.exists(j['done']) else 'waiting' for j in JOBS}
    tries = {j['name']: 0 for j in JOBS}
    running = {}                                   # name -> (Popen, kind, gpu)
    gpu_load = [0, 0, 0, 0]
    log = open('logs/scheduler_axes.log', 'a')

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
                       status=status), open('logs/scheduler_axes_status.json', 'w'), indent=1)
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
