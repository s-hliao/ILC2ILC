#!/usr/bin/env python3
"""
fada_real_car_tf.py --model runs/fig1/fadaT_s<s>/it8.pkl --arm NAME --cars ... --goals PLAN (2026-11-07, FADA_AUDIT_1107.md FULL FIDELITY):
FADA's few-shot target adaptation exactly as the paper (2606.28476v2 Algorithm 1 T1-T3, Eq. 4.4-4.6, Appendix B.3):
  T1  target rollouts with the pretrained FADA-zs policy only (one round; fada_real_car.py's episodes: 1.05 laps of the practised plan from a
      random arc length, start perturbation U(-0.1, 0.1)), up to max(--checkpoints) episodes per car;
  T2  for each budget K (the first K episodes; figure 2's x-axis, the real laps actually flown are recorded): windows (O^H_t, A^H_t,
      Y_exec = o_{t+1..t+K}) -> a_t as executed; ONE offline LoRA fit on the IDM (r 8, alpha 16, dropout 0.05; planner and base IDM
      frozen): 800 steps, lr 3e-4, batch 4096 (sampled with replacement), Adam (the optimizer is unspecified: declared), first-action MSE;
  T3  deploy (P, I + Delta) receding horizon -> runs/hw/<arm>_ep<K>/<car>_policy.npz (+ .pkl): ep0 = FADA-zs.
Observations as the car's policies see them (measured; the paper's "ordinary target rollouts", no privileged relabelling).
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--arm', required=True)
ap.add_argument('--cars', nargs='+', required=True)
ap.add_argument('--goals', nargs='+', required=True)
ap.add_argument('--checkpoints', type=int, nargs='+', default=[8, 24, 96, 384])
ap.add_argument('--steps', type=int, default=800)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--batch', type=int, default=4096)
ap.add_argument('--seed', type=int, default=5151)
a = ap.parse_args()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('FADA_JIT', '1')                    # the rollouts' XLA-compiled FADA step (== numpy, fada_tf_tests.py 9)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402
import optax                                     # noqa: E402

import bank                                      # noqa: E402
import real_car as rc                            # noqa: E402
import fada_tf as F                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
H, K = F.H_HIST, F.K_FUT


def windows(runs, d_o=13):
    """Every (O^H, A^H, mask, Y_exec, a_t) of the episodes (measured obs; o_t = [obs[:13], a_{t-1}])."""
    Os, As, Ms, Ys, Ts = [], [], [], [], []
    for r in runs:
        obs, act = np.asarray(r['obs']), np.asarray(r['a'])
        n = len(obs)
        if n < K + 2:
            continue
        prev = np.concatenate([np.zeros((1, 2)), act[:-1]])
        o = (np.concatenate([obs[:, :13], prev], 1) if d_o == 15 else obs[:, :13]).astype(np.float32)   # 2026-11-07 leak fix: d_o 13
        for t in range(n - K):
            lo = max(0, t - H + 1); m_ = t - lo + 1
            Oh = np.zeros((H, d_o), np.float32); Ah = np.zeros((H, 2), np.float32); mk = np.zeros(H, bool)
            Oh[H - m_:] = o[lo:t + 1]; Ah[H - m_:] = prev[lo:t + 1]; mk[H - m_:] = True
            Os.append(Oh); As.append(Ah); Ms.append(mk); Ys.append(o[t + 1:t + 1 + K]); Ts.append(act[t])
    return [np.stack(x) for x in (Os, As, Ms, Ys, Ts)]


def main():
    global H
    w = F.load(a.model)
    P, I = w['P'], w['I']
    H = F.h_hist(I)                                   # (2026-11-07) the model's own history length
    plans = {p['name']: p for p in bank.load_bank(a.goals)}
    rng = np.random.default_rng(a.seed)
    Ij, Pj = jax.tree_util.tree_map(jnp.asarray, I), jax.tree_util.tree_map(jnp.asarray, P)
    opt = optax.adam(a.lr)

    @jax.jit
    def step(L, s, key, Dv):                                  # one Adam step on a batch drawn with replacement ON the device
        kb, kd = jax.random.split(key)
        b = jax.random.randint(kb, (a.batch,), 0, Dv[4].shape[0])
        Oh, Ah, m, Y, tgt = (x[b] for x in Dv)

        def loss(q):
            return jnp.mean((F.idm_fwd(Ij, Oh, Ah, Y, m, xp=jnp, lora=q, key=kd, jax=jax)[:, 0] - tgt) ** 2)
        l, g = jax.value_and_grad(loss)(L)
        u, s = opt.update(g, s, L)
        return optax.apply_updates(L, u), s, l

    def write(k, c, Imerged, laps, extra):
        d = os.path.join(HERE, 'runs/hw', f'{a.arm}_ep{k}')
        os.makedirs(d, exist_ok=True)
        pk = os.path.join(d, f'{c}_policy.pkl')
        F.save(pk, P, Imerged, dict(model=a.model, episodes=k, real_laps=laps, **extra))
        np.savez(os.path.join(d, f'{c}_policy.npz'), mode='fada_tf', pkl=pk)

    t0 = time.time()
    pol = dict(fada_tf=True, P=P, I=I)
    N = max(a.checkpoints)
    jobs = []
    for c in a.cars:
        for e in range(N):
            pl = plans[a.goals[e % len(a.goals)]]
            jobs.append(dict(car_name=c, plan=pl, policy=pol, laps=1.05, s0=rng.uniform() * float(pl['length']),
                             pert=rng.uniform(-0.1, 0.1, 5), seed=int(rng.integers(1 << 30))))
    print('fit device', jax.devices()[0], flush=True)                          # the LoRA fits on this process's device (GPU if given);
    os.environ['JAX_PLATFORMS'] = 'cpu'                                         # the spawned rollout workers on CPU
    res = rc.run_many(jobs)                                                     # T1: one round, FADA-zs only
    summ = {}
    for ci, c in enumerate(a.cars):
        rr = res[ci * N:(ci + 1) * N]
        lapsum = np.cumsum([max(float(r['laps']), 0.0) for r in rr])
        write(0, c, I, 0.0, dict(lora=None))
        summ[c] = {}
        for k in a.checkpoints:
            Dh = windows(rr[:k], F.d_obs(I, 2))                                                # T2: one offline fit per budget, from the base
            tgt = Dh[4]
            Dv = tuple(jnp.asarray(x) for x in Dh)
            L = jax.tree_util.tree_map(jnp.asarray, F.init_lora(a.seed + k, I))
            s = opt.init(L); key = jax.random.PRNGKey(a.seed + k); ls = []
            import fcntl                                                      # (2026-11-07) one batch-4096 fit per GPU at a time
            with open(f"/tmp/fada_fit_gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'x')}.lock", 'w') as lk:   # (3 at once ran a 4090 out
                fcntl.flock(lk, fcntl.LOCK_EX)                                                                # of memory)
                for st in range(a.steps):
                    key, ks = jax.random.split(key)
                    L, s, l = step(L, s, ks, Dv)
                    ls.append(l)
                ls = [float(x) for x in ls]
            Im = F.merge_lora(I, F.to_np(L))
            write(k, c, Im, float(lapsum[k - 1]), dict(n_windows=int(len(tgt)), loss=[float(np.mean(ls[:20])), float(np.mean(ls[-20:]))]))
            summ[c][k] = dict(laps=float(lapsum[k - 1]), windows=int(len(tgt)), loss0=float(np.mean(ls[:20])), loss1=float(np.mean(ls[-20:])),
                              crashes=int(sum(bool(r['failed']) for r in rr[:k])))
        print(json.dumps(dict(car=c, **{str(k): v for k, v in summ[c].items()}, t=round(time.time() - t0, 1))), flush=True)
    for k in [0] + a.checkpoints:
        d = os.path.join(HERE, 'runs/hw', f'{a.arm}_ep{k}')
        json.dump(dict(args=vars(a), episodes=k, real_laps={c: (summ[c][k]['laps'] if k else 0.0) for c in a.cars}, fit=summ if k else None),
                  open(os.path.join(d, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
