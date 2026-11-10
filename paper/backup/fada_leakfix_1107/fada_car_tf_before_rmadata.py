#!/usr/bin/env python3
"""
fada_car_tf.py --oracle runs/fig1/rma_s<s> --out runs/fig1/fadaT_s<s> (2026-11-07, runs/screen/FADA_AUDIT_1107.md FULL FIDELITY):
FADA's source pretraining (Xie et al. 2606.28476, Algorithm 1 S1-S5, Eq. 4.2 / 4.3, B.2) for the car, with fada_tf.py's transformers.

  oracle      the privileged RL policy trained with the task reward on the source domain: figure 1's RMA teacher (ppo_dr.py --mode rma,
              the actor reading the TRUE normalized DR parameters z), the paper's pi^o
  source      sim_jax's DR family (dr_params, scale 1; the family every figure-1 arm trains in), the six figure-1 plans
  proprio o_t [dev (8), raw (5)] = 13 (2026-11-07 LEAK FIX: was + the previous executed action (2) = 15, so Y contained a_t); command c_t = the plan preview (42)
  S1          roll out the current source policy (iteration 0: the oracle -- declared; the student does not exist yet) and keep the
              realized pairs: (O^H_t, A^H_t, Y_traj = o_{t+1..t+K}) -> a_t as executed
  S2          at every visited state (with the delay buffer) the oracle rolled K steps: Y_orac, U_orac; a*_t = Pi_1(U_orac)
  S3          the IDM on the matched pairs (traj and oracle-shadow batches, 1 : 1 -- the mixing ratio is unspecified, declared), first-action
              MSE (Eq. 4.2), full teacher forcing
  S4          the planner through the stop-gradient IDM to a*_t (Eq. 4.3)
  S5          DAgger: data aggregated over iterations
  suboptimal  2x the optimal data per iteration from the oracle's intermediate checkpoints (ppo_dr.py --save-at: it 5 ... 250; the paper
              uses 20), with oracle labels a* and shadows like every source rollout (B.2)
Unspecified in the paper (declared): the DAgger iteration count (--iters 8), lanes / horizon per rollout (512 x 160, the figure-1 FADA
data rollouts), Adam 3e-4, batch 512, 2000 IDM + 2000 planner steps per iteration, the oracle shadow's failure handling (a shadow that
leaves the track is dropped). Checkpoints after every iteration -> OUT/it<k>.pkl (fada_tf.save; sim laps in its meta).
"""
import argparse
import glob
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--oracle', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--iters', type=int, default=8)
ap.add_argument('--lanes', type=int, default=512)
ap.add_argument('--T', type=int, default=160)
ap.add_argument('--steps', type=int, default=2000)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--o-prev', type=int, default=0, help='(legacy, LEAKY) 1: o_t = [obs[:13], a_{t-1}] as before the 2026-11-07 leak fix')
ap.add_argument('--arch', default='tf', choices=('tf', 'mlp'), help='(2026-11-07) mlp: the Table-8 ablation (fada_tf MLP modules), everything else identical')
ap.add_argument('--source', default='dr', choices=('dr', 'nominal'), help='(2026-11-07, user) nominal: the LABELED variant "FADA, nominal source" -- every source rollout on the nominal car (no DR); the oracle reads its z (= the nominal); everything else identical')
a = ap.parse_args()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402
import optax                                     # noqa: E402

import bank                                      # noqa: E402
import car_model as cm                           # noqa: E402
import sim_jax as sj                             # noqa: E402
import fada_tf as F                              # noqa: E402
from consts import FAIL_EY, FAIL_EPSI            # noqa: E402

H, K, DC, DA = F.H_HIST, F.K_FUT, sj.NOBS - 13, 2
DO = 15 if a.o_prev else 13          # (2026-11-07 LEAK FIX, fada_tf.d_obs) o_t = the state obs[:13]; legacy --o-prev 1 appends a_{t-1}
ZKEYS = ['m', 'muf', 'Cf', 'kt', 'tau_s', 'd_off', 'd_delay', 'i_delay']                         # ppo_dr.py
ZNOM = jnp.array([cm.NOMINAL['m'], cm.NOMINAL['muf'], 250.0, 0.0258, 0.055, 0.0, 0.5, 0.5])
ZSC = jnp.array([0.6, 0.12, 50.0, 0.005, 0.025, 0.03, 0.5, 0.5])


def zvec(p):
    return (jnp.stack([p[k].astype(jnp.float32) for k in ZKEYS], -1) - ZNOM) / ZSC


def load_actor(f):
    d = np.load(f)
    return [(jnp.array(d[f'W{i}']), jnp.array(d[f'b{i}'])) for i in range(3)]


def main():
    os.makedirs(a.out, exist_ok=True)
    if a.smoke:
        a.iters, a.lanes, a.T, a.steps = 2, 16, 40, 20
    src = json.load(open(os.path.join(a.oracle, 'args.json')))
    plans = bank.load_bank(src['plans'])
    P = sj.Plans(plans, src['plans'])
    oracle = load_actor(os.path.join(a.oracle, 'policy_final.npz'))
    subs = [load_actor(f) for f in sorted(glob.glob(os.path.join(a.oracle, 'policy_it*.npz')))
            if not f.endswith(f"policy_it{int(src['iters']):03d}.npz")]
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    Pp = jax.tree_util.tree_map(jnp.asarray, (F.init_planner_mlp if a.arch == 'mlp' else F.init_planner)(a.seed + 1, DO, DC))
    Ip = jax.tree_util.tree_map(jnp.asarray, (F.init_idm_mlp if a.arch == 'mlp' else F.init_idm)(a.seed + 2, DO, DA))
    log = open(os.path.join(a.out, 'train.log'), 'a')

    def say(**kw):
        kw['t'] = round(time.time() - T0, 1)
        print(json.dumps(kw), flush=True); log.write(json.dumps(kw) + '\n'); log.flush()

    # ---------------- one lane: the source policy flies T periods (student: FADA; else an actor reading z)
    def make_fly(kind):
        def lane(x0, i0, pid, p, actor_or_none, Pq, Iq):
            z = zvec(p)

            def body(c, _):
                x, i, failed, buf, Oh, Ah, m, prev = c
                i_new = sj.project(P, pid, i, x[:2])
                obs, err = sj.features(P, pid, i_new, x)
                o = jnp.concatenate([obs[:13], prev]) if a.o_prev else obs[:13]
                if kind == 'student':
                    Oh2 = jnp.concatenate([Oh[1:], o[None]]); Ah2 = jnp.concatenate([Ah[1:], prev[None]])
                    m2 = jnp.concatenate([m[1:], jnp.ones((1,), bool)])
                    Y = F.planner_fwd(Pq, Oh2, obs[13:], m2, xp=jnp)
                    mu = F.idm_fwd(Iq, Oh2, Ah2, Y, m2, xp=jnp)[0]
                else:
                    Oh2, Ah2, m2 = Oh, Ah, m
                    mu = sj.mlp(actor_or_none, jnp.concatenate([obs, z]))
                a_phys = mu * sj.A_SCALE + sj.A_OFF
                buf_new = jnp.stack([a_phys, buf[0]])
                d_cmd = jnp.where(p['d_delay'] > 0, buf[0][0], a_phys[0]) + p['d_off']
                i_cmd = jnp.where(p['i_delay'] > 0, buf[0][1], a_phys[1])
                x_new = sj.step(x, jnp.stack([d_cmd, i_cmd]), p)
                bad = (jnp.abs(err[0]) > FAIL_EY) | (jnp.abs(err[1]) > FAIL_EPSI) | (x[3] < 0.3) | ~jnp.all(jnp.isfinite(x_new))
                fn = failed | bad
                return (jnp.where(fn, x, x_new), i_new, fn, buf_new, Oh2, Ah2, m2, mu), (x, i_new, buf, obs, o, mu, fn)
            c0 = (x0, i0, False, jnp.zeros((2, 2)), jnp.zeros((H, DO)), jnp.zeros((H, DA)), jnp.zeros(H, bool), jnp.zeros(DA))
            _, out = jax.lax.scan(body, c0, None, length=a.T)
            return out
        return jax.jit(jax.vmap(lane, in_axes=(0, 0, 0, 0, None, None, None)))
    fly_student, fly_actor = make_fly('student'), make_fly('actor')

    # ---------------- the oracle's K-step shadow from a visited state (x, i, delay buffer)
    def shadow_one(x, i, buf, pid, p):
        z = zvec(p)

        def body(c, _):
            x, i, buf, ok, prev = c
            i_new = sj.project(P, pid, i, x[:2])
            obs, err = sj.features(P, pid, i_new, x)
            mu = sj.mlp(oracle, jnp.concatenate([obs, z]))
            a_phys = mu * sj.A_SCALE + sj.A_OFF
            buf_new = jnp.stack([a_phys, buf[0]])
            d_cmd = jnp.where(p['d_delay'] > 0, buf[0][0], a_phys[0]) + p['d_off']
            i_cmd = jnp.where(p['i_delay'] > 0, buf[0][1], a_phys[1])
            x_new = sj.step(x, jnp.stack([d_cmd, i_cmd]), p)
            i2 = sj.project(P, pid, i_new, x_new[:2])
            obs2, err2 = sj.features(P, pid, i2, x_new)
            ok2 = ok & (jnp.abs(err2[0]) <= FAIL_EY) & (jnp.abs(err2[1]) <= FAIL_EPSI) & jnp.all(jnp.isfinite(x_new))
            return (x_new, i_new, buf_new, ok2, mu), ((jnp.concatenate([obs2[:13], mu]) if a.o_prev else obs2[:13]), mu)
        (_, _, _, ok, _), (Yo, Uo) = jax.lax.scan(body, (x, i, buf, True, jnp.zeros(DA)), None, length=K)
        return Yo, Uo, ok
    shadow = jax.jit(jax.vmap(jax.vmap(shadow_one, in_axes=(0, 0, 0, None, None)), in_axes=(0, 0, 0, 0, 0)))

    def collect(kind, actor):
        nonlocal key
        pid = rng.integers(0, len(src['plans']), a.lanes)
        s0 = rng.uniform(0, 1, a.lanes) * np.asarray(P.L)[pid]
        xs, ids = [], []
        for k in range(a.lanes):
            x0, i0 = sj.start_states(P, jax.random.PRNGKey(int(rng.integers(1 << 30))), [pid[k]], [s0[k]], pert=1.0)
            xs.append(x0[0]); ids.append(i0[0])
        key, k1 = jax.random.split(key)
        p = sj.dr_params(k1, a.lanes, 1.0) if a.source == 'dr' else sj.nominal_params(a.lanes)
        fly = fly_student if kind == 'student' else fly_actor
        x, i, buf, obs, o, mu, fn = fly(jnp.stack(xs), jnp.stack(ids), jnp.array(pid), p, actor if actor is not None else oracle, Pp, Ip)
        Yo, Uo, ok = shadow(x, i, buf, jnp.array(pid), p)
        fn = np.asarray(fn)
        alive = ~fn                                   # alive after step t
        laps = float(np.sum((~fn).sum(1) * np.asarray([float(P.L[q]) / float(P.lap_time[q]) for q in pid]) * sj.DT / np.asarray(P.L)[pid]))
        return dict(o=np.asarray(o), a=np.asarray(mu), c=np.asarray(obs)[..., 13:], alive=alive, Yo=np.asarray(Yo), astar=np.asarray(Uo)[:, :, 0],
                    ok=np.asarray(ok) & alive), laps

    D = []                                            # aggregated blocks

    def sample(Bn, mode):
        """mode 'idm': half trajectory pairs, half oracle-shadow pairs; 'planner': (O, A, c) -> a*."""
        out = []
        for _ in range(Bn):
            blk = D[rng.integers(len(D))]
            L_, T_ = blk['alive'].shape
            for _try in range(20):
                l, t = rng.integers(L_), rng.integers(T_)
                if mode == 'traj':
                    if t + K < T_ and blk['alive'][l, t + K - 1] and (t == 0 or blk['alive'][l, t - 1] or True) and blk['alive'][l, t]:
                        break
                else:
                    if blk['ok'][l, t]:
                        break
            else:
                continue
            lo = max(0, t - H + 1)
            Oh = np.zeros((H, DO), np.float32); Ah = np.zeros((H, DA), np.float32); m = np.zeros(H, bool)
            n = t - lo + 1
            Oh[H - n:] = blk['o'][l, lo:t + 1]
            prevs = np.concatenate([np.zeros((1, DA)), blk['a'][l, :t]])[lo:t + 1]
            Ah[H - n:] = prevs; m[H - n:] = True
            if mode == 'traj':
                Y = blk['o'][l, t + 1:t + 1 + K]; tgt = blk['a'][l, t]
            else:
                Y = blk['Yo'][l, t]; tgt = blk['astar'][l, t]
            out.append((Oh, Ah, m, Y, tgt, blk['c'][l, t]))
        return [jnp.asarray(np.stack([b[j] for b in out])) for j in range(6)]

    oI, oP = optax.adam(a.lr), optax.adam(a.lr)
    sI, sP = oI.init(Ip), oP.init(Pp)

    @jax.jit
    def stepI(Iq, s, Oh, Ah, m, Y, tgt):
        l, g = jax.value_and_grad(lambda q: jnp.mean((F.idm_fwd(q, Oh, Ah, Y, m, xp=jnp)[:, 0] - tgt) ** 2))(Iq)
        u, s = oI.update(g, s, Iq)
        return optax.apply_updates(Iq, u), s, l

    @jax.jit
    def stepP(Pq, Iq, s, Oh, Ah, m, c, tgt):
        def loss(q):
            Yh = F.planner_fwd(q, Oh, c, m, xp=jnp)
            return jnp.mean((F.idm_fwd(jax.lax.stop_gradient(Iq), Oh, Ah, Yh, m, xp=jnp)[:, 0] - tgt) ** 2)
        l, g = jax.value_and_grad(loss)(Pq)
        u, s = oP.update(g, s, Pq)
        return optax.apply_updates(Pq, u), s, l

    sim_laps = 0.0
    for it in range(a.iters):
        kind = 'oracle' if it == 0 else 'student'
        blk, lp = collect(kind, None); sim_laps += lp; D.append(blk)
        st_alive = float(blk['alive'][:, -1].mean())
        for j in range(2):                            # suboptimal: 2x, from random intermediate checkpoints
            b2, lp2 = collect('actor', subs[rng.integers(len(subs))]); sim_laps += lp2; D.append(b2)
        li, lpn = [], []
        for s_ in range(a.steps):
            hb = a.batch // 2
            b1 = sample(hb, 'traj'); b2 = sample(a.batch - hb, 'orac')
            Oh, Ah, m, Y, tgt, _ = [jnp.concatenate([x1, x2]) for x1, x2 in zip(b1, b2)]
            Ip, sI, l = stepI(Ip, sI, Oh, Ah, m, Y, tgt); li.append(float(l))
        for s_ in range(a.steps):
            Oh, Ah, m, Y, tgt, c = sample(a.batch, 'orac')
            Pp, sP, l = stepP(Pp, Ip, sP, Oh, Ah, m, c, tgt); lpn.append(float(l))
        F.save(os.path.join(a.out, f'it{it + 1}.pkl'), Pp, Ip, dict(iter=it + 1, sim_laps=sim_laps, oracle=a.oracle, H=H, K=K, d_o=DO, d_c=DC))
        say(it=it + 1, rollout=kind, alive_end=round(st_alive, 3), idm_loss=[round(np.mean(li[:50]), 5), round(np.mean(li[-50:]), 5)],
            planner_loss=[round(np.mean(lpn[:50]), 5), round(np.mean(lpn[-50:]), 5)], sim_laps=round(sim_laps, 1), blocks=len(D))
    # the student vs the oracle on fresh DR lanes (the planner-IDM's zero-shot in the source domain)
    bs, _ = collect('student', None); bo, _ = collect('oracle', None)
    say(event='final', student_alive=float(bs['alive'][:, -1].mean()), oracle_alive=float(bo['alive'][:, -1].mean()),
        student_action_vs_oracle_rmse=float(np.sqrt(((bs['a'] - bs['astar']) ** 2)[bs['ok']].mean())))


T0 = time.time()
if __name__ == '__main__':
    main()
