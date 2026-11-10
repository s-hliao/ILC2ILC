#!/usr/bin/env python3
"""fada_tf_tests.py (2026-11-07): checks of fada_tf.py before any FADA run.
 1 numpy == JAX forward (planner, IDM);  2 LoRA: merged == base at init; merged == unmerged LoRA forward after training-like B
 3 padded history tokens do not affect the output;  4 the IDM's future self-attention is full (a later future token moves the first action)
 5 FadaPolicy (step by step) == the batch forward on the same history;  6 the planner loss through the stop-gradient IDM leaves the IDM
 7 learning: the IDM learns an inverse model of a linear system (first-action MSE), the planner (through the frozen IDM) learns to make
   the IDM output an oracle's action;  8 numpy step time"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import fada_tf as F

ok = True


def check(name, cond, info=''):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (' ' + info if info else ''), flush=True)


rng = np.random.default_rng(0)
d_o, d_c, d_a, H, K, B = 15, 42, 2, F.H_HIST, F.K_FUT, 3
P = F.init_planner(1, d_o, d_c); I = F.init_idm(2, d_o, d_a)
O = rng.standard_normal((B, H, d_o)).astype(np.float32); A = rng.standard_normal((B, H, d_a)).astype(np.float32)
C = rng.standard_normal((B, d_c)).astype(np.float32); m = np.ones((B, H), bool); m[0, :10] = False
Y = F.planner_fwd(P, O, C, m); U = F.idm_fwd(I, O, A, Y, m)
jt = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
Yj = F.planner_fwd(jt(P), jnp.asarray(O), jnp.asarray(C), jnp.asarray(m), xp=jnp)
Uj = F.idm_fwd(jt(I), jnp.asarray(O), jnp.asarray(A), Yj, jnp.asarray(m), xp=jnp)
check('1 numpy == jax (planner)', np.allclose(Y, np.asarray(Yj), atol=1e-4), f'{np.abs(Y - np.asarray(Yj)).max():.1e}')
check('1 numpy == jax (IDM)', np.allclose(U, np.asarray(Uj), atol=1e-4), f'{np.abs(U - np.asarray(Uj)).max():.1e}')
check('1 shapes', Y.shape == (B, K, d_o) and U.shape == (B, K, d_a))

L = F.init_lora(3, I)
U0 = F.idm_fwd(F.merge_lora(I, L), O, A, Y, m)
check('2 LoRA merged == base at init', np.allclose(U0, U, atol=1e-6))
L2 = jax.tree_util.tree_map(lambda x: x + 0.05 * np.random.default_rng(4).standard_normal(x.shape).astype(np.float32), L)
Um = F.idm_fwd(F.merge_lora(I, L2), O, A, Y, m)
Ul = np.asarray(F.idm_fwd(jt(I), jnp.asarray(O), jnp.asarray(A), jnp.asarray(Y), jnp.asarray(m), xp=jnp, lora=jt(L2)))
check('2 LoRA merged == unmerged LoRA forward', np.allclose(Um, Ul, atol=1e-4), f'{np.abs(Um - Ul).max():.1e}')
n_lora = sum(1 for _ in jax.tree_util.tree_leaves(L)) // 2
check('2 LoRA on every IDM Linear', n_lora == 3 + 3 * 6 + 2 * 10 + 1, f'{n_lora} adapters')

O2 = O.copy(); O2[0, :10] = 99.0; A2 = A.copy(); A2[0, :10] = -99.0
check('3 padded history ignored', np.allclose(F.idm_fwd(I, O2, A2, Y, m)[0], U[0], atol=1e-5) and
      np.allclose(F.planner_fwd(P, O2, C, m)[0], Y[0], atol=1e-5))

Y3 = Y.copy(); Y3[:, -1] += 1.0
check('4 full future attention (the last future token moves the first action)', np.abs(F.idm_fwd(I, O, A, Y3, m)[:, 0] - U[:, 0]).max() > 1e-4)

pol = F.FadaPolicy(P, I, d_a); seq_o = rng.standard_normal((40, d_o)).astype(np.float32); seq_a = rng.standard_normal((40, d_a)).astype(np.float32)
cc = rng.standard_normal(d_c).astype(np.float32); outs = []
for t in range(40):
    outs.append(pol.act(seq_o[t], cc)); pol.observe_action(seq_a[t])
t = 35
Ob = seq_o[t - H + 1:t + 1][None]; Ab = np.concatenate([seq_a[t - H:t]])[None]; mb = np.ones((1, H), bool)
ub = F.idm_fwd(I, Ob, Ab, F.planner_fwd(P, Ob, cc[None], mb), mb)[0, 0]
t2 = 5
Ob2 = np.zeros((1, H, d_o), np.float32); Ob2[0, H - t2 - 1:] = seq_o[:t2 + 1]; Ab2 = np.zeros((1, H, d_a), np.float32)
Ab2[0, H - t2:] = seq_a[:t2]; mb2 = np.zeros((1, H), bool); mb2[0, H - t2 - 1:] = True
ub2 = F.idm_fwd(I, Ob2, Ab2, F.planner_fwd(P, Ob2, cc[None], mb2), mb2)[0, 0]
check('5 FadaPolicy == batch forward (full history)', np.allclose(outs[t], ub, atol=1e-5), f'{np.abs(outs[t] - ub).max():.1e}')
check('5 FadaPolicy == batch forward (episode start, padded)', np.allclose(outs[t2], ub2, atol=1e-5), f'{np.abs(outs[t2] - ub2).max():.1e}')

Ij, Pj = jt(I), jt(P)
astar = jnp.asarray(rng.standard_normal((B, d_a)).astype(np.float32))
def lp(Pp, Ii):
    Yh = F.planner_fwd(Pp, jnp.asarray(O), jnp.asarray(C), jnp.asarray(m), xp=jnp)
    Uh = F.idm_fwd(jax.lax.stop_gradient(Ii), jnp.asarray(O), jnp.asarray(A), Yh, jnp.asarray(m), xp=jnp)
    return jnp.mean((Uh[:, 0] - astar) ** 2)
gP, gI = jax.grad(lp, argnums=(0, 1))(Pj, Ij)
check('6 planner loss: no gradient into the IDM', all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree_util.tree_leaves(gI)) and
      any(float(jnp.abs(x).max()) > 0 for x in jax.tree_util.tree_leaves(gP)))

# 7 learning on a linear system x' = Ax + Bu (o = x), oracle u* = -G x
dx, du = 4, 2
As = np.eye(dx) + 0.05 * rng.standard_normal((dx, dx)); Bs = rng.standard_normal((dx, du)) * 0.3; G = rng.standard_normal((du, dx)) * 0.3
def roll(n, T, noise):
    X = np.zeros((n, T + K + 1, dx)); Us = np.zeros((n, T + K, du)); X[:, 0] = rng.standard_normal((n, dx))
    for t in range(T + K):
        Us[:, t] = -X[:, t] @ G.T + noise * rng.standard_normal((n, du)); X[:, t + 1] = X[:, t] @ As.T + Us[:, t] @ Bs.T
    return X, Us
X, Us = roll(400, 40, 0.3)
idx_n, idx_t = np.meshgrid(np.arange(400), np.arange(H, 40), indexing='ij'); idx_n, idx_t = idx_n.ravel(), idx_t.ravel()
def batch(ix):
    n, t = idx_n[ix], idx_t[ix]
    Oh = np.stack([X[n, t - H + 1 + j] for j in range(H)], 1); Ah = np.stack([Us[n, t - H + j] for j in range(H)], 1)
    Yf = np.stack([X[n, t + 1 + j] for j in range(K)], 1)
    return [jnp.asarray(z, jnp.float32) for z in (Oh, Ah, Yf, Us[n, t], -X[n, t] @ G.T)]
I7 = jt(F.init_idm(7, dx, du)); P7 = jt(F.init_planner(8, dx, 1)); mh = jnp.ones((256, H), bool)
oi = optax.adam(3e-4); si = oi.init(I7)
@jax.jit
def stepI(Ii, s, Oh, Ah, Yf, Ut):
    l, g = jax.value_and_grad(lambda q: jnp.mean((F.idm_fwd(q, Oh, Ah, Yf, mh, xp=jnp)[:, 0] - Ut) ** 2))(Ii)
    u, s = oi.update(g, s, Ii); return optax.apply_updates(Ii, u), s, l
ls = []
for it in range(1500):
    Oh, Ah, Yf, Ut, _ = batch(rng.integers(0, len(idx_n), 256)); I7, si, l = stepI(I7, si, Oh, Ah, Yf, Ut); ls.append(float(l))
check('7 IDM learns the inverse model (first-action MSE)', np.mean(ls[-50:]) < 0.05 * np.mean(ls[:50]), f'{np.mean(ls[:50]):.4f} -> {np.mean(ls[-50:]):.5f}')
op = optax.adam(3e-4); sp = op.init(P7); c1 = jnp.zeros((256, 1))
@jax.jit
def stepP(Pp, s, Oh, Ah, As_):
    def loss(q):
        Yh = F.planner_fwd(q, Oh, c1, mh, xp=jnp)
        return jnp.mean((F.idm_fwd(jax.lax.stop_gradient(I7), Oh, Ah, Yh, mh, xp=jnp)[:, 0] - As_) ** 2)
    l, g = jax.value_and_grad(loss)(Pp); u, s = op.update(g, s, Pp); return optax.apply_updates(Pp, u), s, l
lsp = []
for it in range(1500):
    Oh, Ah, Yf, Ut, Ast = batch(rng.integers(0, len(idx_n), 256)); P7, sp, l = stepP(P7, sp, Oh, Ah, Ast); lsp.append(float(l))
base = float(np.mean([np.mean(np.asarray(batch(rng.integers(0, len(idx_n), 256))[4]) ** 2) for _ in range(5)]))
check('7 planner (through the frozen IDM) reproduces the oracle action', np.mean(lsp[-50:]) < 0.05 * base, f'{np.mean(lsp[:50]):.4f} -> {np.mean(lsp[-50:]):.5f} (oracle action var {base:.3f})')

pol8 = F.FadaPolicy(P, I, d_a); t0 = time.time()
for t in range(200):
    pol8.act(seq_o[t % 40], cc); pol8.observe_action(seq_a[t % 40])
dt = (time.time() - t0) / 200
check('8 numpy step time < 20 ms', dt < 0.02, f'{1000 * dt:.2f} ms/step')
print('ALL PASS' if ok else 'SOME FAILED')

# 9 (2026-11-07) the XLA-compiled step (FADA_JIT=1) == the numpy step, and its time
import os
os.environ['FADA_JIT'] = '1'
pj = F.FadaPolicy(P, I, d_a); outs_j = []
for t in range(40):
    outs_j.append(pj.act(seq_o[t], cc)); pj.observe_action(seq_a[t])
check('9 jit step == numpy step', np.allclose(np.stack(outs_j), np.stack(outs), atol=1e-4), f'{np.abs(np.stack(outs_j) - np.stack(outs)).max():.1e}')
t0 = time.time()
for t in range(200):
    pj.act(seq_o[t % 40], cc); pj.observe_action(seq_a[t % 40])
dtj = (time.time() - t0) / 200
check('9 jit step time < 5 ms', dtj < 0.005, f'{1000 * dtj:.2f} ms/step')
os.environ['FADA_JIT'] = '0'
print('ALL PASS' if ok else 'SOME FAILED')
