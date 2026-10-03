#!/usr/bin/env python3
"""
audit_critic.py TRUTH.npz CKPT.pt VARIANT [--stage-w c1 c2 c3] [--fall-penalty F]: why the critic's action
gradients are right or wrong, one link at a time, against audit_truth.py's exact closed-loop derivatives (GPU sim)
of the critic's own return, Q_k = -sum_j>=k gamma^(j-k) 0.01 c . cost_j (falls excluded: no gradient there):

  chain-cl   the FD Jacobians chained along the jump with the policy's own feedback (the ILC's co-states,
             what co_fd trains toward): m_k = 0.01 c.grad cost_k + gamma (A_k+1 + B_k+1 K_k+1)' m_k+1, B_k' m_k
  chain-ol   the same with K = 0 (future actions held: VG with a' held, their released code)
  chain-srb  the closed-loop chain with the SRB model's A, B
  critic     -dQ/da_k: the min of the twins (as the actor sees it) and each twin
  tgt-ol     the trainer's VG target with a' held, from this critic: B_k' (grad r_k + gamma d_o' Q(o', a'))
  tgt-cl     the same through the actor (eq. 12): B_k' (grad r_k + gamma d_o' Q(o', pi(o')))

and Q's level: the critic's Q at k=0 against the true return (with the fall penalty).
Run in the dilc env (torch). Needs TRUTH_ep.npz (audit_episodes.py).
"""
import argparse
import json

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("truth")
ap.add_argument("ckpt")
ap.add_argument("variant")
ap.add_argument("--bank", default="/home/henry/ilc_ws/log/dilc/off10/bank.json")
ap.add_argument("--stage-w", type=float, nargs=3, default=[0.0, 0.0, 1.0])
ap.add_argument("--fall-penalty", type=float, default=20.0)
ap.add_argument("--quiet", action="store_true", help="one summary line")
a = ap.parse_args()
t, ep = np.load(a.truth), np.load(a.truth.replace(".npz", "_ep.npz"))
bank = json.load(open(a.bank))
gamma = float(bank.get("gamma", 0.99))
J, Nc, _ = t["act"].shape
Ndc = bank["phases"][0]
c = np.asarray(a.stage_w, float)
mask = np.ones((Nc, 4))
mask[Ndc:, :2] = 0.0
truth = np.einsum("jkas,s->jka", t["dSda_0"], c)
truth1 = np.einsum("jkas,s->jka", t["dSda_1"], c)
ok = np.isfinite(truth) & (mask[None] > 0) & ~t["fell_0"]

A, B = ep["A"].astype(float), ep["B"].astype(float)          # normalized (per unit e, per unit a)
As, Bs = ep["A_srb"].astype(float), ep["B_srb"].astype(float)
K = t["dpi"][:, :, :, :6].astype(float)                     # d a / d e, (J, Nc, 4, 6)
g = 0.01 * np.einsum("jksi,s->jki", ep["gn"].astype(float), c)   # d (0.01 c . cost_k) / d e_k+1


def chain(A, B, K):
    out = np.zeros((J, Nc, 4))
    for j in range(J):
        m = g[j, Nc - 1]
        for k in range(Nc - 1, -1, -1):
            if k < Nc - 1:
                Acl = A[j, k + 1] + (B[j, k + 1] @ K[j, k + 1] if K is not None else 0.0)
                m = g[j, k] + gamma * Acl.T @ m
            out[j, k] = B[j, k].T @ m
    return out


models = {"chain-cl": chain(A, B, K), "chain-ol": chain(A, B, None), "chain-srb": chain(As, Bs, K)}

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
names = [n for n, _ in ck["variants"]]
mi = names.index(a.variant)
W = [ck["critic"][f"ls.{i}.W"] for i in range(3)]
Bc = [ck["critic"][f"ls.{i}.b"] for i in range(3)]
Wa = {k: v[mi] for k, v in ck["actor"].items()}


def Q(o, act, twin):
    x = torch.cat([o, act], -1)
    h = torch.tanh(x @ W[0][twin] + Bc[0][twin][0])
    h = torch.tanh(h @ W[1][twin] + Bc[1][twin][0])
    return (h @ W[2][twin] + Bc[2][twin][0])[..., 0]


def pi(o):
    h = torch.relu(o @ Wa["l1.W"] + Wa["l1.b"][0])
    h = torch.relu(h @ Wa["l2.W"] + Wa["l2.b"][0])
    return torch.tanh(h @ Wa["mu.W"] + Wa["mu.b"][0])


obs = torch.tensor(t["obs"], dtype=torch.float32)
obs[..., 9:] = 0.0                                           # members without history see zeros there
act = torch.tensor(t["act"], dtype=torch.float32)
ac = act.clone().requires_grad_(True)
qs = [Q(obs[:, :Nc], ac, 2 * mi + i) for i in range(2)]
for name, q in (("critic", torch.minimum(qs[0], qs[1])), ("critic-twin0", qs[0]), ("critic-twin1", qs[1])):
    models[name] = -torch.autograd.grad(q.sum(), ac, retain_graph=True)[0].numpy()
msk_next = torch.cat([torch.tensor(mask[1:], dtype=torch.float32), torch.zeros(1, 4)])
for name, closed in (("tgt-ol", False), ("tgt-cl", True)):
    o2 = obs[:, 1:Nc + 1].clone().requires_grad_(True)
    a2 = pi(o2) * msk_next
    if not closed:
        a2 = a2.detach()
    q2 = torch.minimum(Q(o2, a2, 2 * mi), Q(o2, a2, 2 * mi + 1))
    gv = torch.autograd.grad(q2.sum(), o2)[0].numpy()[..., :6]
    lam = -g.copy()                                           # d r_k / d e_k+1 (r = -0.01 c . cost)
    lam[:, :Nc - 1] += gamma * gv[:, :Nc - 1]                  # the last transition ends the episode
    models[name] = -np.einsum("jkia,jki->jka", B, lam)

phases = [("all", slice(0, Nc)), ("k 0-9", slice(0, 10)), ("k 10-29", slice(10, Ndc)), ("k 30-49", slice(Ndc, 50)),
          ("k 50-59", slice(50, Nc))]


def rs(pred, s):
    m = ok[:, s]
    x, y = pred[:, s][m], truth[:, s][m]
    return (np.corrcoef(x, y)[0, 1] if x.std() > 0 else np.nan), (x @ y) / max(x @ x, 1e-12)


q0 = torch.minimum(*[Q(obs[:, :Nc], act, 2 * mi + i) for i in range(2)]).detach().numpy()[:, 0]
true_q0 = -(t["S0"] @ c) - a.fall_penalty * gamma ** (Nc - 1) * t["fell"]
qcorr = np.corrcoef(q0, true_q0)[0, 1] if q0.std() > 0 and true_q0.std() > 0 else np.nan
if a.quiet:
    print(f"critic r {rs(models['critic'], phases[0][1])[0]:+.2f} (chain-cl {rs(models['chain-cl'], phases[0][1])[0]:+.2f}), "
          f"by phase " + " ".join(f"{n} {rs(models['critic'], s)[0]:+.2f}" for n, s in phases[1:])
          + f"; Q(k=0) {q0.mean():+.2f} vs true {true_q0.mean():+.2f}, r {qcorr:+.2f}")
    raise SystemExit
lin = ok & np.isfinite(truth1)
print(f"truth (stage weights {c.tolist()}): {J} jumps, {int(t['fell'].sum())} fell, {ok.sum()} sample-channels; "
      f"eps 0.05 vs 0.01 r {np.corrcoef(truth[lin], truth1[lin])[0, 1]:.3f}; perturbations that fell "
      f"{t['fell_0'][np.broadcast_to(mask[None] > 0, t['fell_0'].shape)].mean():.1%}")
print(f"{'model':13s} " + " ".join(f"{n:>15s}" for n, _ in phases) + "   (r / slope vs truth)")
for name, pred in models.items():
    print(f"{name:13s} " + " ".join(f"{'%+.2f / %+5.2f' % rs(pred, s):>15s}" for _, s in phases))
for tn in ("tgt-ol", "tgt-cl", "chain-cl"):
    x, y = models["critic"][ok], models[tn][ok]
    print(f"critic vs {tn}: r {np.corrcoef(x, y)[0, 1]:+.2f}")
print(f"Q at k=0: critic {q0.mean():+.3f} (spread {q0.std():.3f}) vs true {true_q0.mean():+.3f} "
      f"(spread {true_q0.std():.3f}); r over jumps {qcorr:+.2f}")
