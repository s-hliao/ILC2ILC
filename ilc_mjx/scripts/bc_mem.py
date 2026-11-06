"""bc_mem.py --teacher POLICY.npz --mem M --out DIR [--rounds 5 --batch 2048 --steps 2000] (2026-11-06; log/dilc/plane/mem/
MEMORY_ABLATION_1106.md addendum): a fair start for a reduced-memory network. The student = the teacher with its first-layer weights on
the masked inputs zeroed (history beyond the M most recent errors; with M = 0 also the previous action); DAgger rounds: the student flies
the GPU sim under DR (the PPO batch's goals, DR and async DR), the teacher's mean action at those observations is the label, the student
is fitted by MSE on its mean action with its masked inputs held at zero (so their weights stay zero). -> DIR/policy/policy.npz + bank."""
import argparse, json, os, shutil, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True); ap.add_argument("--mem", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--batch", type=int, default=2048); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args()
os.environ.pop("ILC_OBS_MEM", None)              # the rollouts' observations carry the full history: the teacher reads it all
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax, jax.numpy as jnp
from ilc_mjx.jump import JumpEnv, Config
pdir = os.path.dirname(a.teacher)
bank = json.load(open(os.path.join(pdir, "bank.json")))
env = JumpEnv(bank, Config(est_window=0.025))
H, PA, Nc = env.H, env.PA, env.Nc
T = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(a.teacher).items() if k != "n_hidden"}
od = T["W0"].shape[1]
keep = np.ones(od, np.float32); keep[9 + 6 * a.mem:9 + 6 * H] = 0.0
if a.mem == 0 and PA:
    keep[9 + 6 * H:] = 0.0
KEEP = jnp.asarray(keep)
S = dict(T); S["W0"] = T["W0"] * KEEP[None, :]
mean = jax.jit(jax.vmap(lambda w, o: JumpEnv.actor(w, o), (None, 0)))
@jax.jit
def loss(w, O, Y):
    return jnp.mean((jax.vmap(lambda o: JumpEnv.actor(w, o * KEEP))(O) - Y) ** 2)
grad = jax.jit(jax.value_and_grad(loss))
def adam(p, g, st, lr):
    m, v, t = st; t += 1
    m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, g); v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ ** 2, v, g)
    p = jax.tree_util.tree_map(lambda q, m_, v_: q - lr * (m_ / (1 - 0.9 ** t)) / (jnp.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8), p, m, v)
    return p, (m, v, t)
rng = np.random.default_rng(a.seed); Os, Ys = [], []
def evaluate(w, tag):
    g = env.sample_plane_goals(1024, np.random.default_rng(777)); d = env.dyn_arrays(env.sample_dyn(1024, np.random.default_rng(12345))); d.update(env.sample_async(1024, np.random.default_rng(12346)))
    o = env.rollout(w, env.references(g), jax.random.PRNGKey(0), stochastic=False, dyn=d)
    X = np.asarray(o["X"])[:, env.N, :2]; R = np.asarray(env.references(g)["x_ref"])
    tgt = R[:, 0, :2] + g
    err = np.abs(X - tgt); print(f"{tag}: falls {np.mean(o['fell']):.1%}, landing |ex| median {100 * np.median(err[:, 0]):.1f} cm |ez| {100 * np.median(err[:, 1]):.1f} cm", flush=True)
evaluate(T, "teacher"); evaluate(S, "student start")
for r in range(a.rounds):
    goals = env.sample_plane_goals(a.batch, rng); ref = env.references(goals)
    dyn = env.dyn_arrays(env.sample_dyn(a.batch, rng)); dyn.update(env.sample_async(a.batch, rng))
    q_off, _ = env.explore_noise(a.batch, rng, 0.0, 0.0)
    out = env.rollout(S, ref, jax.random.PRNGKey(100 + r), stochastic=False, q_offset=q_off, dyn=dyn)
    O = np.asarray(out["obs"])[:, :Nc].reshape(-1, od)
    Os.append(O); Ys.append(np.asarray(mean(T, jnp.asarray(O, jnp.float32))))
    Oa, Ya = jnp.asarray(np.concatenate(Os), jnp.float32), jnp.asarray(np.concatenate(Ys), jnp.float32)
    st = (jax.tree_util.tree_map(jnp.zeros_like, S), jax.tree_util.tree_map(jnp.zeros_like, S), 0)
    for s in range(a.steps):
        b = jnp.asarray(rng.integers(0, len(Oa), 4096))
        l, gr = grad(S, Oa[b], Ya[b]); S, st = adam(S, gr, st, a.lr)
    S["W0"] = S["W0"] * KEEP[None, :]
    print(f"round {r}: data {len(Oa)}, mse {float(l):.5f}, falls in its rollouts {np.mean(out['fell']):.1%}", flush=True)
evaluate(S, "student final")
d = os.path.join(a.out, "policy"); os.makedirs(d, exist_ok=True)
np.savez(os.path.join(d, "policy.npz"), **{k: np.asarray(v) for k, v in S.items()}, n_hidden=np.array(2))
for f in ("bank.json", "manifest.json"):
    if os.path.exists(os.path.join(pdir, f)): shutil.copy(os.path.join(pdir, f), d)
print("BC_DONE", d, flush=True)
