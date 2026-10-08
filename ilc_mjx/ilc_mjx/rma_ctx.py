"""RMA with cross-trial context (a baseline): the adaptation module reads the robot's PREVIOUS jump -- its whole contact
phase (observations and actions) and how it landed -- instead of the few samples of the current one, so it sees
what our ILC sees (the robot's past trials). Shared by ppo_train.py --mode student_ctx (training under DR) and
rma_ctx_eval.py (calibration jumps on the robot, then the evaluation)."""
import jax
import jax.numpy as jnp
import numpy as np

OD = 31


def features(obs, act, e_land, fell, sx, Nc):
    """One jump -> its context vector: obs[:Nc, :31] and act[:Nc] (flattened), the landing error / sx[:3] (clipped,
    zero after a fall) and the fall flag. obs (>= Nc, >= 31), act (>= Nc, 4), e_land (3,) raw SRB units."""
    o = np.nan_to_num(np.asarray(obs, float)[:Nc, :OD])
    u = np.nan_to_num(np.asarray(act, float)[:Nc, :4])
    e = np.zeros(3) if fell else np.clip(np.nan_to_num(np.asarray(e_land, float)[:3] / np.asarray(sx, float)[:3]), -10, 10)
    return np.concatenate([np.clip(o, -10, 10).ravel(), u.ravel(), e, [float(fell)]]).astype(np.float32)


def dim(Nc):
    return Nc * (OD + 4) + 4


def init(key, d_in, L, H=256):
    k = jax.random.split(key, 3)
    he = lambda kk, o, i, s=1.0: jax.random.normal(kk, (o, i)) * jnp.sqrt(2.0 / i) * s
    return dict(C_W0=he(k[0], H, d_in), C_b0=jnp.zeros(H), C_W1=he(k[1], H, H), C_b1=jnp.zeros(H),
                C_Wo=he(k[2], L, H, 0.1), C_bo=jnp.zeros(L))


def psi(p, f):
    h = jax.nn.relu(p["C_W0"] @ f + p["C_b0"])
    h = jax.nn.relu(p["C_W1"] @ h + p["C_b1"])
    return p["C_Wo"] @ h + p["C_bo"]


psi_b = jax.jit(jax.vmap(psi, (None, 0)))


def latent_actor(T):
    """The teacher's actor reading [o, l] directly (its encoder E dropped): for per-jump latents through env.priv_dim."""
    return {k: jnp.asarray(v, jnp.float32) for k, v in T.items() if k not in ("E", "bE", "log_sigma")}


def fold(T, l):
    """The deployable 31-d actor with the latent l fixed: its latent inputs folded into the first layer's bias."""
    w = latent_actor(T)
    W0 = np.asarray(T["W0"], float)
    w["W0"] = jnp.asarray(W0[:, :OD], jnp.float32)
    w["b0"] = jnp.asarray(np.asarray(T["b0"], float) + W0[:, OD:] @ np.asarray(l, float), jnp.float32)
    return w


def prior_latent(T, z_nom):
    """E z + bE at z_nom (the DR family's mean z): the first jump's latent, before any context."""
    return np.asarray(T["E"], float) @ np.asarray(z_nom, float) + np.asarray(T["bE"], float)
