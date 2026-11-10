"""fada_quad_tf_core.py (2026-11-07): Adam (beta 0.9 / 0.999, eps 1e-8; optax.adam's update written out -- the quad env has no optax),
the same as fada_quad_tf.py's _Adam."""
import jax
import jax.numpy as jnp


class Adam:
    def __init__(self, lr):
        self.lr = lr

    def init(self, p):
        z = jax.tree_util.tree_map(jnp.zeros_like, p)
        return (z, z, jnp.zeros((), jnp.int32))

    def update(self, g, s, p):
        m, v, t = s
        t = t + 1
        m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ * g_, v, g)
        u = jax.tree_util.tree_map(lambda m_, v_: -self.lr * (m_ / (1 - 0.9 ** t)) / (jnp.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8), m, v)
        return u, (m, v, t)


def apply_updates(p, u):
    return jax.tree_util.tree_map(lambda a_, b_: a_ + b_, p, u)
