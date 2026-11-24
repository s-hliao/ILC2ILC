"""Which full-DR axis breaks the walker? (2026-11-21: full DR v3 from scratch and v4 fine-tuned from the nominal walker both end
STANDING -- reward ~12-13, 0/20 reach the goal.) Runs a fixed policy (default the nominal walker) for one 1000-step episode in
Playground's joystick env with each axis alone, and reports the eval reward and how many episodes end early (falls).

    CUDA_VISIBLE_DEVICES=0 ~/venvs/pg0/bin/python scripts/dr_axis_check.py --params runs/walker_nom_s0/params.pkl
"""
import argparse, functools, importlib.util, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument('--params', default='runs/walker_nom_s0/params.pkl')
ap.add_argument('--n', type=int, default=256)
ap.add_argument('--steps', type=int, default=1000)
ap.add_argument('--variants', nargs='+', default=['flat', 'flat_pg', 'rough', 'rough_kicks', 'flat_latency', 'flat_motors',
                                                   'flat_payload', 'flat_foam', 'full'])
a = ap.parse_args()
import jax, jax.numpy as jnp, numpy as np
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick
from mujoco_playground._src.locomotion.go1 import randomize as go1_rand
TORSO, FLOOR = go1_rand.TORSO_BODY_ID, go1_rand.FLOOR_GEOM_ID


class LatencyJoystick(go1_joystick.Joystick):          # as train_walker_fulldr3.py, with the delay forced on/off
    def __init__(self, *args, p_delay=0.5, **kw):
        super().__init__(*args, **kw); self.p_delay = p_delay
    def reset(self, rng):
        st = super().reset(rng)
        st.info['act_buf'] = jnp.zeros(self.mjx_model.nu)
        st.info['delay'] = jax.random.bernoulli(jax.random.fold_in(rng, 11), self.p_delay)
        return st
    def step(self, state, action):
        applied = jnp.where(state.info['delay'], state.info['act_buf'], action)
        state.info['act_buf'] = action
        return super().step(state, applied)


def rand(m, rng, pg=False, motors=False, payload=False, foam=False, foam_p=0.5):
    """train_walker_fulldr3.full_randomize, one axis at a time (pg = Playground's own randomizer)."""
    m, in_axes = go1_rand.domain_randomize(m, rng)   # every randomized variant starts from Playground's randomizer (the in_axes)
    n = rng.shape[0]
    k = jax.random.split(jax.random.fold_in(rng[0], 7), 12)
    U = lambda i, lo, hi: jax.random.uniform(k[i], (n,), minval=lo, maxval=hi)
    rep, ax = {}, {}
    if motors:
        kp, kd, tq = U(0, 0.8, 1.2), U(1, 0.8, 1.2), U(2, 0.8, 1.2)
        rep['actuator_gainprm'] = jnp.broadcast_to(m.actuator_gainprm, (n,) + m.actuator_gainprm.shape).at[:, :, 0].multiply(kp[:, None])
        rep['actuator_biasprm'] = jnp.broadcast_to(m.actuator_biasprm, (n,) + m.actuator_biasprm.shape).at[:, :, 1].multiply(kp[:, None])
        rep['dof_damping'] = jnp.broadcast_to(m.dof_damping, (n,) + m.dof_damping.shape).at[:, 6:].multiply(kd[:, None])
        rep['actuator_forcerange'] = jnp.broadcast_to(m.actuator_forcerange, (n,) + m.actuator_forcerange.shape) * tq[:, None, None]
    if payload:
        mp, dx, dy, dz = U(3, 0.0, 5.0), U(4, -0.10, 0.10), U(5, -0.12, 0.12), U(6, 0.04, 0.12)
        M = m.body_mass[:, TORSO]; c = m.body_ipos[:, TORSO]
        r = c + jnp.stack([dx, dy, dz], 1); cn = (M[:, None] * c + mp[:, None] * r) / (M + mp)[:, None]
        d, dc = r - cn, cn - c
        par = lambda v, w: w[:, None] * jnp.stack([v[:, 1] ** 2 + v[:, 2] ** 2, v[:, 0] ** 2 + v[:, 2] ** 2, v[:, 0] ** 2 + v[:, 1] ** 2], 1)
        rep['body_mass'] = m.body_mass.at[:, TORSO].set(M + mp)
        rep['body_ipos'] = m.body_ipos.at[:, TORSO].set(cn)
        rep['body_inertia'] = jnp.broadcast_to(m.body_inertia, (n,) + m.body_inertia.shape).at[:, TORSO].add(par(d, mp) + par(dc, M))
    if foam:
        fm = U(7, 0.0, 1.0) < foam_p
        tc, dr, wd = U(8, 0.05, 0.25), U(9, 0.5, 2.0), U(10, 0.05, 0.20)
        sr0 = jnp.broadcast_to(m.geom_solref, (n,) + m.geom_solref.shape)
        si0 = jnp.broadcast_to(m.geom_solimp, (n,) + m.geom_solimp.shape)
        rep['geom_solref'] = sr0.at[:, FLOOR].set(jnp.where(fm[:, None], jnp.stack([tc, dr], 1), sr0[:, FLOOR]))
        rep['geom_solimp'] = si0.at[:, FLOOR].set(jnp.where(fm[:, None], jnp.stack([jnp.zeros(n), jnp.full(n, 0.95), wd, jnp.full(n, 0.5), jnp.full(n, 2.0)], 1), si0[:, FLOOR]))
    m = m.tree_replace(rep)
    in_axes = in_axes.tree_replace({key: 0 for key in rep})
    return m, in_axes


SPEC = {  # name: (task, kicks, p_delay, rand kwargs or None)
    'flat': ('flat_terrain', False, 0.0, None),
    'flat_pg': ('flat_terrain', False, 0.0, dict(pg=True)),
    'rough': ('rough_terrain', False, 0.0, None),
    'rough_kicks': ('rough_terrain', True, 0.0, None),
    'flat_latency': ('flat_terrain', False, 1.0, None),
    'flat_motors': ('flat_terrain', False, 0.0, dict(pg=True, motors=True)),
    'flat_payload': ('flat_terrain', False, 0.0, dict(pg=True, payload=True)),
    'flat_foam': ('flat_terrain', False, 0.0, dict(pg=True, foam=True, foam_p=1.0)),
    'full': ('rough_terrain', True, 0.5, dict(pg=True, motors=True, payload=True, foam=True)),
}
params = model.load_params(a.params)
pp = locomotion_params.brax_ppo_config('Go1JoystickFlatTerrain')
nf = pp.network_factory
for name in a.variants:
    task, kicks, pdel, rk = SPEC[name]
    cfg = registry.get_default_config('Go1JoystickRoughTerrain' if task.startswith('rough') else 'Go1JoystickFlatTerrain')
    cfg.pert_config.enable = kicks
    ovr = dict(njmax=120, naconmax=12 * 8192) if task.startswith('rough') else None
    env = LatencyJoystick(task=task, config=cfg, config_overrides=ovr, p_delay=pdel)
    keys = jax.random.split(jax.random.PRNGKey(0), a.n)
    wenv = wrapper.wrap_for_brax_training(env, episode_length=a.steps, action_repeat=1,
                                          randomization_fn=(lambda m: rand(m, keys, **rk)) if rk else None)
    net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                         preprocess_observations_fn=__import__('brax.training.acme.running_statistics', fromlist=['normalize']).normalize, **nf)
    pol = ppo_networks.make_inference_fn(net)((params[0], params[1]), deterministic=True)
    reset, step = jax.jit(wenv.reset), jax.jit(wenv.step)
    t0 = time.time()
    st = reset(keys)
    tot = jnp.zeros(a.n); alive = jnp.ones(a.n); ended = jnp.zeros(a.n)
    def body(c, _):
        st, tot, alive, k = c
        k, sk = jax.random.split(k)
        act, _ = pol(st.obs, sk)
        st2 = step(st, act)
        tot = tot + alive * st2.reward
        alive = alive * (1 - st2.done)
        return (st2, tot, alive, k), None
    (st, tot, alive, _), _ = jax.lax.scan(body, (st, tot, alive, jax.random.PRNGKey(1)), None, length=a.steps)
    tot, alive = np.asarray(tot), np.asarray(alive)
    print(f'{name:14s} reward {tot.mean():6.2f} +- {tot.std():5.2f} | ended early (fell) {100 * (1 - alive.mean()):5.1f} % | {time.time() - t0:5.0f} s', flush=True)
