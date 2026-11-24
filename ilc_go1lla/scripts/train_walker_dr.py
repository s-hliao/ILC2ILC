"""Train the Go1 joystick walker WITH domain randomization (2026-11-19; the baselines next to the DistILC stage).

  --dr light   Playground's own Go1 randomizer (floor friction, joint friction / armature, CoM jitter, link masses, torso
               mass +-1 kg, qpos0): the usual "small DR"
  --dr full    light + the test perturbations' whole range, so DR COVERS the experiments it is compared on:
               an off-centre payload m ~ U(0, 5) kg at dy ~ U(-0.12, 0.12) m, dz 0.08 m (walkenv.perturb_model's model:
               torso mass, CoM and parallel-axis inertia), and half the robots on foam: floor solref time constant
               ~ U(0.05, 0.25) s, solimp [0, 0.95, width ~ U(0.05, 0.2), 0.5, 2] (the tests: 0.15/0.15 and 0.2/0.1)
Same PPO config / network as train_walker.py; wall-clock time logged at every eval (the compute-gap number).

    ~/venvs/pg0/bin/python scripts/train_walker_dr.py --dr full --steps 300_000_000 --evals 30 --out runs/walker_fulldr_s0
"""
import argparse
import functools
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument('--dr', choices=['light', 'full'], required=True)
ap.add_argument('--steps', type=int, default=60_000_000)
ap.add_argument('--evals', type=int, default=6)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()

import jax                                                      # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

if not hasattr(jax, 'device_put_replicated'):
    def _put_replicated(x, devices):
        assert len(devices) == 1, 'shim supports one device'
        return jax.tree.map(lambda v: jax.device_put(jnp.expand_dims(jnp.asarray(v), 0), devices[0]), x)
    jax.device_put_replicated = _put_replicated
from brax.io import model                                       # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks   # noqa: E402
from brax.training.agents.ppo import train as ppo               # noqa: E402
from mujoco_playground import registry, wrapper                 # noqa: E402
from mujoco_playground.config import locomotion_params          # noqa: E402
from mujoco_playground._src.locomotion.go1 import randomize as go1_rand   # noqa: E402

NAME = 'Go1JoystickFlatTerrain'
TORSO, FLOOR = go1_rand.TORSO_BODY_ID, go1_rand.FLOOR_GEOM_ID


def full_randomize(m, rng):
    """Playground's randomizer, then per environment a payload and (half the time) foam."""
    m, in_axes = go1_rand.domain_randomize(m, rng)
    n = rng.shape[0]
    k = jax.random.split(jax.random.fold_in(rng[0], 7), 6)
    mp = jax.random.uniform(k[0], (n,), minval=0.0, maxval=5.0)
    dy = jax.random.uniform(k[1], (n,), minval=-0.12, maxval=0.12)
    dz = 0.08
    M = m.body_mass[:, TORSO]
    c = m.body_ipos[:, TORSO]
    r = c + jnp.stack([jnp.zeros(n), dy, jnp.full(n, dz)], 1)
    cn = (M[:, None] * c + mp[:, None] * r) / (M + mp)[:, None]
    d, dc = r - cn, cn - c
    par = lambda v, w: w[:, None] * jnp.stack([v[:, 1] ** 2 + v[:, 2] ** 2, v[:, 0] ** 2 + v[:, 2] ** 2, v[:, 0] ** 2 + v[:, 1] ** 2], 1)
    dI = par(d, mp) + par(dc, M)
    body_mass = m.body_mass.at[:, TORSO].set(M + mp)
    body_ipos = m.body_ipos.at[:, TORSO].set(cn)
    body_inertia = jnp.broadcast_to(m.body_inertia, (n,) + m.body_inertia.shape).at[:, TORSO].add(dI)
    foam = jax.random.uniform(k[2], (n,)) < 0.5
    tc = jax.random.uniform(k[3], (n,), minval=0.05, maxval=0.25)
    wd = jax.random.uniform(k[4], (n,), minval=0.05, maxval=0.20)
    sr0 = jnp.broadcast_to(m.geom_solref, (n,) + m.geom_solref.shape)
    si0 = jnp.broadcast_to(m.geom_solimp, (n,) + m.geom_solimp.shape)
    solref = jnp.where(foam[:, None], jnp.stack([tc, jnp.ones(n)], 1), sr0[:, FLOOR])
    solimp = jnp.where(foam[:, None], jnp.stack([jnp.zeros(n), jnp.full(n, 0.95), wd, jnp.full(n, 0.5), jnp.full(n, 2.0)], 1), si0[:, FLOOR])
    geom_solref = sr0.at[:, FLOOR].set(solref)
    geom_solimp = si0.at[:, FLOOR].set(solimp)
    m = m.tree_replace(dict(body_mass=body_mass, body_ipos=body_ipos, body_inertia=body_inertia,
                            geom_solref=geom_solref, geom_solimp=geom_solimp))
    in_axes = in_axes.tree_replace(dict(body_inertia=0, geom_solref=0, geom_solimp=0))
    return m, in_axes


os.makedirs(a.out, exist_ok=True)
env_cfg = registry.get_default_config(NAME)
env = registry.load(NAME, config=env_cfg)
eval_env = registry.load(NAME, config=env_cfg)
pp = locomotion_params.brax_ppo_config(NAME)
pp.num_timesteps = a.steps
pp.num_evals = a.evals
nf = pp.network_factory
del pp['network_factory']
t0 = time.time()
log = open(os.path.join(a.out, 'progress.jsonl'), 'w')


def progress(step, metrics):
    row = dict(step=int(step), wall_s=time.time() - t0,
               reward=float(metrics.get('eval/episode_reward', float('nan'))),
               reward_std=float(metrics.get('eval/episode_reward_std', float('nan'))))
    log.write(json.dumps(row) + '\n'); log.flush()
    print(json.dumps(row), flush=True)


rand = go1_rand.domain_randomize if a.dr == 'light' else full_randomize
os.makedirs(os.path.join(a.out, 'ck'), exist_ok=True)


def save_ck(step, make_policy, params):          # (2026-11-19) params at every eval: pick the earliest checkpoint that walks
    model.save_params(os.path.join(a.out, 'ck', f'params_{int(step):011d}.pkl'), params)


train_fn = functools.partial(
    ppo.train, **dict(pp), network_factory=functools.partial(ppo_networks.make_ppo_networks, **nf),
    progress_fn=progress, policy_params_fn=save_ck, seed=a.seed, randomization_fn=rand)
make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)
model.save_params(os.path.join(a.out, 'params.pkl'), params)
json.dump(dict(args=vars(a), env=NAME, env_cfg=env_cfg.to_dict(), ppo=pp.to_dict(), network=nf.to_dict(),
               wall_s=time.time() - t0, jax=jax.__version__), open(os.path.join(a.out, 'meta.json'), 'w'), indent=1, default=str)
print('DONE wall_s', time.time() - t0, flush=True)
