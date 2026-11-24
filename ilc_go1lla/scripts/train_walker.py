"""Train the Go1 joystick walker (MuJoCo Playground Go1JoystickFlatTerrain, Brax PPO) on the NOMINAL Go1, briefly.

The network the stage later corrects: obs -> joint-position-target offsets (PD Kp 35, Kd 0.5, action scale 0.5, 50 Hz).
Wall-clock time to each checkpoint is logged (the compute-gap comparison: stage minutes vs PPO / PPO-DR hours).

    ~/venvs/pg0/bin/python scripts/train_walker.py --steps 60_000_000 --out runs/walker_nom_s0 [--dr]
"""
import argparse
import functools
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=60_000_000)
ap.add_argument('--evals', type=int, default=6)
ap.add_argument('--dr', action='store_true', help="Playground's own Go1 domain randomizer (friction, mass, CoM, motor)")
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()

import jax                                                      # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

if not hasattr(jax, 'device_put_replicated'):                   # removed in jax >= 0.11; brax 0.14 still calls it
    def _put_replicated(x, devices):                            # single-GPU stand-in: a leading device axis of 1
        assert len(devices) == 1, 'shim supports one device'
        return jax.tree.map(lambda v: jax.device_put(jnp.expand_dims(jnp.asarray(v), 0), devices[0]), x)
    jax.device_put_replicated = _put_replicated
from brax.io import model                                       # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks   # noqa: E402
from brax.training.agents.ppo import train as ppo               # noqa: E402
from mujoco_playground import registry, wrapper                 # noqa: E402
from mujoco_playground.config import locomotion_params          # noqa: E402

NAME = 'Go1JoystickFlatTerrain'
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


train_fn = functools.partial(
    ppo.train, **dict(pp), network_factory=functools.partial(ppo_networks.make_ppo_networks, **nf),
    progress_fn=progress, seed=a.seed,
    randomization_fn=registry.get_domain_randomizer(NAME) if a.dr else None)
make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env,
                                        wrap_env_fn=wrapper.wrap_for_brax_training)
model.save_params(os.path.join(a.out, 'params.pkl'), params)
json.dump(dict(args=vars(a), env=NAME, env_cfg=env_cfg.to_dict(), ppo=pp.to_dict(), network=nf.to_dict(),
               wall_s=time.time() - t0, jax=jax.__version__), open(os.path.join(a.out, 'meta.json'), 'w'), indent=1,
          default=str)
print('DONE wall_s', time.time() - t0, flush=True)
