"""(2026-11-24) Train the GOAL-CONDITIONED Go1 walker (ilc_go1lla/goalwalk.py: goal in the observation, any heading, learned
turning; no command law) with Playground's Go1 joystick PPO settings, on the NOMINAL Go1.
Copied from train_walker.py.

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
ap.add_argument('--variant', default='vel', choices=['vel', 'vel2', 'prog'])
ap.add_argument('--cone', default='pyramidal', choices=['pyramidal', 'elliptic'], help='(11-24) friction cones; elliptic = the deployment-foam fit (runs/foamfit)')
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
import sys                                                     # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ilc_go1lla import goalwalk                                 # noqa: E402
env_cfg = goalwalk.default_config(a.variant)
env = goalwalk.GoalWalker(config=env_cfg)
eval_env = goalwalk.GoalWalker(config=goalwalk.default_config(a.variant))
if a.cone == 'elliptic':
    import mujoco as _mj
    for _e in (env, eval_env):
        _e._mjx_model = _e._mjx_model.replace(opt=_e._mjx_model.opt.replace(cone=int(_mj.mjtCone.mjCONE_ELLIPTIC)))
        _e._mj_model.opt.cone = _mj.mjtCone.mjCONE_ELLIPTIC
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
               reward_std=float(metrics.get('eval/episode_reward_std', float('nan'))),
               goals=float(metrics.get('eval/episode_goals_reached', float('nan'))),
               dist=float(metrics.get('eval/episode_dist', float('nan'))))
    log.write(json.dumps(row) + '\n'); log.flush()
    print(json.dumps(row), flush=True)


train_fn = functools.partial(
    ppo.train, **dict(pp), network_factory=functools.partial(ppo_networks.make_ppo_networks, **nf),
    progress_fn=progress, seed=a.seed,
    randomization_fn=registry.get_domain_randomizer(NAME) if a.dr else None)
make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env,
                                        wrap_env_fn=wrapper.wrap_for_brax_training)
model.save_params(os.path.join(a.out, 'params.pkl'), params)
json.dump(dict(args=vars(a), env='GoalWalker(' + NAME + ')', env_cfg=env_cfg.to_dict(), ppo=pp.to_dict(), network=nf.to_dict(),
               wall_s=time.time() - t0, jax=jax.__version__), open(os.path.join(a.out, 'meta.json'), 'w'), indent=1,
          default=str)
print('DONE wall_s', time.time() - t0, flush=True)
