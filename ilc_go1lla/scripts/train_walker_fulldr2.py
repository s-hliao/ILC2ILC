"""PPO + FULL domain randomization for the Go1 walker, trained to convergence (2026-11-19, user: "there are more axes of
randomization that are necessary ... I want the full randomization" -- the thorough hardware-style DR baseline).

Environment: Playground's Go1 joystick on ROUGH TERRAIN (heightfield, 5 cm), random body kicks ON (Playground's pert_config),
observation noise ON (Playground default, as every walker here). Per environment (fixed for its episodes):
  Playground's randomizer   floor friction U(0.4, 1.0), joint friction x U(0.9, 1.1), armature x U(1, 1.05),
                            CoM +-5 cm, link masses x U(0.9, 1.1), torso mass +-1 kg, qpos0 +-0.05
  motors                    Kp x U(0.8, 1.2), Kd x U(0.8, 1.2), torque limits x U(0.8, 1.2)
  payload                   m ~ U(0, 5) kg at dx ~ U(-0.10, 0.10), dy ~ U(-0.12, 0.12), dz ~ U(0.04, 0.12) m (torso mass, CoM,
                            parallel-axis inertia, as walkenv.perturb_model)
  foam (half the envs)      floor solref [tc ~ U(0.05, 0.25) s, damping ratio ~ U(0.5, 2.0)], solimp [0, 0.95, width ~ U(0.05,
                            0.20) m, 0.5, 2] (the tests: 0.15/0.15 and 0.2/0.1)
  latency (half the envs)   the action is applied one control step (20 ms) late
Same PPO config / network as train_walker.py. Logged at every eval: steps, wall time, reward; params saved at every eval
(runs/<out>/ck/), so the run can be stopped at convergence and the time-to-plateau reported (not the budget).

    ~/venvs/pg0/bin/python scripts/train_walker_fulldr2.py --steps 1_000_000_000 --evals 100 --out runs/walker_fulldr2_s0
"""
import argparse
import functools
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=1_000_000_000)
ap.add_argument('--evals', type=int, default=100)
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
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick   # noqa: E402
from mujoco_playground._src.locomotion.go1 import randomize as go1_rand      # noqa: E402

NAME = 'Go1JoystickFlatTerrain'          # PPO config / network as the baseline walker; the env is the rough-terrain task
TORSO, FLOOR = go1_rand.TORSO_BODY_ID, go1_rand.FLOOR_GEOM_ID


class LatencyJoystick(go1_joystick.Joystick):
    """Half the episodes apply each action one control step late (a buffer of the policy's last raw action)."""

    def reset(self, rng):
        st = super().reset(rng)
        st.info['act_buf'] = jnp.zeros(self.mjx_model.nu)
        st.info['delay'] = jax.random.bernoulli(jax.random.fold_in(rng, 11), 0.5)
        return st

    def step(self, state, action):
        applied = jnp.where(state.info['delay'], state.info['act_buf'], action)
        state.info['act_buf'] = action
        return super().step(state, applied)


def full_randomize(m, rng):
    m, in_axes = go1_rand.domain_randomize(m, rng)
    n = rng.shape[0]
    k = jax.random.split(jax.random.fold_in(rng[0], 7), 12)
    U = lambda i, lo, hi, shape=(n,): jax.random.uniform(k[i], shape, minval=lo, maxval=hi)
    # motors
    kp, kd, tq = U(0, 0.8, 1.2), U(1, 0.8, 1.2), U(2, 0.8, 1.2)
    gainprm = jnp.broadcast_to(m.actuator_gainprm, (n,) + m.actuator_gainprm.shape)
    biasprm = jnp.broadcast_to(m.actuator_biasprm, (n,) + m.actuator_biasprm.shape)
    gainprm = gainprm.at[:, :, 0].multiply(kp[:, None])
    biasprm = biasprm.at[:, :, 1].multiply(kp[:, None])
    dof_damping = jnp.broadcast_to(m.dof_damping, (n,) + m.dof_damping.shape)
    dof_damping = dof_damping.at[:, 6:].multiply(kd[:, None])
    forcerange = jnp.broadcast_to(m.actuator_forcerange, (n,) + m.actuator_forcerange.shape) * tq[:, None, None]
    # payload
    mp, dx, dy, dz = U(3, 0.0, 5.0), U(4, -0.10, 0.10), U(5, -0.12, 0.12), U(6, 0.04, 0.12)
    M = m.body_mass[:, TORSO]
    c = m.body_ipos[:, TORSO]
    r = c + jnp.stack([dx, dy, dz], 1)
    cn = (M[:, None] * c + mp[:, None] * r) / (M + mp)[:, None]
    d, dc = r - cn, cn - c
    par = lambda v, w: w[:, None] * jnp.stack([v[:, 1] ** 2 + v[:, 2] ** 2, v[:, 0] ** 2 + v[:, 2] ** 2, v[:, 0] ** 2 + v[:, 1] ** 2], 1)
    body_mass = m.body_mass.at[:, TORSO].set(M + mp)
    body_ipos = m.body_ipos.at[:, TORSO].set(cn)
    body_inertia = jnp.broadcast_to(m.body_inertia, (n,) + m.body_inertia.shape).at[:, TORSO].add(par(d, mp) + par(dc, M))
    # foam
    foam = U(7, 0.0, 1.0) < 0.5
    tc, dr, wd = U(8, 0.05, 0.25), U(9, 0.5, 2.0), U(10, 0.05, 0.20)
    sr0 = jnp.broadcast_to(m.geom_solref, (n,) + m.geom_solref.shape)
    si0 = jnp.broadcast_to(m.geom_solimp, (n,) + m.geom_solimp.shape)
    solref = jnp.where(foam[:, None], jnp.stack([tc, dr], 1), sr0[:, FLOOR])
    solimp = jnp.where(foam[:, None], jnp.stack([jnp.zeros(n), jnp.full(n, 0.95), wd, jnp.full(n, 0.5), jnp.full(n, 2.0)], 1), si0[:, FLOOR])
    m = m.tree_replace(dict(body_mass=body_mass, body_ipos=body_ipos, body_inertia=body_inertia,
                            geom_solref=sr0.at[:, FLOOR].set(solref), geom_solimp=si0.at[:, FLOOR].set(solimp),
                            actuator_gainprm=gainprm, actuator_biasprm=biasprm, dof_damping=dof_damping,
                            actuator_forcerange=forcerange))
    in_axes = in_axes.tree_replace(dict(body_inertia=0, geom_solref=0, geom_solimp=0, actuator_gainprm=0,
                                        actuator_biasprm=0, dof_damping=0, actuator_forcerange=0))
    return m, in_axes


os.makedirs(os.path.join(a.out, 'ck'), exist_ok=True)
cfg = registry.get_default_config('Go1JoystickRoughTerrain')
cfg.pert_config.enable = True
env = LatencyJoystick(task='rough_terrain', config=cfg)
eval_env = LatencyJoystick(task='rough_terrain', config=cfg)
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


def save_ck(step, make_policy, params):
    model.save_params(os.path.join(a.out, 'ck', f'params_{int(step):011d}.pkl'), params)


json.dump(dict(args=vars(a), env='Go1JoystickRoughTerrain+latency', env_cfg=cfg.to_dict(), ppo=pp.to_dict(), network=nf.to_dict(),
               jax=jax.__version__), open(os.path.join(a.out, 'meta_start.json'), 'w'), indent=1, default=str)
train_fn = functools.partial(
    ppo.train, **dict(pp), network_factory=functools.partial(ppo_networks.make_ppo_networks, **nf),
    progress_fn=progress, policy_params_fn=save_ck, seed=a.seed, randomization_fn=full_randomize)
make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)
model.save_params(os.path.join(a.out, 'params.pkl'), params)
json.dump(dict(args=vars(a), env='Go1JoystickRoughTerrain+latency', wall_s=time.time() - t0), open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
print('DONE wall_s', time.time() - t0, flush=True)
