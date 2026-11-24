"""(2026-11-24) DR-AXIS SWEEP for the goal walker: DR axes added cumulatively, every run warm-started from the same nominal walker,
elliptic friction cones by default, trained toward a plateau (checkpoint every eval) -> steps-to-plateau per number of axes.
  --axes  comma list of: playground (friction / masses / CoM / joint friction; always first), foam (soft floor, --foam-p of the envs),
          motors (Kp / Kd / torque x U(0.8, 1.2)), payload (0-5 kg off-centre), actlat (action one step late, half the envs),
          rough (5 cm heightfield), kicks (random body pushes), obslat (policy observation one step = 20 ms late, half the envs;
          hardware IMU 10-24 ms), jitter (each step with --jitter-p the previous action is held again: control-period jitter;
          hardware policy period up to 45 ms)
The randomizer parts are copied verbatim from train_goalwalker_dr.py (full_randomize).
    python scripts/train_goalwalker_axes.py --axes playground,foam,motors --restore runs/goalwalk/vel2_s0/params.pkl --out ...
"""
import argparse
import functools
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--axes', required=True)
ap.add_argument('--variant', default='vel2')
ap.add_argument('--cone', default='elliptic', choices=['pyramidal', 'elliptic'])
ap.add_argument('--steps', type=int, default=600_000_000)
ap.add_argument('--evals', type=int, default=48)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
ap.add_argument('--restore', default=None)
ap.add_argument('--foam-p', type=float, default=0.5)
ap.add_argument('--foam-tc', type=float, nargs=2, default=[0.05, 0.25])
ap.add_argument('--foam-w', type=float, nargs=2, default=[0.05, 0.20])
ap.add_argument('--foam-dr', type=float, nargs=2, default=[0.5, 2.0])
ap.add_argument('--jitter-p', type=float, default=0.15)
a = ap.parse_args()
AX = set(a.axes.split(','))
assert AX <= {'playground', 'foam', 'motors', 'payload', 'actlat', 'rough', 'kicks', 'obslat', 'jitter'}, AX
assert 'playground' in AX

import jax                                                      # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
if not hasattr(jax, 'device_put_replicated'):
    def _put_replicated(x, devices):
        assert len(devices) == 1, 'shim supports one device'
        return jax.tree.map(lambda v: jax.device_put(jnp.expand_dims(jnp.asarray(v), 0), devices[0]), x)
    jax.device_put_replicated = _put_replicated
import mujoco                                                   # noqa: E402
from brax.io import model                                       # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks   # noqa: E402
from brax.training.agents.ppo import train as ppo               # noqa: E402
from mujoco_playground import wrapper                           # noqa: E402
from mujoco_playground.config import locomotion_params          # noqa: E402
from mujoco_playground._src.locomotion.go1 import randomize as go1_rand      # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ilc_go1lla import goalwalk                                 # noqa: E402

NAME = 'Go1JoystickFlatTerrain'
TORSO, FLOOR = go1_rand.TORSO_BODY_ID, go1_rand.FLOOR_GEOM_ID


class AxesGoalWalker(goalwalk.GoalWalker):
    """actlat: half the episodes apply each action one step late; obslat: half the episodes see the policy observation one step
    late; jitter: each step, with probability jitter_p, the previous applied action is held again."""

    def reset(self, rng):
        st = super().reset(rng)
        st.info['act_buf'] = jnp.zeros(self.mjx_model.nu)
        st.info['delay'] = jax.random.bernoulli(jax.random.fold_in(rng, 11), 0.5) & ('actlat' in AX)
        st.info['odelay'] = jax.random.bernoulli(jax.random.fold_in(rng, 13), 0.5) & ('obslat' in AX)
        st.info['obs_buf'] = st.obs['state']
        st.info['applied_prev'] = jnp.zeros(self.mjx_model.nu)
        return st

    def step(self, state, action):
        applied = jnp.where(state.info['delay'], state.info['act_buf'], action)
        if 'jitter' in AX:
            state.info['rng'], kj = jax.random.split(state.info['rng'])
            hold = jax.random.bernoulli(kj, a.jitter_p)
            applied = jnp.where(hold, state.info['applied_prev'], applied)
        state.info['act_buf'] = action
        state.info['applied_prev'] = applied
        st = super().step(state, applied)
        new = st.obs['state']
        seen = jnp.where(st.info['odelay'], st.info['obs_buf'], new)
        st.info['obs_buf'] = new
        return st.replace(obs=dict(st.obs, state=seen))


def rand(m, rng):
    m, in_axes = go1_rand.domain_randomize(m, rng)
    n = rng.shape[0]
    k = jax.random.split(jax.random.fold_in(rng[0], 7), 12)
    U = lambda i, lo, hi, shape=(n,): jax.random.uniform(k[i], shape, minval=lo, maxval=hi)
    upd, ax = {}, {}
    if 'motors' in AX:
        # motors
        kp, kd, tq = U(0, 0.8, 1.2), U(1, 0.8, 1.2), U(2, 0.8, 1.2)
        gainprm = jnp.broadcast_to(m.actuator_gainprm, (n,) + m.actuator_gainprm.shape)
        biasprm = jnp.broadcast_to(m.actuator_biasprm, (n,) + m.actuator_biasprm.shape)
        gainprm = gainprm.at[:, :, 0].multiply(kp[:, None])
        biasprm = biasprm.at[:, :, 1].multiply(kp[:, None])
        dof_damping = jnp.broadcast_to(m.dof_damping, (n,) + m.dof_damping.shape)
        dof_damping = dof_damping.at[:, 6:].multiply(kd[:, None])
        forcerange = jnp.broadcast_to(m.actuator_forcerange, (n,) + m.actuator_forcerange.shape) * tq[:, None, None]
        upd.update(actuator_gainprm=gainprm, actuator_biasprm=biasprm, dof_damping=dof_damping, actuator_forcerange=forcerange)
        ax.update(actuator_gainprm=0, actuator_biasprm=0, dof_damping=0, actuator_forcerange=0)
    if 'payload' in AX:
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
        upd.update(body_mass=body_mass, body_ipos=body_ipos, body_inertia=body_inertia)
        ax.update(body_inertia=0)
    if 'foam' in AX:
        # foam
        foam = U(7, 0.0, 1.0) < a.foam_p
        tc, dr, wd = U(8, *a.foam_tc), U(9, *a.foam_dr), U(10, *a.foam_w)
        sr0 = jnp.broadcast_to(m.geom_solref, (n,) + m.geom_solref.shape)
        si0 = jnp.broadcast_to(m.geom_solimp, (n,) + m.geom_solimp.shape)
        solref = jnp.where(foam[:, None], jnp.stack([tc, dr], 1), sr0[:, FLOOR])
        solimp = jnp.where(foam[:, None], jnp.stack([jnp.zeros(n), jnp.full(n, 0.95), wd, jnp.full(n, 0.5), jnp.full(n, 2.0)], 1), si0[:, FLOOR])
        upd.update(geom_solref=sr0.at[:, FLOOR].set(solref), geom_solimp=si0.at[:, FLOOR].set(solimp))
        ax.update(geom_solref=0, geom_solimp=0)
    if upd:
        m = m.tree_replace(upd)
        in_axes = in_axes.tree_replace(ax)
    return m, in_axes


os.makedirs(os.path.join(a.out, 'ck'), exist_ok=True)


def mk():
    cfg = goalwalk.default_config(a.variant)
    cfg.pert_config.enable = 'kicks' in AX
    if 'rough' in AX:
        return AxesGoalWalker(task='rough_terrain', config=cfg, config_overrides=dict(njmax=120, naconmax=12 * 8192))
    return AxesGoalWalker(config=cfg)


env, eval_env = mk(), mk()
if a.cone == 'elliptic':
    for _e in (env, eval_env):
        _e._mjx_model = _e._mjx_model.replace(opt=_e._mjx_model.opt.replace(cone=int(mujoco.mjtCone.mjCONE_ELLIPTIC)))
        _e._mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
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


def save_ck(step, make_policy, params):
    model.save_params(os.path.join(a.out, 'ck', f'params_{int(step):011d}.pkl'), params)


json.dump(dict(args=vars(a), axes=sorted(AX)), open(os.path.join(a.out, 'meta_start.json'), 'w'), indent=1)
kw = dict(restore_params=model.load_params(a.restore)) if a.restore else {}
train_fn = functools.partial(
    ppo.train, **dict(pp), network_factory=functools.partial(ppo_networks.make_ppo_networks, **nf),
    progress_fn=progress, policy_params_fn=save_ck, seed=a.seed, randomization_fn=rand, **kw)
make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)
model.save_params(os.path.join(a.out, 'params.pkl'), params)
json.dump(dict(args=vars(a), axes=sorted(AX), wall_s=time.time() - t0), open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
print('DONE wall_s', time.time() - t0, flush=True)
