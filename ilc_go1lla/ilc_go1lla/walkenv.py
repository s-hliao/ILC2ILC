"""Go1 walks to goals with the Playground joystick walker, nominal or perturbed (LLA-MPPI-style, arXiv 2610.10465).

Perturbations (MJX model edits; the walker never trained on any of them):
  weak_leg=(leg, scale)   one leg's three actuator torque caps x scale (leg in FR, FL, RR, RL; scale 0 = no torque)
  payload=(m, dy, dz)     a point mass m kg on the trunk, dy to the side (+ left), dz above the trunk CoM
  foam=(timeconst, width) the floor soft: contact solref time constant (s) and solimp width (m) -> feet sink
The command law (fixed, not learned): body-frame velocity toward the goal, v = clip(K (goal - p)), yaw rate to face it.
Observation noise off (deterministic walks).
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

if not hasattr(jax, 'device_put_replicated'):                   # brax 0.14 on jax >= 0.11 (single GPU)
    def _put_replicated(x, devices):
        return jax.tree.map(lambda v: jax.device_put(jnp.expand_dims(jnp.asarray(v), 0), devices[0]), x)
    jax.device_put_replicated = _put_replicated

from mujoco import mjx                                          # noqa: E402
from brax.io import model as brax_model                          # noqa: E402
from brax.training.acme import running_statistics               # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks   # noqa: E402
from mujoco_playground import registry                          # noqa: E402
from mujoco_playground.config import locomotion_params          # noqa: E402

NAME = 'Go1JoystickFlatTerrain'
LEGS = ('FR', 'FL', 'RR', 'RL')                                 # actuator order in go1_mjx_feetonly.xml
TORSO = 1
FLOOR = 0
CMD_MAX = np.array([1.5, 0.5, 1.0])                             # vx, vy, wz (the training ranges: vx 1.5, vy 0.8, wz 1.2)
K_POS, K_YAW = 1.5, 2.0


def make_env():
    cfg = registry.get_default_config(NAME)
    cfg.noise_config.level = 0.0
    return registry.load(NAME, config=cfg)


def perturb_model(m, weak_leg=None, payload=None, foam=None):
    upd = {}
    if weak_leg is not None:
        leg, scale = weak_leg
        i = LEGS.index(leg) * 3
        upd['actuator_forcerange'] = m.actuator_forcerange.at[i:i + 3].multiply(scale)
    if payload is not None:
        mp, dy, dz = payload
        M = m.body_mass[TORSO]
        c = m.body_ipos[TORSO]
        r = c + jnp.array([0.0, dy, dz])
        cn = (M * c + mp * r) / (M + mp)
        d = r - cn
        dI = mp * jnp.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
        dc = cn - c                                              # the trunk's own inertia about the shifted CoM
        dI = dI + M * jnp.array([dc[1] ** 2 + dc[2] ** 2, dc[0] ** 2 + dc[2] ** 2, dc[0] ** 2 + dc[1] ** 2])
        upd['body_mass'] = m.body_mass.at[TORSO].set(M + mp)
        upd['body_ipos'] = m.body_ipos.at[TORSO].set(cn)
        upd['body_inertia'] = m.body_inertia.at[TORSO].add(dI)
    if foam is not None:
        tc, width, dr = (tuple(foam) + (1.0,))[:3]                # optional third entry: contact damping ratio
        upd['geom_solref'] = m.geom_solref.at[FLOOR].set(jnp.array([tc, dr]))
        upd['geom_solimp'] = m.geom_solimp.at[FLOOR].set(jnp.array([0.0, 0.95, width, 0.5, 2.0]))
    return m.tree_replace(upd) if upd else m


def load_policy(env, params_path, network_cfg=None):
    nf = network_cfg or locomotion_params.brax_ppo_config(NAME).network_factory
    net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                         preprocess_observations_fn=running_statistics.normalize, **nf)
    params = brax_model.load_params(params_path)
    return params, net


def policy_apply(net, params, obs):
    """Deterministic action (tanh of the mean), differentiable in obs."""
    logits = net.policy_network.apply(params[0], params[1], obs)
    mean = jnp.split(logits, 2, axis=-1)[0]
    return jnp.tanh(mean)


def yaw_of(q):
    w, x, y, z = q
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


V_REF, A_REF, T0 = 0.8, 1.0, 0.5                               # reference speed (m/s), accel, standing start (s)


def reference(t, goal):
    """Straight start->goal reference: trapezoidal along-track speed; returns (p_ref, v_ref, heading).
    goal = (x, y) or (x, y, speed): the optional third entry is the reference speed (default V_REF)."""
    vmax = goal[2] if goal.shape[0] == 3 else V_REF
    g = goal[:2]
    L = jnp.linalg.norm(g)
    u = g / L
    tt = jnp.maximum(t - T0, 0.0)
    t_acc = vmax / A_REF
    s_acc = 0.5 * A_REF * jnp.minimum(tt, t_acc) ** 2
    s = s_acc + vmax * jnp.maximum(tt - t_acc, 0.0)
    v = jnp.where(tt < t_acc, A_REF * tt, vmax)
    t_dec = 0.5 * vmax ** 2 / A_REF                               # decelerate into the goal
    rem = L - s
    v = jnp.where(rem < t_dec, jnp.sqrt(jnp.maximum(2 * A_REF * rem, 0.0)), v)
    s = jnp.minimum(s, L)
    v = jnp.where(s >= L, 0.0, v)
    return s * u, v * u, jnp.arctan2(u[1], u[0])


def ref_end_time(goal):
    """When the reference point reaches the goal (s): standing start, accelerate, cruise (L >= vmax^2 / 2 A_REF)."""
    vmax = goal[2] if goal.shape[0] == 3 else V_REF
    return T0 + jnp.linalg.norm(goal[:2]) / vmax + vmax / (2 * A_REF)


STAND_DELAY = 0.5                                              # s after ref_end_time: the walker settles, then stands


def command(p, yaw, goal, t):
    """Track the reference: body-frame v = v_ref + K (p_ref - p), yaw rate to the line heading."""
    pr, vr, head = reference(t, goal)
    d = vr + K_POS * (pr - p[:2])
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    db = jnp.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    v = jnp.clip(db, -CMD_MAX[:2], CMD_MAX[:2])
    eh = jnp.arctan2(jnp.sin(head - yaw), jnp.cos(head - yaw))
    wz = jnp.clip(K_YAW * eh, -CMD_MAX[2], CMD_MAX[2])
    return jnp.array([v[0], v[1], wz])


def track_errors(p, yaw, goal, t):
    """(cross-track, along-track lag, heading) errors vs the reference at time t (m, m, rad)."""
    pr, _, head = reference(t, goal)
    u = goal[:2] / jnp.linalg.norm(goal[:2])
    d = p[:2] - pr
    cross = u[0] * d[1] - u[1] * d[0]
    along = u[0] * d[0] + u[1] * d[1]
    eh = jnp.arctan2(jnp.sin(yaw - head), jnp.cos(yaw - head))
    return cross, along, eh


class Walker:
    def __init__(self, params_path, stand_at_goal=False, **perturb):
        """stand_at_goal: once the reference has reached the goal (ref_end_time), the walker is replaced by the standard
        standing controller (joint PD to the home pose, action 0) -- the task is getting between goals."""
        self.stand_at_goal = stand_at_goal
        self.env = make_env()
        self.env._mjx_model = perturb_model(self.env.mjx_model, **perturb)
        self.params, self.net = load_policy(self.env, params_path)
        self._roll = jax.jit(self._rollout, static_argnames=('n',))

    def _rollout(self, goal, du, key, n):
        env, net, params = self.env, self.net, self.params
        st = env.reset(key)
        st.info['steps_until_next_cmd'] = jnp.asarray(10 ** 6, st.info['steps_until_next_cmd'].dtype)
        # the same start every walk: the origin, at rest, facing the goal (Playground resets randomize all three)
        head = jnp.arctan2(goal[1], goal[0])
        qpos = jnp.asarray(env._init_q).at[0:2].set(0.0).at[3:7].set(
            jnp.array([jnp.cos(head / 2), 0.0, 0.0, jnp.sin(head / 2)]))
        data = st.data.replace(qpos=qpos, qvel=jnp.zeros_like(st.data.qvel), ctrl=jnp.asarray(env._default_pose))
        st = st.replace(data=mjx.forward(env.mjx_model, data))

        def body(st, k):
            q = st.data.qpos
            cmd = command(q[:3], yaw_of(q[3:7]), goal, k * 0.02)
            st.info['command'] = cmd
            obs = env._get_obs(st.data, st.info)
            a = policy_apply(net, params, obs) + du[k]
            if self.stand_at_goal:
                a = jnp.where(k * 0.02 >= ref_end_time(goal) + STAND_DELAY, jnp.zeros_like(a), a)
            st2 = env.step(st.replace(obs=obs), a)
            out = dict(p=q[:3], quat=q[3:7], qj=q[7:], v=st.data.qvel[:6], qd=st.data.qvel[6:], cmd=cmd, a=a,
                       contact=st.info['last_contact'], obs_state=obs['state'],
                       tau=st2.data.actuator_force, q_after=st2.data.qpos[7:])   # measured motor torque (Go1: tauEst)
            return st2, out

        st, out = jax.lax.scan(body, st, jnp.arange(n))
        return out

    def rollout(self, goal, n=250, du=None, seed=0):
        du = jnp.zeros((n, 12)) if du is None else jnp.asarray(du)
        return jax.tree.map(np.asarray, self._roll(jnp.asarray(goal, jnp.float32), du, jax.random.PRNGKey(seed), n=n))


def fell(out):
    q = out['quat']
    up_z = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)                 # body z . world z
    return bool(np.any(up_z < np.cos(np.radians(45))) or np.any(out['p'][:, 2] < 0.15))
