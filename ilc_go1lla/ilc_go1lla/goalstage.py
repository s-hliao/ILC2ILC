"""Stage pieces for the GOAL-CONDITIONED walker (2026-11-24): a simulated robot (nominal or perturbed) that walks the goal walker
to a goal and logs the fields srb.make_G reads, the network's goal observation as a JAX function (srb's cmd_fn hook), and the
three row definitions the user asked to compare (srb's row_fn hook):

  time      vs the BASE network's walk to the same goal on the NOMINAL sim robot, at the same time: cross / along (line frame
            of start->goal, / pos_scale), heading (/ yaw_scale), roll, pitch (/ ATT_SCALE)                       -> 5 rows
  progress  vs the same nominal walk at the matched point of its PATH (nearest point, ties broken toward the same time):
            lateral (perpendicular to the path there), heading at that point, time lag at the same progress (/ 0.1 s, weighted
            by the nominal speed there, 0 while the nominal robot stands), roll, pitch                             -> 5 rows
  goal      every window against the FINAL error: position minus the goal (along / across the start->goal line, / pos_scale)
                                                                                                                   -> 2 rows
Start of every walk: the origin, at rest, heading +x; the goal (x, y) is in that frame (any bearing).
"""
import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from . import goalwalk as GW
from . import walkenv as WE
from .srb import ATT_SCALE, quat_R

DT = 0.02
NRS = dict(time=5, progress=5, goal=2)


def walk_steps(goal, W=10, extra=4.0):
    """Control steps for one walk: the straight-line time at VMAX + `extra` s, a multiple of W, + 1."""
    t = float(np.linalg.norm(np.asarray(goal)[:2])) / GW.VMAX + extra
    return int(np.ceil(t / DT / W) * W) + 1


class GoalSimWalker:
    def __init__(self, params_path, variant='vel2', cone=None, **perturb):
        """cone='elliptic' (11-24): MuJoCo's elliptic friction cones (the deployment-foam model: with the default pyramidal cones a
        soft floor holds the feet far more firmly than the real foam; runs/foamfit)."""
        cfg = GW.default_config(variant)
        cfg.noise_config.level = 0.0
        self.env = GW.GoalWalker(config=cfg)
        mm = WE.perturb_model(self.env.mjx_model, **perturb)
        if cone == 'elliptic':
            import mujoco
            mm = mm.replace(opt=mm.opt.replace(cone=int(mujoco.mjtCone.mjCONE_ELLIPTIC)))
        self.env._mjx_model = mm
        self.params, self.net = WE.load_policy(self.env, params_path)
        self._roll = jax.jit(self._rollout, static_argnames=('n',))

    def _rollout(self, goal, du, key, n):
        env, net, params = self.env, self.net, self.params
        st = env.reset(key)
        qpos = jnp.asarray(env._init_q).at[0:2].set(0.0).at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
        data = st.data.replace(qpos=qpos, qvel=jnp.zeros_like(st.data.qvel), ctrl=jnp.asarray(env._default_pose))
        data = mjx.forward(env.mjx_model, data)
        info = dict(st.info, goal=goal, d_prev=jnp.linalg.norm(goal), at_goal=jnp.zeros((), jnp.int32))
        st = st.replace(data=data, info=info)

        def body(st, k):
            q = st.data.qpos
            obs = env._get_obs(st.data, st.info)
            a = WE.policy_apply(net, params, obs) + du[k]
            st2 = env.step(st.replace(obs=obs), a)
            st2.info['goal'] = goal                               # ONE goal per walk (training draws a new one after the hold)
            st2.info['at_goal'] = jnp.zeros((), jnp.int32)
            out = dict(p=q[:3], quat=q[3:7], qj=q[7:], v=st.data.qvel[:6], qd=st.data.qvel[6:], cmd=obs['state'][-3:], a=a,
                       contact=st.info['last_contact'], obs_state=obs['state'], tau=st2.data.actuator_force, q_after=st2.data.qpos[7:])
            return st2, out

        _, out = jax.lax.scan(body, st, jnp.arange(n))
        return out

    def rollout(self, goal, n, du=None, seed=0):
        du = jnp.zeros((n, 12)) if du is None else jnp.asarray(du)
        return jax.tree.map(np.asarray, self._roll(jnp.asarray(goal, jnp.float32)[:2], du, jax.random.PRNGKey(seed), n=n))


def cmd_fn_for(goal):
    """goalwalk.GoalWalker._goal_obs in the walk frame (srb.make_G's cmd_fn)."""
    goal = jnp.asarray(goal, jnp.float32)[:2]

    def f(p, yaw, k):
        dw = goal - p[:2]
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        gb = jnp.array([c * dw[0] + s * dw[1], -s * dw[0] + c * dw[1]])
        d = jnp.linalg.norm(gb)
        gbc = gb * jnp.minimum(1.0, GW.OBS_R / jnp.maximum(d, 1e-6))
        return jnp.array([gbc[0], gbc[1], jnp.where(d > GW.ARRIVE, jnp.arctan2(gb[1], gb[0]), 0.0)])
    return f


def _wrap(x):
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


def _rpy(R):
    return jnp.arctan2(R[2, 1], R[2, 2]), -jnp.arcsin(jnp.clip(R[2, 0], -1.0, 1.0)), jnp.arctan2(R[1, 0], R[0, 0])


def reference(nom_out):
    """The nominal walk as arrays: p (n, 2), yaw (n,), unit path tangents (n, 2), speed (n,)."""
    p = np.asarray(nom_out['p'][:, :2], np.float32)
    R = np.stack([np.asarray(quat_R(jnp.asarray(q))) for q in nom_out['quat']])
    yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0])).astype(np.float32)
    dp = np.gradient(p, DT, axis=0)
    sp = np.linalg.norm(dp, axis=1)
    tan = dp / np.maximum(sp, 1e-6)[:, None]
    return dict(p=p, yaw=yaw, tan=tan.astype(np.float32), speed=sp.astype(np.float32), n=len(p))


def row_fn_for(kind, goal, ref=None, pos_scale=0.05, yaw_scale=0.1, lag_scale=0.1, tie_w=0.05):
    goal = jnp.asarray(goal, jnp.float32)[:2]
    u = goal / jnp.linalg.norm(goal)
    nrm = jnp.array([-u[1], u[0]])
    if ref is not None:
        P, Y, T, V = (jnp.asarray(ref[k]) for k in ('p', 'yaw', 'tan', 'speed'))
        N = ref['n']
        tref = jnp.arange(N) * DT

    if kind == 'time':
        def f(p, R, t):
            k = jnp.clip(jnp.round(t / DT).astype(jnp.int32), 0, N - 1)
            e = p[:2] - P[k]
            roll, pitch, yaw = _rpy(R)
            return jnp.array([e @ nrm / pos_scale, e @ u / pos_scale, _wrap(yaw - Y[k]) / yaw_scale, roll / ATT_SCALE,
                              pitch / ATT_SCALE])
    elif kind == 'progress':
        def f(p, R, t):
            cost = jnp.sum((P - p[None, :2]) ** 2, 1) + (tie_w * (tref - t)) ** 2
            j = jax.lax.stop_gradient(jnp.argmin(cost))
            e = p[:2] - P[j]
            tj = jnp.where(V[j] > 0.1, T[j], u)                   # the path's direction there (the line while it turns / stands)
            nj = jnp.array([-tj[1], tj[0]])
            roll, pitch, yaw = _rpy(R)
            moving = jnp.minimum(1.0, V[j] / 0.3)
            lag = (t - tref[j] - (e @ tj) / jnp.maximum(V[j], 0.2)) * moving
            return jnp.array([e @ nj / pos_scale, _wrap(yaw - Y[j]) / yaw_scale, lag / lag_scale, roll / ATT_SCALE,
                              pitch / ATT_SCALE])
    elif kind == 'goal':
        def f(p, R, t):
            e = p[:2] - goal
            return jnp.array([e @ u / pos_scale, e @ nrm / pos_scale])
    else:
        raise ValueError(kind)
    return f
