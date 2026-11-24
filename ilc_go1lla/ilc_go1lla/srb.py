"""SRB-linearized G for the joint-target Go1 walker (no contact simulated: the stance schedule is the measured one).

Along a measured walk (p, R, v, omega, q, qd, foot positions, contact flags, actions), the nominal single rigid body
(the Go1's total mass and composite inertia at the default pose) with:
  stance legs  F_i = -R J_i^-T [Kp (q0 + s a_i - q_i(x)) - Kd qd_i(x)]  (the PD spring the walker stands on), q_i(x) from
               the foot held where it was measured (IK linearized, J_i the body-frame foot Jacobian at the measured q)
  footholds    a leg touching down lands at p + R FK(q_i + s (a_i - a_i,meas))   (foot placement depends on the trunk state
               and on the policy's swing targets: the stabilizer the earlier SRBD test lacked)
  feedback     a = pi(obs(x)) + du, obs rebuilt from the perturbed trunk state (local linvel, gyro, gravity, stance joints,
               last action, the command law's command)
The perturbation dynamics are propagated as x_hat = x_meas (+) dx, so at du = 0 dx = 0 and d rows / d du is the closed-loop
linearization. Rows: path deviation from the start->goal line and heading, averaged over windows of W control steps.
"""
import functools

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from . import walkenv as WE

DT = 0.02
NR = 5             # rows per window: cross-track, along-track (/5 cm), heading, roll, pitch (/0.1 rad)
ATT_SCALE = 0.1
N_SUB = 5
G = jnp.array([0.0, 0.0, -9.81])


def skew(w):
    return jnp.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def expm(th):
    a = jnp.sqrt(jnp.sum(th ** 2) + 1e-12)
    K = skew(th / a)
    return jnp.eye(3) + jnp.sin(a) * K + (1 - jnp.cos(a)) * K @ K


def quat_R(q):
    w, x, y, z = q
    return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class LegKin:
    """Body-frame foot positions and leg Jacobians from the Playground Go1 model (CPU MuJoCo, base at identity)."""

    def __init__(self, mj_model):
        self.m = mj_model
        self.d = mujoco.MjData(mj_model)
        from mujoco_playground._src.locomotion.go1 import go1_constants as C
        self.sites = [mj_model.site(s).id for s in C.FEET_SITES]          # FR, FL, RR, RL (actuator order)

    def at(self, q):
        d = self.d
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[7:] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        pos = np.zeros((4, 3))
        J = np.zeros((4, 3, 3))
        jac = np.zeros((3, self.m.nv))
        for i, s in enumerate(self.sites):
            pos[i] = d.site_xpos[s]
            mujoco.mj_jacSite(self.m, d, jac, None, s)
            J[i] = jac[:, 6 + 3 * i: 9 + 3 * i]
        return pos, J

    def body_inertia(self, q0):
        """Total mass, CoM (body frame) and composite inertia about it, at joint pose q0."""
        m, d = self.m, self.d
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[7:] = q0
        mujoco.mj_kinematics(m, d)
        M = float(m.body_mass[1:].sum())
        com = (m.body_mass[1:, None] * d.xipos[1:]).sum(0) / M
        I = np.zeros((3, 3))
        for b in range(1, m.nbody):
            Rb = d.ximat[b].reshape(3, 3)
            Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
            r = d.xipos[b] - com
            I += Ib + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
        return M, com, I


def measured(out, kin):
    """Per-step arrays the linearization needs (numpy)."""
    n = len(out['p'])
    R = np.stack([np.asarray(quat_R(jnp.asarray(q))) for q in out['quat']])
    fb, J = zip(*[kin.at(q) for q in out['qj']])
    fb, J = np.stack(fb), np.stack(J)
    feet_w = out['p'][:, None, :] + np.einsum('tij,tfj->tfi', R, fb)
    omega_w = np.einsum('tij,tj->ti', R, out['v'][:, 3:6])               # free-joint qvel[3:6] is body-frame
    return dict(n=n, p=out['p'], R=R, v=out['v'][:, :3], w=omega_w, q=out['qj'], qd=out['qd'], fb=fb, J=J,
                feet_w=feet_w, c=load_stance(out, R, J), a=out['a'], obs=out['obs_state'],
                last_a=np.concatenate([np.zeros((1, 12)), out['a'][:-1]], 0), tau=out['tau'], q_after=out['q_after'])


SAT_TOL = 0.5      # N m: a motor delivering this much less than its PD command is saturated (or disabled)


def saturation(out, env):
    """Per step and motor: True where the MEASURED torque falls short of the PD command (a weak / locked / capped motor).
    Measured on hardware as tauEst vs Kp (q_target - q); no motor parameters are identified."""
    q0 = np.asarray(env._default_pose)
    tau_cmd = float(env._config.Kp) * (q0 + float(env._config.action_scale) * out['a'] - out['q_after'])
    return np.abs(tau_cmd) > np.abs(out['tau']) + SAT_TOL


FZ_STANCE = 20.0   # N: a leg is in stance when the vertical ground force implied by its MEASURED torques exceeds this


def load_stance(out, R=None, J=None, kin=None):
    """Stance flags from load, not geometry: Fz of -R J^-T tau_meas per leg (on foam a swinging foot is still inside the
    soft contact zone, so geometric contact reads 'down'; on hardware the same signal comes from tauEst)."""
    if J is None:
        fbJ = [kin.at(q) for q in out['qj']]
        J = np.stack([j for _, j in fbJ])
    if R is None:
        R = np.stack([np.asarray(quat_R(jnp.asarray(q))) for q in out['quat']])
    tau = out['tau'].reshape(-1, 4, 3)
    F_b = -np.einsum('tfji,tfj->tfi', np.linalg.inv(J), tau)
    Fz = np.einsum('tij,tfj->tfi', R, F_b)[..., 2]
    return (Fz > FZ_STANCE).astype(np.float32)


def make_G(walker, out, goal, W=10, pos_scale=0.05, yaw_scale=0.1, measured_sat=True, cmd_fn=None, row_fn=None, nr=None):
    """(11-24 hooks, default = unchanged) cmd_fn(p, yaw, k): the observation's last 3 entries (goal walker: the goal in the
    body frame); row_fn(p, R, t): the row vector at time t (length nr)."""
    NRr = NR if nr is None else nr
    """Rows (windowed cross-track, along-track, heading errors vs the reference) and G = d rows / d du (n x 12) by reverse-mode autodiff."""
    env = walker.env
    kin = LegKin(env.mj_model)
    q0 = np.asarray(env._default_pose)
    M, com_b, I_b = kin.body_inertia(q0)
    Kp = float(env._config.Kp)
    Kd = float(env._config.Kd)
    s_act = float(env._config.action_scale)
    frange = np.asarray(env.mj_model.actuator_forcerange)[:, 1]          # NOMINAL caps (the model's belief)
    meas = {k: jnp.asarray(v, jnp.float32) if isinstance(v, np.ndarray) else v for k, v in measured(out, kin).items()}
    sat = jnp.asarray(saturation(out, env) if measured_sat else np.zeros((len(out['a']), 12), bool))
    n = meas['n']
    goal = jnp.asarray(goal, jnp.float32)
    line = goal / jnp.linalg.norm(goal)
    head = jnp.arctan2(line[1], line[0])
    com_b, I_b, q0 = (jnp.asarray(z, jnp.float32) for z in (com_b, I_b, q0))
    pol = functools.partial(WE.policy_apply, walker.net, walker.params)
    # touchdown flags: contact now, not at the previous step
    c = meas['c']
    td = c * (1 - jnp.concatenate([c[:1], c[:-1]], 0))

    def forces(p, R, v, w, a, k, r_feet):
        """Stance forces (world) at state (p, R, v, w) with action a; feet held at r_feet."""
        fb_hat = jnp.einsum('ji,fj->fi', R, r_feet - p[None])                # R^T (r - p)
        dq = jnp.einsum('fij,fj->fi', jnp.linalg.inv(meas['J'][k]), fb_hat - meas['fb'][k])
        q_hat = meas['q'][k].reshape(4, 3) + dq
        w_b = R.T @ w
        vfoot_b = -jnp.cross(w_b[None], fb_hat) - (R.T @ v)[None]           # d/dt R^T (r - p), r fixed
        qd_hat = jnp.einsum('fij,fj->fi', jnp.linalg.inv(meas['J'][k]), vfoot_b)
        tau_pd = jnp.clip(Kp * (q0.reshape(4, 3) + s_act * a.reshape(4, 3) - q_hat), -frange.reshape(4, 3),
                          frange.reshape(4, 3))
        # a saturated / disabled motor gives its MEASURED torque whatever the target or the posture
        tau = jnp.where(sat[k].reshape(4, 3), meas['tau'][k].reshape(4, 3), tau_pd) - Kd * qd_hat
        F_b = -jnp.einsum('fji,fj->fi', jnp.linalg.inv(meas['J'][k]), tau)  # -J^-T tau  (body frame)
        F = jnp.einsum('ij,fj->fi', R, F_b) * c[k][:, None]
        return F, q_hat, qd_hat

    def raw_obs(p, R, v, w, qj, qdj, last_a, k):
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        cmd = WE.command(p, yaw, goal, k * DT) if cmd_fn is None else cmd_fn(p, yaw, k)
        return jnp.concatenate([R.T @ v, R.T @ w, R.T @ jnp.array([0.0, 0.0, -1.0]), qj.ravel() - q0, qdj.ravel(),
                                last_a, cmd])

    def obs_of(p, R, v, w, q_hat, qd_hat, last_a, k):
        """The measured observation plus the model's change of it (so at the measured state pi gives a_meas exactly)."""
        st = c[k][:, None]
        qm, qdm = meas['q'][k].reshape(4, 3), meas['qd'][k].reshape(4, 3)
        _, qh0, qdh0 = forces(meas['p'][k], meas['R'][k], meas['v'][k], meas['w'][k], meas['a'][k], k, meas['feet_w'][k])
        x1 = raw_obs(p, R, v, w, st * q_hat + (1 - st) * qm, st * qd_hat + (1 - st) * qdm, last_a, k)
        x0 = raw_obs(meas['p'][k], meas['R'][k], meas['v'][k], meas['w'][k], st * qh0 + (1 - st) * qm,
                     st * qdh0 + (1 - st) * qdm, meas['last_a'][k], k)
        o = meas['obs'][k]
        return (o + (x1 - x0)).astype(o.dtype)

    def srb_step(p, R, v, w, F, r_feet):
        Iw_inv = R @ jnp.linalg.inv(I_b) @ R.T
        pc = p + R @ com_b
        tau = jnp.sum(jnp.cross(r_feet - pc[None], F), 0)
        acc = jnp.sum(F, 0) / M + G
        h = DT / N_SUB
        for _ in range(N_SUB):
            v = v + h * acc
            w = w + h * (Iw_inv @ tau)
            p = p + h * v
            R = expm(h * w) @ R
        return p, R, v, w

    def full_step(state, k, du_k):
        p, R, v, w, r_feet, last_a = state
        # policy action at this (perturbed) state; stance q from the feet held
        F0, q_hat, qd_hat = forces(p, R, v, w, meas['a'][k], k, r_feet)
        a = pol({'state': obs_of(p, R, v, w, q_hat, qd_hat, last_a, k)}) + du_k
        F, _, _ = forces(p, R, v, w, a, k, r_feet)
        p2, R2, v2, w2 = srb_step(p, R, v, w, F, r_feet)
        # footholds of legs touching down at k+1: FK of the (perturbed) joint targets carried by the trunk
        kk = jnp.minimum(k + 1, n - 1)
        fb_new = meas['fb'][kk] + jnp.einsum('fij,fj->fi', meas['J'][kk], s_act * (a - meas['a'][k]).reshape(4, 3))
        land = p2[None] + jnp.einsum('ij,fj->fi', R2, fb_new)
        r_feet = jnp.where(td[kk][:, None] > 0, land, r_feet)
        return (p2, R2, v2, w2, r_feet, a)

    def row_vec(pk, Rk, t):
        yaw = jnp.arctan2(Rk[1, 0], Rk[0, 0])
        cr, al, eh = WE.track_errors(pk, yaw, goal, t)
        roll = jnp.arctan2(Rk[2, 1], Rk[2, 2])
        pitch = -jnp.arcsin(jnp.clip(Rk[2, 0], -1.0, 1.0))
        return jnp.array([cr / pos_scale, al / pos_scale, eh / yaw_scale, roll / ATT_SCALE, pitch / ATT_SCALE])

    def rollout_rows(du):
        """Perturbed propagation as x_meas (+) dx: dx_{k+1} = F(x_meas_k (+) dx_k, du_k) (-) F(x_meas_k, 0)."""
        def body(dx, k):
            dp, dth, dv, dw, dr, da_last = dx
            Rm = meas['R'][k]
            pert = (meas['p'][k] + dp, expm(dth) @ Rm, meas['v'][k] + dv, meas['w'][k] + dw,
                    meas['feet_w'][k] + dr, meas['last_a'][k] + da_last)
            base = (meas['p'][k], Rm, meas['v'][k], meas['w'][k], meas['feet_w'][k], meas['last_a'][k])
            s1 = full_step(pert, k, du[k])
            s0 = full_step(base, k, jnp.zeros(12))
            dR = s1[1] @ s0[1].T
            dth2 = 0.5 * jnp.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            dx2 = (s1[0] - s0[0], dth2, s1[2] - s0[2], s1[3] - s0[3], s1[4] - s0[4], s1[5] - s0[5])
            # rows at k+1 on the measured + perturbation
            pk = meas['p'][jnp.minimum(k + 1, n - 1)] + dx2[0]
            Rk = expm(dth2) @ meas['R'][jnp.minimum(k + 1, n - 1)]
            return dx2, (row_vec(pk, Rk, (k + 1) * DT) if row_fn is None else row_fn(pk, Rk, (k + 1) * DT))
        dx0 = (jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), jnp.zeros((4, 3)), jnp.zeros(12))
        _, r = jax.lax.scan(body, dx0, jnp.arange(n - 1))
        nw = (n - 1) // W
        return r[:nw * W].reshape(nw, W, NRr).mean(1).ravel()

    du0 = jnp.zeros((n, 12), jnp.float32)
    rows = np.asarray(jax.jit(rollout_rows)(du0))
    Gm = np.asarray(jax.jit(jax.jacrev(rollout_rows))(du0)).reshape(len(rows), n * 12)
    return rows, Gm


def measured_rows(out, goal, W=10, pos_scale=0.05, yaw_scale=0.1, row_fn=None, nr=None):
    """The same rows from a flown walk (cross-track, along-track, heading, roll, pitch; windows of W steps).
    (11-24) row_fn(p, R, t) / nr: as make_G's hook."""
    goal = jnp.asarray(goal, jnp.float32)
    n = len(out['p'])
    if row_fn is not None:
        Rm = jnp.stack([quat_R(jnp.asarray(q, jnp.float32)) for q in out['quat'][1:]])
        r = np.asarray(jax.vmap(row_fn)(jnp.asarray(out['p'][1:], jnp.float32), Rm, jnp.arange(1, n) * DT))
        nw = len(r) // W
        return r[:nw * W].reshape(nw, W, nr).mean(1).ravel()
    R = np.stack([np.asarray(quat_R(jnp.asarray(q))) for q in out['quat'][1:]])
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    pitch = -np.arcsin(np.clip(R[:, 2, 0], -1, 1))
    t = np.arange(1, n) * DT
    cr, al, eh = jax.vmap(lambda p, y, tt: WE.track_errors(p, y, goal, tt))(jnp.asarray(out['p'][1:]), jnp.asarray(yaw), jnp.asarray(t))
    r = np.stack([np.asarray(cr) / pos_scale, np.asarray(al) / pos_scale, np.asarray(eh) / yaw_scale,
                  roll / ATT_SCALE, pitch / ATT_SCALE], 1)
    nw = len(r) // W
    return r[:nw * W].reshape(nw, W, NR).mean(1).ravel()


def fall_index(out):
    """First control step that counts as fallen (None if upright throughout)."""
    q = out['quat']
    up_z = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    bad = np.where((up_z < np.cos(np.radians(45))) | (out['p'][:, 2] < 0.15))[0]
    return int(bad[0]) if len(bad) else None


def fall_index_feet(out, kin, h_min=0.12, tilt_deg=45.0):
    """Fall test that works on a soft floor (the feet sink, so the trunk's world height alone misreads foam as a fall):
    fallen = trunk tilt > tilt_deg, or the trunk less than h_min above its lowest foot. kin: LegKin."""
    q = out['quat']
    up_z = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    h = np.empty(len(q))
    for k in range(len(q)):
        pos, _ = kin.at(out['qj'][k])
        h[k] = -(np.asarray(quat_R(q[k])) @ pos.T)[2].min()      # trunk height above its lowest foot
    bad = np.where((up_z < np.cos(np.radians(tilt_deg))) | (h < h_min))[0]
    return int(bad[0]) if len(bad) else None
