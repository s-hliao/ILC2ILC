"""SRB-linearized G with a CONTACT SCHEDULE THAT MOVES (2026-11-24; user-approved method change for the goal-walker stage).

srb.make_G holds the measured stance schedule fixed: a foot touches down / lifts off exactly when it did on the trial, so the
model cannot represent "this foot should have landed but hung in the air" or "the trunk tilt unloaded this foot", the
precursors of light falls. Here, on top of srb.make_G's single rigid body + stance PD springs + policy feedback:

  ground       NO ground model: each swing foot's contact height is where that foot's NEXT MEASURED touchdown happened
               (touchdown_heights: the surface as the foot met it -- floor, foam top, a gap; same foot, frame and place, so mocap
               offsets cancel); the model itself stays NOMINAL and rigid (no foam: foam is a test robot, not part of the model)
  swing feet   predicted position = trunk pose (+ perturbation) carrying the body-frame foot at its measured place plus the
               leg Jacobian times the change of its joint target  (as srb's touchdown footholds)
  touchdown    a swing foot lands when its predicted height reaches its measured touchdown height: land = (1 - m) sigmoid(-h / EPS_H),
               h = z_pred - z_touchdown (first-order event timing around the measured touchdown)
  lift-off     a stance foot stays only while its predicted vertical force is positive:  stay = m sigmoid(Fz / EPS_F)
  contact m    soft contact state per foot, m' = clip(stay + land, 0, 1); stance force = m sigmoid(Fz / EPS_F) x the PD spring
  anchors      a stance foot keeps its anchor; a landing foot anchors where its free foot is
As in srb.make_G the perturbation is propagated as x_meas (+) dx (contact state: m = measured stance + dm), so at du = 0 the
model reproduces the measured walk exactly and d rows / d du is the closed-loop linearization, now including contact timing.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import walkenv as WE
from .srb import DT, NR, N_SUB, G as GRAV, expm, measured, saturation

EPS_H = 0.01       # m: touchdown smoothing (foot height above the fitted ground)
EPS_F = 10.0       # N: lift-off smoothing (predicted vertical stance force)


def touchdown_heights(meas):
    """(11-24, replaces a fitted ground plane: on foam the loaded stance feet sit cm below where a swing foot meets the surface)
    Per step and foot: the height at which THAT foot's next measured touchdown happened (stance = load from the measured torques).
    After a foot's last touchdown (or if it never lands again) its height at its last measured stance step. Same frame and same
    foot as the prediction, so mocap offsets cancel; no ground model."""
    fz = np.asarray(meas['feet_w'])[:, :, 2]
    c = np.asarray(meas['c']) > 0.5
    n = len(fz)
    H = np.zeros((n, 4), np.float32)
    for j in range(4):
        nxt = None
        last = fz[np.where(c[:, j])[0][-1], j] if c[:, j].any() else fz[:, j].min()
        for k in range(n - 1, -1, -1):
            if c[k, j] and (k == 0 or not c[k - 1, j]):
                nxt = fz[k, j]
            H[k, j] = nxt if nxt is not None else last
    return H


def make_G_contact(walker, out, goal, W=10, pos_scale=0.05, yaw_scale=0.1, measured_sat=True, cmd_fn=None, row_fn=None, nr=None,
                   eps_h=EPS_H, eps_f=EPS_F, kin=None):
    from .srb import LegKin
    assert row_fn is not None, 'srbc: pass row_fn (goalstage.row_fn_for)'
    env = walker.env
    kin = kin or LegKin(env.mj_model)
    NRr = NR if nr is None else nr
    q0 = np.asarray(env._default_pose)
    M, com_b, I_b = kin.body_inertia(q0)
    Kp, Kd = float(env._config.Kp), float(env._config.Kd)
    s_act = float(env._config.action_scale)
    frange = np.asarray(env.mj_model.actuator_forcerange)[:, 1]
    meas_np = measured(out, kin)
    HTD = jnp.asarray(touchdown_heights(meas_np))
    meas = {k: jnp.asarray(v, jnp.float32) if isinstance(v, np.ndarray) else v for k, v in meas_np.items()}
    sat = jnp.asarray(saturation(out, env) if measured_sat else np.zeros((len(out['a']), 12), bool))
    n = meas['n']
    goal = jnp.asarray(goal, jnp.float32)
    com_b, I_b, q0 = (jnp.asarray(z, jnp.float32) for z in (com_b, I_b, q0))
    pol = functools.partial(WE.policy_apply, walker.net, walker.params)
    c = meas['c']
    Jinv = jnp.linalg.inv(meas['J'])

    def leg_state(p, R, v, w, k, r_feet):
        """Stance-leg joints implied by the feet held at r_feet (IK linearized at the measured q)."""
        fb_hat = jnp.einsum('ji,fj->fi', R, r_feet - p[None])
        dq = jnp.einsum('fij,fj->fi', Jinv[k], fb_hat - meas['fb'][k])
        q_hat = meas['q'][k].reshape(4, 3) + dq
        w_b = R.T @ w
        vfoot_b = -jnp.cross(w_b[None], fb_hat) - (R.T @ v)[None]
        qd_hat = jnp.einsum('fij,fj->fi', Jinv[k], vfoot_b)
        return q_hat, qd_hat

    def raw_forces(p, R, v, w, a, k, r_feet):
        """World stance-spring force per foot (not yet gated by contact), from the PD the walker stands on."""
        q_hat, qd_hat = leg_state(p, R, v, w, k, r_feet)
        tau_pd = jnp.clip(Kp * (q0.reshape(4, 3) + s_act * a.reshape(4, 3) - q_hat), -frange.reshape(4, 3), frange.reshape(4, 3))
        tau = jnp.where(sat[k].reshape(4, 3), meas['tau'][k].reshape(4, 3), tau_pd) - Kd * qd_hat
        F_b = -jnp.einsum('fji,fj->fi', Jinv[k], tau)
        return jnp.einsum('ij,fj->fi', R, F_b), q_hat, qd_hat

    def raw_obs(p, R, v, w, qj, qdj, last_a, k):
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        cmd = WE.command(p, yaw, goal, k * DT) if cmd_fn is None else cmd_fn(p, yaw, k)
        return jnp.concatenate([R.T @ v, R.T @ w, R.T @ jnp.array([0.0, 0.0, -1.0]), qj.ravel() - q0, qdj.ravel(), last_a, cmd])

    def obs_of(p, R, v, w, q_hat, qd_hat, last_a, mc, k):
        st = mc[:, None]
        qm, qdm = meas['q'][k].reshape(4, 3), meas['qd'][k].reshape(4, 3)
        qh0, qdh0 = leg_state(meas['p'][k], meas['R'][k], meas['v'][k], meas['w'][k], k, meas['feet_w'][k])
        s0 = c[k][:, None]
        x1 = raw_obs(p, R, v, w, st * q_hat + (1 - st) * qm, st * qd_hat + (1 - st) * qdm, last_a, k)
        x0 = raw_obs(meas['p'][k], meas['R'][k], meas['v'][k], meas['w'][k], s0 * qh0 + (1 - s0) * qm, s0 * qdh0 + (1 - s0) * qdm,
                     meas['last_a'][k], k)
        o = meas['obs'][k]
        return (o + (x1 - x0)).astype(o.dtype)

    def srb_step(p, R, v, w, F, r_feet):
        Iw_inv = R @ jnp.linalg.inv(I_b) @ R.T
        pc = p + R @ com_b
        tau = jnp.sum(jnp.cross(r_feet - pc[None], F), 0)
        acc = jnp.sum(F, 0) / M + GRAV
        h = DT / N_SUB
        for _ in range(N_SUB):
            v = v + h * acc
            w = w + h * (Iw_inv @ tau)
            p = p + h * v
            R = expm(h * w) @ R
        return p, R, v, w

    def full_step(state, k, du_k):
        p, R, v, w, r_feet, last_a, mc = state
        _, q_hat, qd_hat = raw_forces(p, R, v, w, meas['a'][k], k, r_feet)
        a = pol({'state': obs_of(p, R, v, w, q_hat, qd_hat, last_a, mc, k)}) + du_k
        Fr, _, _ = raw_forces(p, R, v, w, a, k, r_feet)
        load = jax.nn.sigmoid(Fr[:, 2] / eps_f)
        F = Fr * (mc * load)[:, None]
        p2, R2, v2, w2 = srb_step(p, R, v, w, F, r_feet)
        kk = jnp.minimum(k + 1, n - 1)
        fb_new = meas['fb'][kk] + jnp.einsum('fij,fj->fi', meas['J'][kk], s_act * (a - meas['a'][k]).reshape(4, 3))
        x_free = p2[None] + jnp.einsum('ij,fj->fi', R2, fb_new)
        h = x_free[:, 2] - HTD[kk]
        stay = mc * load
        land = (1 - mc) * jax.nn.sigmoid(-h / eps_h)
        mc2 = jnp.clip(stay + land, 0.0, 1.0)
        wsum = stay + land + 1e-6
        r2 = (stay[:, None] * r_feet + land[:, None] * x_free) / wsum[:, None]
        r2 = jnp.where((wsum > 1e-3)[:, None], r2, x_free)
        return (p2, R2, v2, w2, r2, a, mc2)

    def rollout_rows(du):
        def body(dx, k):
            dp, dth, dv, dw, dr, da_last, dm = dx
            Rm = meas['R'][k]
            pert = (meas['p'][k] + dp, expm(dth) @ Rm, meas['v'][k] + dv, meas['w'][k] + dw, meas['feet_w'][k] + dr,
                    meas['last_a'][k] + da_last, jnp.clip(c[k] + dm, 0.0, 1.0))
            base = (meas['p'][k], Rm, meas['v'][k], meas['w'][k], meas['feet_w'][k], meas['last_a'][k], c[k])
            s1 = full_step(pert, k, du[k])
            s0 = full_step(base, k, jnp.zeros(12))
            dR = s1[1] @ s0[1].T
            dth2 = 0.5 * jnp.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            dx2 = (s1[0] - s0[0], dth2, s1[2] - s0[2], s1[3] - s0[3], s1[4] - s0[4], s1[5] - s0[5], s1[6] - s0[6])
            kk = jnp.minimum(k + 1, n - 1)
            pk = meas['p'][kk] + dx2[0]
            Rk = expm(dth2) @ meas['R'][kk]
            rv = row_fn(pk, Rk, (k + 1) * DT)
            return dx2, rv
        dx0 = (jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), jnp.zeros((4, 3)), jnp.zeros(12), jnp.zeros(4))
        _, r = jax.lax.scan(body, dx0, jnp.arange(n - 1))
        nw = (n - 1) // W
        return r[:nw * W].reshape(nw, W, NRr).mean(1).ravel()

    du0 = jnp.zeros((n, 12), jnp.float32)
    rows = np.asarray(jax.jit(rollout_rows)(du0))
    Gm = np.asarray(jax.jit(jax.jacrev(rollout_rows))(du0)).reshape(len(rows), n * 12)
    return rows, Gm
