"""Go1 (Edu, Unitree high-level walking mode) carrying a soccer ball loose in a flat box on its back: the reduced model.

The policy commands the built-in walking controller's velocity set points c = (vx, vy, wz) (body frame) at 20 Hz; the
trunk's planar motion follows them through a command filter (latency) and a rate-limited first-order response. The ball
(size-5 soccer ball, R 0.11 m, hollow: rolling gain k = 1 / (1 + 2/3) = 0.6; its mass cancels) rolls on the box floor
under the box's inertial acceleration, the trunk tilt, and the rotating-frame terms; padded walls are a stiff,
damped smooth contact.

State x (16): X, Y, psi, vx, vy, wz (trunk velocity, body frame), bx, by, bvx, bvy (ball offset from the box centre and
its velocity, box = body frame), cf (3: the filtered command), a_prev (3: the last normalized action).

NOMINAL holds the nominal robot (Unitree's documented walking response, no hardware identification). PERTURBED is the
sim stand-in for the real robot (robot-side mismatches only; see its comment).
"""
import numpy as np
import jax.numpy as jnp

G = 9.81
BALL_R = 0.11
BOX_IN = (0.45, 0.35)                                  # inner length (x, along the trunk), width (y), m
TRAVEL = np.array([BOX_IN[0] / 2 - BALL_R, BOX_IN[1] / 2 - BALL_R])   # ball-centre travel from the box centre: 0.115, 0.065
CMD_LO = np.array([-0.6, -0.6, -2.0])                 # vx, vy (m/s), wz (rad/s) set-point box (fast trot in the room)
CMD_HI = np.array([1.5, 0.6, 2.0])
NX, NA = 16, 3

NOMINAL = dict(tau_c=0.03, tau_v=0.20, tau_w=0.15, a_max=2.5, al_max=8.0, gain_v=1.0, gain_w=1.0,
               k_roll=0.6, c_rr=0.3, tilt_k=0.01, tilt0_x=0.0, tilt0_y=0.0, k_wall=3000.0, c_wall=40.0,
               sway_a=0.0, sway_f=2.0, pitch_a=0.0, pitch_f=4.0, vy_bias=0.0, wz_bias=0.0)
# 2026-11-16: the stand-in's mismatch is DYNAMIC: the walking controller responds more abruptly than the documented
# response (faster, higher acceleration limit, overshooting gain), with 80 ms command latency and trot sway / pitch
# rocking; the box is levelled to within 0.3 deg (a constant tilt would be an uncorrectable bias: no policy can hold
# the ball off the downhill wall at rest; it made up most of the first version's gap).
PERTURBED = dict(NOMINAL, tau_c=0.08, tau_v=0.12, tau_w=0.10, a_max=3.3, al_max=10.0, gain_v=1.1, gain_w=1.1,
                 c_rr=0.45, tilt0_x=np.radians(0.3), tilt0_y=np.radians(-0.2), sway_a=0.8, pitch_a=np.radians(1.5))


def action_to_cmd(a):
    """Normalized action in [-1, 1]^3 -> velocity set points."""
    a = jnp.clip(a, -1.0, 1.0)
    return 0.5 * (CMD_HI + CMD_LO) + 0.5 * (CMD_HI - CMD_LO) * a


def _softplus(z):
    return jnp.logaddexp(0.0, z)


def wall_force(b, bv, p):
    """Smooth padded walls (per unit ball mass): stiff spring on the penetration, damping only while in contact."""
    out = []
    for i in range(2):
        pen_hi = _softplus(200 * (b[i] - TRAVEL[i])) / 200
        pen_lo = _softplus(200 * (-TRAVEL[i] - b[i])) / 200
        on = jnp.tanh(200 * (pen_hi + pen_lo))
        out.append(-p['k_wall'] * pen_hi + p['k_wall'] * pen_lo - p['c_wall'] * on * bv[i])
    return jnp.stack(out)


def dynamics(x, a, t, p):
    """dx/dt for the normalized action a (held over the control period) at time t."""
    vx, vy, wz = x[3], x[4], x[5]
    b, bv, cf = x[6:8], x[8:10], x[10:13]
    c = action_to_cmd(a)
    dcf = (c - cf) / p['tau_c']
    ax = p['a_max'] * jnp.tanh((p['gain_v'] * cf[0] - vx) / (p['tau_v'] * p['a_max']))
    ay = p['a_max'] * jnp.tanh((p['gain_v'] * cf[1] + p['vy_bias'] - vy) / (p['tau_v'] * p['a_max']))
    aw = p['al_max'] * jnp.tanh((p['gain_w'] * cf[2] + p['wz_bias'] - wz) / (p['tau_w'] * p['al_max']))
    c_, s_ = jnp.cos(x[2]), jnp.sin(x[2])
    # inertial acceleration of the box, body frame
    abx = ax - wz * vy
    aby = ay + wz * vx
    # floor slope (downhill acceleration factors): mounting offset + trunk lean with acceleration + gait rocking
    sx = p['tilt0_x'] + p['tilt_k'] * abx + p['pitch_a'] * jnp.sin(2 * np.pi * p['pitch_f'] * t)
    sy = p['tilt0_y'] + p['tilt_k'] * aby
    sway = p['sway_a'] * jnp.sin(2 * np.pi * p['sway_f'] * t)
    fx = -abx + G * sx + wz ** 2 * b[0] + 2 * wz * bv[1] + aw * b[1]
    fy = -aby + G * sy - sway + wz ** 2 * b[1] - 2 * wz * bv[0] - aw * b[0]
    bacc = p['k_roll'] * jnp.stack([fx, fy]) - p['c_rr'] * bv + wall_force(b, bv, p)
    return jnp.concatenate([jnp.stack([vx * c_ - vy * s_, vx * s_ + vy * c_, wz, ax, ay, aw]), bv, bacc, dcf,
                            jnp.zeros(3)])


DT = 0.05
SUB = 10


def step(x, a, t, p):
    """One 50 ms control period (RK4, 5 ms) from time t; a_prev is set to a."""
    h = DT / SUB

    def body(i, x):
        ti = t + i * h
        k1 = dynamics(x, a, ti, p)
        k2 = dynamics(x + h / 2 * k1, a, ti + h / 2, p)
        k3 = dynamics(x + h / 2 * k2, a, ti + h / 2, p)
        k4 = dynamics(x + h * k3, a, ti + h, p)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    import jax
    x = jax.lax.fori_loop(0, SUB, body, x)
    return x.at[13:16].set(jnp.clip(a, -1.0, 1.0))


# 2026-11-16 (path task): the stand-in for the real Go1's walking-mode quirks that matter on a repeated path -- ASSUMED values,
# typical of Unitree's high-level mode, not identified (no hardware sysID): speed / yaw under-tracking, sideways and yaw drift
# at the commanded gait, 80 ms latency, trot sway. All repeatable lap to lap.
PATH_REAL = dict(NOMINAL, tau_c=0.08, gain_v=0.85, gain_w=0.85, vy_bias=0.06, wz_bias=0.10, c_rr=0.45,
                 tilt0_x=np.radians(0.3), tilt0_y=np.radians(-0.2), sway_a=0.8, pitch_a=np.radians(1.5))


# 2026-11-16 (user: "make the real robot a hollow ball, and training a filled ball"): the BALL is the mismatch, drone soccer's
# structure (its real ball was 0.4x the trained mass). ASSUMED values (no hardware identification):
#   FILLED (the nominal / training model): a sand- or gel-filled size-5 ball, ~3 kg: solid-sphere inertia -> rolling gain
#     k = 1 / (1 + 2/5) = 0.714; heavy internal damping (c_rr 1.5 1/s); the 3 kg payload makes the walking response slower
#     (tau_v 0.24 s, a_max 2.2 m/s^2) and the trunk lean more under acceleration (tilt_k 0.02 rad per m/s^2)
#   HOLLOW (the real robot): the ordinary inflated ball, 0.43 kg: k = 0.6, free rolling (c_rr 0.3), the documented response
#     (tau_v 0.20, a_max 2.5, tilt_k 0.01)
FILLED = dict(NOMINAL, k_roll=1 / (1 + 2 / 5), c_rr=1.5, tau_v=0.24, a_max=2.2, tilt_k=0.02)
HOLLOW = dict(NOMINAL, k_roll=0.6, c_rr=0.3, tau_v=0.20, a_max=2.5, tilt_k=0.01)
