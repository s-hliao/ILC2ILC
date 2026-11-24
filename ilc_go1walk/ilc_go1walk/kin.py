"""Go1 leg kinematics (Menagerie go1.xml), canonical leg order FL, FR, RL, RR, joints (hip abduction about x, thigh about y,
calf about y). Foot = the calf's foot sphere centre (0, 0, -0.213) in the calf frame. All in the BASE frame."""
import jax
import jax.numpy as jnp
import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
HIP = np.array([[0.1881, 0.04675, 0.0], [0.1881, -0.04675, 0.0], [-0.1881, 0.04675, 0.0], [-0.1881, -0.04675, 0.0]])
ABD = np.array([0.08, -0.08, 0.08, -0.08])          # thigh offset along y from the hip joint
L_THIGH, L_CALF = 0.213, 0.213
FOOT_R = 0.023


def _rx(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def foot_leg(q3, leg):
    """Foot position (3,) in the base frame for one leg's joint angles q3."""
    calf = jnp.array([0.0, 0.0, -L_CALF])
    knee = jnp.array([0.0, 0.0, -L_THIGH]) + _ry(q3[2]) @ calf
    thigh = jnp.array([0.0, ABD[leg], 0.0]) + _ry(q3[1]) @ knee
    return jnp.asarray(HIP[leg]) + _rx(q3[0]) @ thigh


def feet(q12):
    """(4, 3) feet in the base frame."""
    return jnp.stack([foot_leg(q12[3 * i:3 * i + 3], i) for i in range(4)])


def jacs(q12):
    """(4, 3, 3) per-leg Jacobians d foot / d q3 in the base frame."""
    return jnp.stack([jax.jacfwd(foot_leg)(q12[3 * i:3 * i + 3], i) for i in range(4)])
