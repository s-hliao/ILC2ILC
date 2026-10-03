"""ilc_quad's PlanarQuadModel kinematics in JAX (the sagittal model the controller and the SRB state use):
s = [px, pz, theta, qF_thigh, qF_calf, qR_thigh, qR_calf], theta nose-up. Same parameters, read from the
CasADi model; check_planar.py compares the two."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _rot2(a, v):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class Planar:
    def __init__(self, fb):
        self.m_trunk, self.I_trunk = float(fb.m_trunk), float(fb.I_trunk)
        self.c_trunk = jnp.asarray(fb.c_trunk, jnp.float32)
        self.legs = [{k: (jnp.asarray(v, jnp.float32) if np.ndim(v) else float(v)) for k, v in leg.items()}
                     for leg in fb.legs]
        self.total_mass = float(fb.total_mass)
        self.armature = jnp.asarray(fb.armature, jnp.float32)
        self.joint_range = jnp.asarray(fb.joint_range, jnp.float32)
        self.feet_jac = jax.jacfwd(self.feet)
        self.com_jac = jax.jacfwd(self.com)

    def _bodies(self, s):
        p, th = s[0:2], s[2]
        bodies = [(self.m_trunk, p + _rot2(th, self.c_trunk), th, self.I_trunk)]
        feet = []
        for i, leg in enumerate(self.legs):
            q1, q2 = s[3 + 2 * i], s[4 + 2 * i]
            hip = p + _rot2(th, leg["hip"])
            a_th = th - q1
            a_ca = a_th - q2
            knee = hip + _rot2(a_th, leg["knee"])
            bodies.append((leg["m_th"], hip + _rot2(a_th, leg["c_th"]), a_th, leg["I_th"]))
            bodies.append((leg["m_ca"], knee + _rot2(a_ca, leg["c_ca"]), a_ca, leg["I_ca"]))
            feet.append(knee + _rot2(a_ca, leg["foot"]))
        return bodies, jnp.concatenate(feet)

    def feet(self, s):
        """Foot sphere centres [xF, zF, xR, zR]."""
        return self._bodies(s)[1]

    def com(self, s):
        bodies, _ = self._bodies(s)
        return sum(mb * c for mb, c, _, _ in bodies) / self.total_mass

    def mass_matrix(self, s):
        """H(s) (7, 7), the joint armature included."""
        def parts(s_):
            bodies, _ = self._bodies(s_)
            return [(c, a) for _, c, a, _ in bodies]
        J = jax.jacfwd(lambda s_: jnp.stack([jnp.concatenate([c, a[None]]) for c, a in parts(s_)]))(s)
        bodies, _ = self._bodies(s)
        H = jnp.diag(jnp.concatenate([jnp.zeros(3), self.armature]))
        for i, (mb, _, _, I) in enumerate(bodies):
            Jv, Jw = J[i, :2], J[i, 2]
            H = H + mb * Jv.T @ Jv + I * jnp.outer(Jw, Jw)
        return H

    def torque_map(self, q, theta):
        """(4 motors, 4 pair forces): tau = -0.5 J(q)^T R(theta)^T f per motor."""
        s = jnp.concatenate([jnp.zeros(2), theta[None], q])
        return -0.5 * self.feet_jac(s)[:, 3:].T

    def ik(self, s, target, q0, iters=6):
        """Joints putting the feet at target [xF, zF, xR, zR], trunk at s: damped least squares from q0,
        inside the joint range (ilc_jump_sim._ik; a fixed number of steps)."""
        lo, hi = self.joint_range[:, 0], self.joint_range[:, 1]

        def step(q, _):
            s_ik = s.at[3:].set(q)
            e = target - self.feet(s_ik)
            J = self.feet_jac(s_ik)[:, 3:]
            dq = J.T @ jnp.linalg.solve(J @ J.T + 1e-4 * jnp.eye(4), e)
            q_new = jnp.clip(q + dq, lo, hi)
            return jnp.where(jnp.abs(e).max() < 1e-4, q, q_new), None
        return jax.lax.scan(step, q0, None, length=iters)[0]
