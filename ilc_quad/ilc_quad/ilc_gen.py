"""
Iterative Learning Control (ILC) for quadruped target jumping, in CasADi + IPOPT.

The reference comes from a full-body trajectory optimization (PlanarQuadModel,
QuadILCStageSolver.init_trajopt); the ILC learns on the SRB model, as in the paper.
Run as a module so the model layer import resolves:

    python -m ilc_quad.ilc_gen
"""

import json

import mujoco
import numpy as np
import casadi as ca

from .sim_quad_model import CANONICAL_JOINT_NAMES, CANONICAL_LEGS, QuadModel


# --------------------------------------------------------------------------
# 1) Planar SRB dynamics (eq. 24) and RK4 discretization
# --------------------------------------------------------------------------
class SRBModel:
    """
    State x = [px, pz, theta, vx, vz, omega]   (R^6, matches Qe_t = diag(wx,wz,wtheta,...))
    Input u = [f1x, f1z, f2x, f2z]             (front foot, rear foot, world frame)
    r1, r2: foot positions relative to CoM (time-varying, from contact schedule/kinematics)
    """

    def __init__(self, mass, inertia, g=9.81):
        self.m = mass
        self.I = inertia
        self.g = g
        self.nx = 6
        self.nu = 4
        self._build()

    def _build(self):
        x = ca.MX.sym("x", self.nx)
        u = ca.MX.sym("u", self.nu)
        r1 = ca.MX.sym("r1", 2)   # [rx, rz] rel. to CoM, front foot
        r2 = ca.MX.sym("r2", 2)   # rear foot
        dt = ca.MX.sym("dt")

        theta, vx, vz, omega = x[2], x[3], x[4], x[5]
        f1 = u[0:2]
        f2 = u[2:4]

        pdd = (f1 + f2) / self.m - ca.vertcat(0, self.g)
        # planar cross product r x f = rx*fz - rz*fx
        torque = (r1[0] * f1[1] - r1[1] * f1[0]) + (r2[0] * f2[1] - r2[1] * f2[0])
        thetadd = torque / self.I

        xdot = ca.vertcat(vx, vz, omega, pdd[0], pdd[1], thetadd)
        f_cont = ca.Function("f_cont", [x, u, r1, r2], [xdot])

        # RK4 step
        def rk4(x0, u0, r1_0, r2_0, h):
            k1 = f_cont(x0, u0, r1_0, r2_0)
            k2 = f_cont(x0 + h / 2 * k1, u0, r1_0, r2_0)
            k3 = f_cont(x0 + h / 2 * k2, u0, r1_0, r2_0)
            k4 = f_cont(x0 + h * k3, u0, r1_0, r2_0)
            return x0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        x_next = rk4(x, u, r1, r2, dt)
        self.f_disc = ca.Function("f_disc", [x, u, r1, r2, dt], [x_next])

        # Jacobians for linearization along a trial trajectory
        A = ca.jacobian(x_next, x)
        B = ca.jacobian(x_next, u)
        self.f_lin = ca.Function("f_lin", [x, u, r1, r2, dt], [A, B])

    def simulate(self, x0, U, R1, R2, dt):
        """Roll out one trial open-loop; returns X (N+1 x nx)."""
        X = [np.asarray(x0).reshape(-1)]
        for t in range(len(U)):
            xn = self.f_disc(X[-1], U[t], R1[t], R2[t], dt).full().flatten()
            X.append(xn)
        return np.array(X)

    def linearize_along_trial(self, X, U, R1, R2, dt):
        """Return lists A_t, B_t for t = 0..Nc-1 (only where control acts)."""
        A_list, B_list = [], []
        for t in range(len(U)):
            A, B = self.f_lin(X[t], U[t], R1[t], R2[t], dt)
            A_list.append(A.full())
            B_list.append(B.full())
        return A_list, B_list


# --------------------------------------------------------------------------
# 2) Planar full-body dynamics (paper Sec. II-C), parameters from the MJCF
# --------------------------------------------------------------------------
def _rot2(a, v):
    """Rotate the x-z vector v by the CCW angle a."""
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(c * v[0] - s * v[1], s * v[0] + c * v[1])


class PlanarQuadModel:
    """
    Sagittal-plane full-body model: trunk + front and rear leg pairs (thigh, calf), the left
    and right legs of a pair lumped together, hip abduction locked at 0.

    s = [px, pz, theta, qF_thigh, qF_calf, qR_thigh, qR_calf]           (R^7)
      p     : base body origin in the world x-z plane (MuJoCo qpos[0], qpos[2])
      theta : pitch, nose up positive (SRBModel's convention; = -MuJoCo rotation about +y)
      q     : joint angles with MuJoCo's sign, so they map 1:1 onto the thigh/calf joints
    Link angles (CCW): trunk theta, thigh theta - q_thigh, calf theta - q_thigh - q_calf.
    f = [fFx, fFz, fRx, fRz] (world frame, ground on robot) and tau are pair totals: each
    motor of a pair carries half.

    Dynamics (paper's lambda):  H(s) sdd + h(s, sd) = S tau + S tau_fric(qd) + Jc(s)^T f
    with h = C sd + g, Lagrangian-derived in CasADi. Parameters are read from the compiled
    QuadModel, so they are the MJCF's rather than a copy of them.
    """

    PAIRS = (("FL", "FR"), ("RL", "RR"))

    def __init__(self, qm, g=9.81, fric_eps=0.05):
        self.qm = qm
        self.g = g
        self.ns, self.nj, self.nf = 7, 4, 4
        m = qm.model

        def body(name):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"{qm.robot} has no body {name!r}")
            if not np.allclose(m.body_quat[bid], [1, 0, 0, 0]):
                raise ValueError(f"body {name!r} is rotated in its parent; the planar model assumes not")
            return bid

        def xz(v):
            return np.array([v[0], v[2]], float)

        def iyy(bid):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.body_iquat[bid])
            R = R.reshape(3, 3)
            return float((R @ np.diag(m.body_inertia[bid]) @ R.T)[1, 1])

        # trunk = base + the four hip (abduction) bodies, rigid while abduction is locked at 0
        base = qm.base_body_id
        parts = [(m.body_mass[base], xz(m.body_ipos[base]), iyy(base))]
        for leg in CANONICAL_LEGS:
            b = body(f"{leg}_hip")
            parts.append((m.body_mass[b], xz(m.body_pos[b] + m.body_ipos[b]), iyy(b)))
        self.m_trunk = float(sum(pm for pm, _, _ in parts))
        self.c_trunk = sum(pm * pc for pm, pc, _ in parts) / self.m_trunk
        self.I_trunk = float(sum(pI + pm * np.sum((pc - self.c_trunk) ** 2) for pm, pc, pI in parts))

        # leg pairs: geometry from the left leg (asserted identical in x-z to the right one),
        # mass, inertia and joint friction/armature doubled
        self.legs = []
        for left, right in self.PAIRS:
            geo = []
            for leg in (left, right):
                hip, thigh, calf = body(f"{leg}_hip"), body(f"{leg}_thigh"), body(f"{leg}_calf")
                foot = qm.foot_geom_ids[CANONICAL_LEGS.index(leg)]
                assert m.geom_bodyid[foot] == calf, f"foot geom {leg} is not on {leg}_calf"
                geo.append(dict(
                    hip=xz(m.body_pos[hip] + m.body_pos[thigh]),
                    c_th=xz(m.body_ipos[thigh]), m_th=m.body_mass[thigh], I_th=iyy(thigh),
                    knee=xz(m.body_pos[calf]),
                    c_ca=xz(m.body_ipos[calf]), m_ca=m.body_mass[calf], I_ca=iyy(calf),
                    foot=xz(m.geom_pos[foot]), radius=float(m.geom_size[foot][0])))
            for key in geo[0]:
                assert np.allclose(geo[0][key], geo[1][key]), f"{left}/{right} differ in {key}"
            leg = geo[0]
            for key in ("m_th", "I_th", "m_ca", "I_ca"):
                leg[key] = 2.0 * float(leg[key])
            self.legs.append(leg)
        self.foot_radius = self.legs[0]["radius"]

        # joints in planar order [F thigh, F calf, R thigh, R calf], canonical (left) indices
        idx = [CANONICAL_JOINT_NAMES.index(f"{leg}_{j}_joint")
               for leg in (self.PAIRS[0][0], self.PAIRS[1][0]) for j in ("thigh", "calf")]
        dofs = qm.qvel_adr[idx]
        for i in idx:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, CANONICAL_JOINT_NAMES[i])
            assert np.allclose(m.jnt_pos[jid], 0), "joint must sit at its body's origin"
        self.armature = 2.0 * m.dof_armature[dofs]
        self.damping = 2.0 * m.dof_damping[dofs]
        self.frictionloss = 2.0 * m.dof_frictionloss[dofs]
        self.joint_range = qm.joint_range[idx].copy()
        self.torque_limit = qm.torque_limit[idx].copy()        # per motor; a pair gives twice this
        self.q_home = qm.home_qpos[idx].copy()
        self.total_mass = self.m_trunk + sum(l["m_th"] + l["m_ca"] for l in self.legs)
        assert np.isclose(self.total_mass, qm.total_mass), "planar model lost mass"

        self._build(fric_eps)

    def _build(self, fric_eps):
        ns = self.ns
        s = ca.SX.sym("s", ns)
        sd = ca.SX.sym("sd", ns)
        p, th = s[0:2], s[2]

        bodies = [(self.m_trunk, p + _rot2(th, self.c_trunk), th, self.I_trunk)]
        feet, knees = [], []
        for i, leg in enumerate(self.legs):
            q1, q2 = s[3 + 2 * i], s[4 + 2 * i]
            hip = p + _rot2(th, leg["hip"])
            a_th = th - q1
            a_ca = a_th - q2
            knee = hip + _rot2(a_th, leg["knee"])
            bodies.append((leg["m_th"], hip + _rot2(a_th, leg["c_th"]), a_th, leg["I_th"]))
            bodies.append((leg["m_ca"], knee + _rot2(a_ca, leg["c_ca"]), a_ca, leg["I_ca"]))
            feet.append(knee + _rot2(a_ca, leg["foot"]))
            knees.append(knee)

        H = ca.SX(ca.diag(ca.vertcat(0, 0, 0, *self.armature)))
        V, mc = 0, 0
        for mb, c, a, I in bodies:
            Jv, Jw = ca.jacobian(c, s), ca.jacobian(a, s)
            H += mb * ca.mtimes(Jv.T, Jv) + I * ca.mtimes(Jw.T, Jw)
            V += mb * self.g * c[1]
            mc += mb * c
        com = mc / self.total_mass
        T = 0.5 * ca.dot(sd, ca.mtimes(H, sd))
        # d/dt(dT/dsd) - dT/ds + dV/ds with sdd = 0: Coriolis/centrifugal + gravity
        h = ca.mtimes(ca.jacobian(ca.mtimes(H, sd), s), sd) - ca.gradient(T, s) + ca.gradient(V, s)
        feet = ca.vertcat(*feet)                                        # [xF, zF, xR, zR]
        Jc = ca.jacobian(feet, s)
        Iyy = sum(I + mb * ca.sumsqr(c - com) for mb, c, _, I in bodies)

        qd = ca.SX.sym("qd", self.nj)
        # paper's S_f tau_f: viscous damping + Coulomb frictionloss, smoothed with tanh
        tau_fric = -ca.DM(self.damping) * qd - ca.DM(self.frictionloss) * ca.tanh(qd / fric_eps)

        self.H = ca.Function("H", [s], [H])
        self.h = ca.Function("h", [s, sd], [h])
        self.tau_fric = ca.Function("tau_fric", [qd], [tau_fric])
        self.feet = ca.Function("feet", [s], [feet])                    # foot sphere centres
        self.knees = ca.Function("knees", [s], [ca.vertcat(*knees)])
        self.Jc = ca.Function("Jc", [s], [Jc])
        self.com = ca.Function("com", [s], [com])
        self.Jcom = ca.Function("Jcom", [s], [ca.jacobian(com, s)])
        self.pitch_inertia = ca.Function("pitch_inertia", [s], [Iyy])  # about the whole-body CoM
        # quasi-static stance torque per MOTOR from pair forces: the joint rows of the dynamics
        # give tau = -(d feet / d q)^T f = -J(q)^T R(theta)^T f; a motor carries half the pair's
        self._torque_map = ca.Function("torque_map", [s], [-0.5 * Jc[:, 3:].T])
        self.S = np.vstack([np.zeros((3, self.nj)), np.eye(self.nj)])

    def torque_map(self, q, theta):
        """(4 motors, 4 forces) map for QuadILCStageSolver.torque_map_fn; p does not enter."""
        s = np.concatenate([[0.0, 0.0, theta], np.asarray(q, float)])
        return self._torque_map(s).full()

    def standing_state(self):
        """Home joints, level trunk, both foot spheres touching z = 0, whole-body CoM at x = 0."""
        s = np.concatenate([[0.0, 0.0, 0.0], self.q_home])
        feet = self.feet(s).full().ravel()
        assert np.isclose(feet[1], feet[3]), "home pose must put both feet at one height"
        s[1] = self.foot_radius - feet[1]
        s[0] = -float(self.com(s)[0])
        return s


def build_lifted_G(A_list, B_list, N, Nc, nx, nu, flatten=False):
    """
    G[t, j] = d x_{t+1} / d u_j, controls in [0, Nc).
    A_list[t], B_list[t] are the Jacobians of x_{t+1} = f(x_t, u_t), t = 0..N-1.
    flatten=False -> (N, Nc, nx, nu)
    flatten=True  -> (N*nx, Nc*nu), time-major rows and columns (paper's block G_k)
    """
    G = np.zeros((N, Nc, nx, nu))
    for tc in range(Nc):
        G[tc, tc] = B_list[tc]                 # immediate effect on x_{tc+1}
        for t in range(tc + 1, N):             # propagate down the column
            G[t, tc] = A_list[t] @ G[t - 1, tc]
    if flatten:
        G = G.transpose(0, 2, 1, 3).reshape(N * nx, Nc * nu)
    return G



class QuadILCStageSolver:
    def __init__(self, N, Nc, Ndc, nx, nu, mdc_params, torque_map_fn,
                 solver="osqp", slack_weight=1e2):
        """
        mdc_params    : dict with rho, sigma, Vmax (>0), Vmin (<0), tau_max, fmin, fmax, mu
        torque_map_fn : (q_t, theta_t) -> T_t = J(q_t)^T R(theta_t)^T, shape (n_joints, nu)  (eq. 14/27)
                        evaluated at trial k's logged configuration (T_{k+1} ~= T_k, like B_{k+1} ~= B_k)
        solver        : "osqp" (fast convex QP) or "ipopt"
        slack_weight  : penalty on torque/MDC violation; these limits are softened so the
                        QP stays feasible when |qdot| is large enough that the MDC and
                        saturation bands stop overlapping. 1e2 already makes 1 N·m of
                        violation cost ~1e4 x a typical tracking cost; 1e6 made the QP so
                        badly scaled that OSQP failed and IPOPT ran out of iterations
        """
        self.N, self.Nc, self.Ndc = N, Nc, Ndc
        self.nx, self.nu = nx, nu
        self.p = mdc_params
        self.torque_map_fn = torque_map_fn
        self.solver = solver
        self.slack_weight = slack_weight

    def stage_rows(self, stage):
        """
        0-indexed time steps of G / e_k included in the cost. Row m is sample x_{m+1}, so
        the paper's 1-indexed windows S{1}=[1,Nc], S{2}=[Ndc,N], S{3}={N} shift down by one.
        """
        if stage == 1:
            return list(range(0, self.Nc))              # contact priority
        elif stage == 2:
            return list(range(self.Ndc - 1, self.N))     # rear contact + flight
        elif stage == 3:
            return [self.N - 1]                           # goal priority: landing sample
        raise ValueError("stage must be 1, 2 or 3")

    # ------------------------------------------------------------------------------------
    # Initial trajectory optimization: full-body TO (paper Sec. II-C, eqs. 5-7)
    # ------------------------------------------------------------------------------------
    @staticmethod
    def box_profile(xf, ground_z, box, width=0.005):
        """Smooth terrain height under a foot at world x = xf: ground before the box, box top after."""
        if box is None:
            return ground_z
        return ground_z + 0.5 * box["height"] * (1 + ca.tanh((xf - box["x_front"]) / width))

    def init_trajopt(self, fb, goal, swing_mask, dt, theta_goal=0.0, box=None, clearance=0.02,
                     n_touchdown=3, qd_max=30.0, margin=0.8, joint_margin=0.1,
                     swing_qd=10.0, w_swing=1.0, w_goal=1e5, w_tau=1e-4, w_smooth=1e-4):
        """
        Reference for trial 1 and x_ref for every trial, from the full-body dynamics.

        fb         : PlanarQuadModel, the NOMINAL robot (the ILC corrects the mismatch)
        goal       : (c_x, c_z) whole-body CoM at sample N. The robot starts at
                     fb.standing_state() (ground at z = 0, CoM at x = 0), so standing on a box
                     of height h is c_z = c_z(0) + h
        swing_mask : (Nc, nu) contact schedule (Ephase); force u_t acts over [t, t+1]
        box        : None, or dict(x_front=..., height=...) in the same world frame
        clearance  : minimum foot height above the terrain during flight (m)
        n_touchdown: last samples before N where only "not inside the terrain" is required,
                     so the feet can descend onto the landing surface (a 2 cm margin one
                     10 ms step before touchdown would otherwise hold the CoM too high)
        qd_max     : joint speed limit (rad/s); the MJCF declares none
        margin     : fraction of each hard limit the plan may use: mu, fmax, torque and speed
                     are scaled by it and fmin divided by it, so the reference sits inside
                     every bound and the ILC has room to correct in any direction (a plan
                     on the friction cone leaves no forward force to add, for one)
        joint_margin: rad kept clear of each end of every joint range
        swing_qd   : soft speed limit (rad/s) for a leg whose foot is off the ground; each
                     rad/s above it costs w_swing * dt * excess^2. Unpenalized, the TO uses
                     the free legs as a reaction wheel to steer pitch, chattering them
                     between ~12 and 24 rad/s every sample -- motion no joint PD can track

        The paper's eqs. (5)-(7) on s = [p; theta; q]:
          lambda : H(s_k)(sd_{k+1} - sd_k)/dt + h(s_k, sd_k) = S(tau_k + tau_fric) + Jc^T f_k,
                   s_{k+1} = s_k + dt sd_{k+1}                          (semi-implicit Euler)
          alpha  : standing start at rest, stance feet fixed where they stand, pre-landing
                   joints q_N = q_home
          beta   : joint angle / speed / torque limits, the ILC's Icontact set (fz in
                   [fmin, fmax], linearized friction cone), feet and knees above the
                   ground/box, every foot past the box edge at N
          cost   : J = CoM/pitch miss at N in Stage III's 1:3:3 shape, + dt sum w_tau |tau|^2
                   (energy) + w_smooth sum |f_{k+1} - f_k|^2 (smooth forces for the ILC)
                   + w_swing dt sum (swing-leg joint speed above swing_qd)^2
        Forces and torques are pair totals, so the torque bound is 2x the per-motor limit
        (times margin). Motor (MDC) limits are left to the ILC, as in the paper.

        returns x_ref (N+1, 6) = [c_x, c_z, theta, cd_x, cd_z, theta_d] on the whole-body CoM,
                u_ref (Nc, nu) = the optimal contact forces (trial 1, eq. 29), and info with
                the joint trajectory (q_ref, qd_ref), torques, the SRB lever arms
                R1/R2 (N, 2) = contact point - CoM, and feet_rel (N+1, 2, 2) for min_clearance
        """
        nu, Nc, N = self.nu, self.Nc, self.N
        ns, nj = fb.ns, fb.nj
        p = self.p
        assert nu == fb.nf, "ILC forces must be the model's [fFx, fFz, fRx, fRz]"
        ground_z = 0.0
        radius = fb.foot_radius
        s0 = fb.standing_state()
        c0 = fb.com(s0).full().ravel()
        feet0 = fb.feet(s0).full().ravel()

        # the plan's limits: every hard bound tightened by the margin
        assert 0 < margin <= 1, "margin is the usable fraction of each limit"
        mu, fmin, fmax = margin * p["mu"], p["fmin"] / margin, margin * p["fmax"]
        tau_max = margin * 2.0 * fb.torque_limit
        qd_lim = margin * qd_max
        q_lo, q_hi = fb.joint_range[:, 0] + joint_margin, fb.joint_range[:, 1] - joint_margin
        assert np.all(q_lo < fb.q_home) and np.all(fb.q_home < q_hi), "joint_margin excludes home"

        opti = ca.Opti()
        S = opti.variable(N + 1, ns)
        Sd = opti.variable(N + 1, ns)
        Tau = opti.variable(N, nj)
        F = opti.variable(Nc, nu)
        opti.subject_to(S[0, :].T == s0)
        opti.subject_to(Sd[0, :].T == 0)

        # lambda: full-body dynamics, no contact force in flight
        for k in range(N):
            sk, sdk, sdn = S[k, :].T, Sd[k, :].T, Sd[k + 1, :].T
            f = F[k, :].T if k < Nc else ca.DM.zeros(nu)
            lhs = ca.mtimes(fb.H(sk), sdn - sdk) / dt + fb.h(sk, sdk)
            rhs = ca.mtimes(fb.S, Tau[k, :].T + fb.tau_fric(sdk[3:])) + ca.mtimes(fb.Jc(sk).T, f)
            opti.subject_to(lhs == rhs)
            opti.subject_to(S[k + 1, :].T == sk + dt * sdn)

        # contact phases: the ILC's Ephase / Icontact sets, tightened; a foot pushing over
        # [t, t+1] stays where it stood at both samples
        pinned = [set(), set()]
        for t in range(Nc):
            for foot in range(nu // 2):
                fx, fz = F[t, 2 * foot], F[t, 2 * foot + 1]
                if swing_mask[t, 2 * foot] or swing_mask[t, 2 * foot + 1]:
                    opti.subject_to(fx == 0)
                    opti.subject_to(fz == 0)
                else:
                    opti.subject_to(opti.bounded(fmin, fz, fmax))
                    opti.subject_to(fx - mu * fz <= 0)
                    opti.subject_to(-fx - mu * fz <= 0)
                    pinned[foot] |= {t, t + 1}

        # feet off the ground stay above the terrain (with a margin in flight); knees too
        for k in range(1, N + 1):
            feet = fb.feet(S[k, :].T)
            for foot in range(2):
                xz = feet[2 * foot:2 * foot + 2]
                if k in pinned[foot]:
                    opti.subject_to(xz == feet0[2 * foot:2 * foot + 2])
                    continue
                height = xz[1] - radius - self.box_profile(xz[0], ground_z, box)
                if k <= Nc:                           # lifted off while the other foot pushes
                    opti.subject_to(height >= 0)
                elif k < N - n_touchdown:             # flight
                    opti.subject_to(height >= clearance)
                else:                                 # descending onto the landing surface
                    opti.subject_to(height >= 0)
            knees = fb.knees(S[k, :].T)
            for i in range(2):
                opti.subject_to(knees[2 * i + 1] >= self.box_profile(knees[2 * i], ground_z, box))
        if box is not None:
            # land fully on top: every foot past the front edge at sample N
            feet = fb.feet(S[N, :].T)
            for foot in range(2):
                opti.subject_to(feet[2 * foot] - radius >= box["x_front"] + clearance)

        # joint angle / speed / torque limits; legs back in the home pose for landing
        for k in range(N + 1):
            opti.subject_to(opti.bounded(q_lo, S[k, 3:].T, q_hi))
            opti.subject_to(opti.bounded(-qd_lim, Sd[k, 3:].T, qd_lim))
        for k in range(N):
            opti.subject_to(opti.bounded(-tau_max, Tau[k, :].T, tau_max))
        opti.subject_to(S[N, 3:].T == fb.q_home)

        # J(q_N) + running cost
        cN = fb.com(S[N, :].T)
        cost = w_goal * ((cN[0] - goal[0]) ** 2 + 3 * (cN[1] - goal[1]) ** 2
                         + 3 * (S[N, 2] - theta_goal) ** 2)
        cost += dt * w_tau * ca.sumsqr(Tau) + w_smooth * ca.sumsqr(F[1:, :] - F[:-1, :])

        # swing legs: joint speed above swing_qd is penalized, through a slack per joint and
        # sample (excess >= |qd| - swing_qd, excess >= 0) so the cost stays smooth
        swing_rows = [(k, foot) for k in range(1, N + 1) for foot in range(2)
                      if k not in pinned[foot]]
        if swing_rows and w_swing > 0:
            excess = opti.variable(len(swing_rows), 2)
            opti.subject_to(ca.vec(excess) >= 0)
            for r, (k, foot) in enumerate(swing_rows):
                qd_leg = Sd[k, 3 + 2 * foot:5 + 2 * foot].T
                opti.subject_to(excess[r, :].T >= qd_leg - swing_qd)
                opti.subject_to(excess[r, :].T >= -qd_leg - swing_qd)
            cost += w_swing * dt * ca.sumsqr(excess)
        opti.minimize(cost)

        # initial guess: base on a straight line to the goal, home joints, weight shared
        # between the stance feet, the matching static torques
        tt = np.linspace(0, 1, N + 1)
        s_guess = np.tile(s0, (N + 1, 1))
        s_guess[:, 0] += tt * (goal[0] - c0[0])
        s_guess[:, 1] += tt * (goal[1] - c0[1])
        opti.set_initial(S, s_guess)
        opti.set_initial(Sd, np.vstack([np.zeros(ns), np.diff(s_guess, axis=0) / dt]))
        stance = ~swing_mask[:, 1::2]                                       # (Nc, 2)
        f_guess = np.zeros((Nc, nu))
        f_guess[:, 1::2] = stance * fb.total_mass * fb.g / np.maximum(stance.sum(1, keepdims=True), 1)
        opti.set_initial(F, f_guess)
        tau_guess = np.zeros((N, nj))
        tau_guess[:Nc] = 2.0 * f_guess @ fb.torque_map(fb.q_home, 0.0).T
        opti.set_initial(Tau, tau_guess)

        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "sb": "yes", "max_iter": 3000})
        try:
            sol = opti.solve()
            ok = True
        except RuntimeError:
            sol, ok = opti.debug, False
        s_opt = np.array(sol.value(S)).reshape(N + 1, ns)
        sd_opt = np.array(sol.value(Sd)).reshape(N + 1, ns)
        tau_opt = np.array(sol.value(Tau)).reshape(N, nj)
        u_ref = np.array(sol.value(F)).reshape(Nc, nu)

        # SRB view of the plan: whole-body CoM + trunk pitch, lever arms from the CoM
        com = np.array([fb.com(s).full().ravel() for s in s_opt])
        comd = np.array([(fb.Jcom(s) @ sd).full().ravel() for s, sd in zip(s_opt, sd_opt)])
        x_ref = np.column_stack([com, s_opt[:, 2], comd, sd_opt[:, 2]])
        feet_rel = np.array([fb.feet(s).full().reshape(2, 2) for s in s_opt]) - com[:, None, :]
        contact = feet_rel - np.array([0.0, radius])                     # sphere bottom
        info = dict(success=ok,
                    landing=x_ref[-1, :3].copy(),
                    goal_miss=float(np.hypot(x_ref[-1, 0] - goal[0], x_ref[-1, 1] - goal[1])),
                    min_clearance=self.min_clearance(x_ref, feet_rel, x_ref[:, 2], radius,
                                                     ground_z, box),
                    s=s_opt, sd=sd_opt, tau=tau_opt, q_ref=s_opt[:, 3:], qd_ref=sd_opt[:, 3:],
                    R1=contact[:N, 0], R2=contact[:N, 1], feet_rel=feet_rel,
                    foot_radius=radius, ground_z=ground_z)
        return x_ref, u_ref, info

    def min_clearance(self, X, feet_rel, theta_ref, foot_radius, ground_z=0.0, box=None):
        """
        Smallest foot height above the terrain during flight (samples Nc+1 .. N-1) for a
        recorded or planned trajectory X (N+1, nx). Negative = a foot went through the
        ground or the box: use it to flag a failed trial on the plant / robot.

        Foot centres are placed at c + Rot(theta - theta_ref) feet_rel: the TO's leg motion
        carried on the trajectory's body. Exact for the TO's own x_ref; for the SRB plant,
        which has no legs, it assumes the joints track the reference.
        """
        worst = np.inf
        for t in range(self.Nc + 1, self.N):
            a = X[t, 2] - theta_ref[t]
            c, s = np.cos(a), np.sin(a)
            for rx, rz in feet_rel[t]:
                xf = X[t, 0] + c * rx - s * rz
                zf = X[t, 1] + s * rx + c * rz - foot_radius
                terrain = ground_z
                if box is not None and xf >= box["x_front"]:
                    terrain = ground_z + box["height"]
                worst = min(worst, zf - terrain)
        return worst

    def solve(self, G, x_k, x_ref, u_k, q_k, theta_k, qdot_k, tau_pd_k, swing_mask,
              Qe_diag, Qu_diag, stage, solver=None):
        """
        G          : (N*nx, Nc*nu) from build_lifted_G(..., flatten=True); time-major:
                     G[t*nx + a, j*nu + b] = d x_{t+1}[a] / d u_j[b]
        x_k        : (N+1, nx) state trajectory recorded this trial, x_k[0] = initial state
        x_ref      : (N+1, nx) reference trajectory on the same grid
        u_k        : (Nc, nu)  commanded contact forces this trial (not measured GRF)
        q_k        : (Nc, n_joints) joint angles this trial      -> torque map
        theta_k    : (Nc,)      body pitch this trial             -> torque map
        qdot_k     : (Nc, n_joints) joint velocities this trial  -> MDC (eq. 18), per joint
        tau_pd_k   : (Nc, n_joints) joint PD torque this trial   -> limits apply to
                     tau_total = tau_PD + tau_ILC (eq. 28a), not tau_ILC alone
        swing_mask : (Nc, nu) bool, True where the force component must be 0 (Ephase)
        Qe_diag    : (nx,) or (N, nx)   error weights, constant or per row of e_k
        Qu_diag    : (nu,) or (Nc, nu)  control-offset weights, constant or per time step
        stage      : 1, 2 or 3
        solver     : override self.solver for this call
        returns    : (delta_u_star (Nc, nu), info dict)

        OSQP stalls when a bad trial pushes the torque rows far outside their bounds (the
        slack cost then dwarfs the tracking cost by ~12 orders); such a QP is retried with
        IPOPT, and info["fallback"] says so.
        """
        solver = solver or self.solver
        nx, nu, Nc, N = self.nx, self.nu, self.Nc, self.N
        p = self.p
        assert x_k.shape == (N + 1, nx), f"x_k must be (N+1, nx), got {x_k.shape}"
        assert x_ref.shape == (N + 1, nx), f"x_ref must be (N+1, nx), got {x_ref.shape}"
        assert u_k.shape == (Nc, nu), f"u_k must be (Nc, nu), got {u_k.shape}"
        assert G.shape == (N * nx, Nc * nu), f"G must be flattened (N*nx, Nc*nu), got {G.shape}"

        # trial-k error; x_0 is identical every trial, so row m of e_k is sample x_{m+1} (same as G)
        e_k = x_ref[1:] - x_k[1:]                                            # (N, nx)

        Qe = np.broadcast_to(np.asarray(Qe_diag, float), (N, nx))
        Qu = np.broadcast_to(np.asarray(Qu_diag, float), (Nc, nu))

        opti = ca.Opti("conic") if solver == "osqp" else ca.Opti()
        du = opti.variable(Nc, nu)
        du_flat = ca.reshape(du.T, Nc * nu, 1)           # [du_0; du_1; ...]: time-major, like G's columns

        # ---- objective (eq. 16): predict the whole next-trial error with the full G, and
        # select the stage through the weights: Qe_t as given for t in S{i}, Qe_t = 0 otherwise.
        e_flat = e_k.reshape(N * nx)                      # time-major, matches G's rows
        e_next = e_flat - ca.mtimes(G, du_flat)           # eq. (19), all N samples
        Qe_stage = np.zeros((N, nx))
        rows = self.stage_rows(stage)
        Qe_stage[rows] = Qe[rows]
        cost = ca.dot(e_next, ca.mtimes(ca.diag(Qe_stage.reshape(N * nx)), e_next))
        cost += ca.dot(du_flat, ca.mtimes(ca.diag(Qu.reshape(-1)), du_flat))

        # ---- constraints
        slacks = []
        for t in range(Nc):
            u_next = ca.reshape(u_k[t, :], nu, 1) + du[t, :].T

            # Ephase: swing components pinned to zero
            for j in np.where(swing_mask[t])[0]:
                opti.subject_to(u_next[int(j)] == 0)

            # Icontact: force limits + linearized friction cone, stance feet only (planar: (fx, fz) per foot)
            for foot in range(nu // 2):
                if swing_mask[t, 2 * foot] or swing_mask[t, 2 * foot + 1]:
                    continue
                fx, fz = u_next[2 * foot], u_next[2 * foot + 1]
                opti.subject_to(opti.bounded(p["fmin"], fz, p["fmax"]))
                opti.subject_to(fx - p["mu"] * fz <= 0)
                opti.subject_to(-fx - p["mu"] * fz <= 0)

            # total joint torque the motors will see (eq. 27/28a), with trial-k torque map and PD torque
            T_t = np.asarray(self.torque_map_fn(q_k[t], theta_k[t]), float)
            nj = T_t.shape[0]
            tau_total = ca.mtimes(T_t, u_next) + np.asarray(tau_pd_k[t], float).reshape(nj, 1)

            # combined Isat and Imdc bounds (eq. 18), per joint, softened with a slack --
            # only on joints the forces reach: a swing leg's torque (TO feedforward + PD)
            # is outside the ILC's authority, and its row would just add a constant slack
            reach = np.any(np.abs(T_t[:, ~np.asarray(swing_mask[t], bool)]) > 0, axis=1)
            idx = np.where(reach)[0]
            if idx.size == 0:
                continue
            qd = np.asarray(qdot_k[t], float).reshape(nj)[idx]
            lb = np.maximum(np.broadcast_to(-np.asarray(p["tau_max"], float), (nj,))[idx],
                            (p["Vmin"] - p["sigma"] * qd) / p["rho"])
            ub = np.minimum(np.broadcast_to(np.asarray(p["tau_max"], float), (nj,))[idx],
                            (p["Vmax"] - p["sigma"] * qd) / p["rho"])
            s = opti.variable(idx.size)
            opti.subject_to(s >= 0)
            rows = ca.vertcat(*[tau_total[int(i)] for i in idx])
            opti.subject_to(rows <= ub.reshape(-1, 1) + s)
            opti.subject_to(rows >= lb.reshape(-1, 1) - s)
            slacks.append(s)

        S = ca.vertcat(*slacks) if slacks else ca.MX.zeros(0, 1)
        if slacks:
            cost += self.slack_weight * ca.dot(S, S)
        opti.minimize(cost)

        if solver == "osqp":
            opti.solver("osqp", {"print_time": False, "error_on_fail": False},
                        {"verbose": False, "eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 20000,
                         "polish": True})
        else:
            opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes"})

        try:
            sol = opti.solve()
            if solver == "osqp" and not opti.stats()["success"]:
                du_star, info = self.solve(G, x_k, x_ref, u_k, q_k, theta_k, qdot_k, tau_pd_k,
                                           swing_mask, Qe_diag, Qu_diag, stage, solver="ipopt")
                info["fallback"] = f"ipopt (osqp: {opti.stats()['return_status']})"
                return du_star, info
            du_star = np.array(sol.value(du)).reshape(Nc, nu)
            slack_max = float(np.max(np.abs(sol.value(S)))) if S.numel() else 0.0
            info = dict(success=True, slack_max=slack_max,
                        predicted_cost=float(sol.value(cost)))
        except RuntimeError as err:
            # never crash mid-experiment: repeat the last trial's forces
            du_star = np.zeros((Nc, nu))
            info = dict(success=False, slack_max=np.nan, error=str(err).splitlines()[-1])
        return du_star, info



class JumpILC:
    """
    The paper's pipeline for one jump, independent of what flies the trials: the full-body
    TO gives the reference and the trial-1 forces (eq. 29), then after every trial the
    3-stage ILC update (eq. 15) moves the forces U for the next one.

        ilc = JumpILC(QuadModel("go2"), jump=(0.5, 0.1), box=dict(x_front=0.25, height=0.1))
        while True:
            log = fly(ilc.U)                 # X (N+1, 6) on the TO grid, and q, theta,
            result = ilc.update(log)         # qdot, tau_pd (Nc, ...) -- see update()
            if result["converged"] or ilc.trial >= max_trials:
                break

    jump      : CoM displacement (dx, dz) from standing to landing, landing pitch 0. The
                robot stands with its CoM at x = 0 on ground z = 0, so for a box dz = height
    box       : None or dict(x_front, height), same frame; x_front must be ahead of the
                standing front feet (x ~ +0.2 m) and behind the landing rear feet
    phases    : (Ndc, Nsc, Nfl) samples of all-leg contact, rear-leg contact and flight
    schedule  : n_stage1 trials of Stage I, n_stage2 of Stage II, then Stage III until the
                landing is within pos_tol / theta_tol (paper, Sec. III-C)
    margin    : share of each hard limit the TO may use (init_trajopt)
    swing_qd, w_swing : soft speed limit on swing/flight legs and its weight (init_trajopt)
    Qe_diag, Qu_diag : ILC weights on the state error and the trial-to-trial force step
    Qu_stage3 : step weight for Stage III alone (None: Qu_diag). Stage III weighs only
                the landing state, a handful of rows against every force sample, so a
                step small enough for Stages I-II can overshoot there, and the unweighted
                directions drift trial to trial
    reference : None to run the TO, or a path written by save_reference() to reuse one
    """

    nx, nu = 6, 4                 # x = [px, pz, theta, vx, vz, omega], u = [f1x, f1z, f2x, f2z]

    def __init__(self, qm, jump=(0.50, 0.10), box=dict(x_front=0.25, height=0.10),
                 phases=(20, 20, 25), dt=0.01, n_stage1=5, n_stage2=5, pos_tol=0.01,
                 theta_tol=np.radians(1.0), margin=0.8, swing_qd=10.0, w_swing=1.0,
                 mdc_params=None, Qe_diag=(1.0, 3.0, 3.0, 0.01, 0.01, 0.01), Qu_diag=1e-5,
                 Qu_stage3=None, reference=None):
        nx, nu = self.nx, self.nu
        self.fb = fb = PlanarQuadModel(qm)
        self.Ndc, self.Nsc, self.Nfl = (int(n) for n in phases)
        self.Nc = self.Ndc + self.Nsc
        self.N = self.Nc + self.Nfl
        self.dt = float(dt)
        self.box = None if box is None else {k: float(v) for k, v in box.items()}
        self.n_stage1, self.n_stage2 = n_stage1, n_stage2
        self.pos_tol, self.theta_tol = pos_tol, theta_tol
        # contact limits per leg pair -- the planar model lumps a left/right pair into one
        # leg, so fmax is two legs' worth of the paper's 250 N per leg; per-motor torque
        # limits from the MJCF (thigh, calf, front then rear); rho, sigma, V are still the
        # A1 motor values (Table III)
        self.mdc_params = mdc_params or dict(rho=0.35, sigma=0.02, Vmax=21.5, Vmin=-21.5,
                                             tau_max=fb.torque_limit, fmin=5.0, fmax=500.0,
                                             mu=0.6)
        self.config = dict(robot=qm.robot, jump=[float(v) for v in jump], box=self.box,
                           phases=[self.Ndc, self.Nsc, self.Nfl], dt=self.dt, margin=margin,
                           swing_qd=float(swing_qd), w_swing=float(w_swing),
                           fmax=float(self.mdc_params["fmax"]))

        self.s_home = fb.standing_state()
        self.goal = fb.com(self.s_home).full().ravel() + np.asarray(jump, float)
        self.nominal = SRBModel(mass=fb.total_mass, inertia=float(fb.pitch_inertia(self.s_home)))

        self.swing_mask = np.zeros((self.Nc, nu), dtype=bool)
        self.swing_mask[self.Ndc:, 0:2] = True    # front foot in the air during rear-leg contact
        self.Qe_diag = np.asarray(Qe_diag, float)
        self.Qu_diag = np.broadcast_to(np.asarray(Qu_diag, float), (nu,)).copy()
        self.Qu_stage3 = self.Qu_diag if Qu_stage3 is None else \
            np.broadcast_to(np.asarray(Qu_stage3, float), (nu,)).copy()

        # J(q)^T R(theta)^T of the planar model, per motor (eq. 14/27)
        self.solver = QuadILCStageSolver(self.N, self.Nc, self.Ndc, nx, nu, self.mdc_params,
                                         fb.torque_map)
        if reference is None:
            self.x_ref, self.u_ref, self.to_info = self.solver.init_trajopt(
                fb, self.goal, self.swing_mask, self.dt, box=self.box, margin=margin,
                swing_qd=swing_qd, w_swing=w_swing)
        else:
            self._load_reference(reference)
        # time-varying lever arms (contact point - CoM) from the TO
        self.R1, self.R2 = self.to_info["R1"], self.to_info["R2"]

        self.U = self.u_ref.copy()                # trial 1 flies the TO forces (eq. 29)
        self.trial = 0                            # trials flown so far
        self.history = []

    # -- reference persistence (the TO takes tens of seconds) ------------------------------
    _INFO_KEYS = ("s", "sd", "tau", "q_ref", "qd_ref", "R1", "R2", "feet_rel", "landing",
                  "foot_radius", "ground_z", "success", "goal_miss", "min_clearance")

    def save_reference(self, path):
        np.savez(path, x_ref=self.x_ref, u_ref=self.u_ref, config=json.dumps(self.config),
                 **{f"info_{k}": self.to_info[k] for k in self._INFO_KEYS})

    def _load_reference(self, path):
        data = np.load(path)
        saved = json.loads(str(data["config"]))
        if saved != json.loads(json.dumps(self.config)):
            raise ValueError(f"reference {path} was made for {saved}, not {self.config}")
        self.x_ref, self.u_ref = data["x_ref"], data["u_ref"]
        self.to_info = {k: data[f"info_{k}"] for k in self._INFO_KEYS}
        for k in ("foot_radius", "ground_z", "goal_miss", "min_clearance"):
            self.to_info[k] = float(self.to_info[k])
        self.to_info["success"] = bool(self.to_info["success"])

    # -- per-trial bookkeeping --------------------------------------------------------------
    def stage(self, trial=None):
        """Stage for a 0-indexed trial (default: the next one to fly)."""
        k = self.trial if trial is None else trial
        return 1 if k < self.n_stage1 else 2 if k < self.n_stage1 + self.n_stage2 else 3

    def landing_error(self, x_N):
        """(CoM position miss, |pitch|) at the landing sample."""
        return float(np.hypot(x_N[0] - self.goal[0], x_N[1] - self.goal[1])), abs(float(x_N[2]))

    def clearance(self, X):
        return self.solver.min_clearance(X, self.to_info["feet_rel"], self.x_ref[:, 2],
                                         self.to_info["foot_radius"], self.to_info["ground_z"],
                                         self.box)

    def update(self, log):
        """
        Score the trial that just flew self.U and compute the next U (eq. 15).

        log : X      (N+1, 6) SRB state [c_x, c_z, theta, cd_x, cd_z, theta_d] on the TO grid,
                     whole-body CoM in the TO's frame (standing CoM at x = 0, ground z = 0)
              q      (Nc, 4) planar joint angles [F thigh, F calf, R thigh, R calf]
              theta  (Nc,)   pitch
              qdot   (Nc, 4) planar joint velocities
              tau_pd (Nc, 4) per-motor torque during the trial besides J^T f of the
                     forces U (PD, and any other feedforward): the torque limits
                     apply to  T u + tau_pd
        returns dict(trial, stage, pos_err, theta_err, clearance, converged[, solver])
        """
        X = np.asarray(log["X"], float)
        stage = self.stage()
        pos_err, th_err = self.landing_error(X[-1])
        result = dict(trial=self.trial + 1, stage=stage, pos_err=pos_err, theta_err=th_err,
                      clearance=self.clearance(X),
                      converged=stage == 3 and pos_err < self.pos_tol and th_err < self.theta_tol)
        self.trial += 1
        self.history.append(result)
        if result["converged"]:
            return result

        nx, nu, N, Nc = self.nx, self.nu, self.N, self.Nc
        U_full = np.vstack([self.U, np.zeros((self.Nfl, nu))])
        A_list, B_list = self.nominal.linearize_along_trial(X, U_full, self.R1, self.R2, self.dt)
        G = build_lifted_G(A_list, B_list, N, Nc, nx, nu, flatten=True)
        du, info = self.solver.solve(G, X, self.x_ref, self.U, log["q"], log["theta"],
                                     log["qdot"], log["tau_pd"], self.swing_mask,
                                     self.Qe_diag, self.Qu_stage3 if stage == 3 else self.Qu_diag,
                                     stage)
        result["solver"] = info
        self.U = self.U + du                              # eq. (15): u_{k+1} = u_k + du*
        self.U[self.swing_mask] = 0.0
        return result

    def describe_reference(self):
        info = self.to_info
        return (f"TO: success={info['success']}, planned landing {np.round(info['landing'], 3)} "
                f"(goal {np.round(self.goal, 3)}), min foot clearance "
                f"{info['min_clearance']*100:.1f} cm, peak |f| {np.abs(self.u_ref).max():.0f} N, "
                f"peak |tau| per motor {np.round(np.abs(info['tau']).max(0) / 2, 1)} N·m")


def describe_result(r):
    line = (f"trial {r['trial']:2d} | stage {r['stage']} | landing miss {r['pos_err']*100:5.2f} cm | "
            f"pitch {np.degrees(r['theta_err']):4.1f} deg | min foot clearance "
            f"{r['clearance']*100:5.1f} cm" + ("  <-- feet hit the ground/box" if r["clearance"] < 0 else ""))
    info = r.get("solver")
    if info is not None and not info["success"]:
        line += f"\n   solver failed ({info['error']}); repeating last forces"
    elif info is not None and info["slack_max"] > 1e-3:
        line += f"\n   torque/MDC limits softened by up to {info['slack_max']:.2f} N·m"
    if info is not None and "fallback" in info:
        line += f"\n   QP solved by {info['fallback']}"
    return line


def run_ilc_demo(jump=(0.50, 0.10), box=dict(x_front=0.25, height=0.10), robot="go2",
                 max_trials=25, seed=0, **ilc_kwargs):
    """
    Paper's 3-stage ILC on an offline simulated jump (JumpILC with a stand-in plant).
      nominal model : SRB with the robot's mass and pitch inertia, for A_t, B_t (what the ILC believes)
      plant         : SRB with +15 % mass, +25 % inertia, 2 % actuation noise (stands in for hardware)
    The phases default to 20/20/25: with the TO held inside every limit (margin), Go2 needs
    the longer push to reach a 0.5 m box jump (at 80 %, 10/10/20 fell ~6 cm short).
    """
    ilc = JumpILC(QuadModel(robot), jump=jump, box=box, **ilc_kwargs)
    print(ilc.describe_reference())
    N, Nc, dt, nu = ilc.N, ilc.Nc, ilc.dt, ilc.nu
    plant = SRBModel(mass=ilc.nominal.m * 1.15, inertia=ilc.nominal.I * 1.25)
    act_noise = 0.02
    rng = np.random.default_rng(seed)

    def execute_trial(U_cmd):
        x = ilc.x_ref[0].copy()
        X = [x.copy()]
        for t in range(N):
            u = U_cmd[t] * (1.0 + act_noise * rng.standard_normal(nu)) if t < Nc else np.zeros(nu)
            x = plant.f_disc(x, u, ilc.R1[t], ilc.R2[t], dt).full().flatten()
            X.append(x.copy())
        X = np.array(X)
        return dict(
            X=X,
            # the SRB plant has no legs: assume the joints track the TO's joint trajectory
            q=ilc.to_info["q_ref"][:Nc],
            theta=X[:Nc, 2],
            qdot=ilc.to_info["qd_ref"][:Nc],
            tau_pd=np.zeros((Nc, ilc.fb.nj)),      # placeholder: no PD in this plant
        )

    while ilc.trial < max_trials:
        result = ilc.update(execute_trial(ilc.U))
        print(describe_result(result))
        if result["converged"]:
            print("Target reached.")
            break
    return ilc.history, ilc.U


if __name__ == "__main__":
    run_ilc_demo()
