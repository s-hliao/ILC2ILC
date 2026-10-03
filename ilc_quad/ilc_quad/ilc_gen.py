"""
Iterative Learning Control (ILC) for quadruped target jumping, in CasADi + IPOPT.

The reference comes from a full-body trajectory optimization (PlanarQuadModel,
QuadILCStageSolver.init_trajopt); the ILC learns on the SRB model, as in the paper.
Run as a module so the model layer import resolves:

    python -m ilc_quad.ilc_gen
"""

import json
import os

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



# Go1 joint output speeds at which the motors run out of voltage (Unitree's spec sheet:
# hip/thigh 30.1 rad/s, knee 20.06 rad/s), in the planar order [F thigh, F calf, R thigh,
# R calf] -- ilc_jump_lockstep.py's GO1_NO_LOAD_SPEED, for mdc="go1"
GO1_NO_LOAD_SPEED_PLANAR = np.array([30.1, 20.06, 30.1, 20.06])


class QuadILCStageSolver:
    def __init__(self, N, Nc, Ndc, nx, nu, mdc_params, torque_map_fn,
                 solver="osqp", slack_weight=1e2, mdc="legacy", mdc_speed_scale=1.0):
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
        mdc           : the motor rows' model. "legacy": the paper's MDC (eq. 18, rho, sigma,
                        V) intersected with tau_max. "go1": the Go1 torque-speed envelope
                        (as ilc_jump_lockstep.py --motor-curve): tau_max up to half the
                        no-load speed, then down linearly to none at it, while motoring
                        (torque along qdot); braking keeps tau_max
        mdc_speed_scale : "go1": the no-load speeds x this (a sagging battery lowers them)
        """
        self.N, self.Nc, self.Ndc = N, Nc, Ndc
        self.nx, self.nu = nx, nu
        self.p = mdc_params
        self.torque_map_fn = torque_map_fn
        self.solver = solver
        self.slack_weight = slack_weight
        if mdc not in ("legacy", "go1"):
            raise ValueError(f"mdc must be legacy or go1, got {mdc!r}")
        self.mdc = mdc
        self.w_max = GO1_NO_LOAD_SPEED_PLANAR * float(mdc_speed_scale)

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

    @staticmethod
    def box_edge(xf, box, clearance, setback, width=0.005):
        """Height a moving foot must clear near the box's front edge: box top + clearance
        within `setback` of the edge (either side), smoothly 0 elsewhere."""
        lo, hi = box["x_front"] - setback, box["x_front"] + setback
        return (box["height"] + clearance) * 0.5 * (ca.tanh((xf - lo) / width)
                                                    - ca.tanh((xf - hi) / width))

    def srb_guess(self, fb, goal, swing_mask, dt, box=None, margin=0.8, clearance=0.02,
                  box_clearance=0.0, box_setback=0.0, theta_goal=0.0, tuck=0.06, joint_margin=0.1,
                  landing_pitch_rate=None, flight_pitch_rate=None, flight_min_pitch=None,
                  landing_clearance=None, n_landing_ramp=6, w_goal=1e5, w_force=1e-6, w_smooth=1e-4, w_omega=1e-3):
        """
        A full-body initial guess for init_trajopt from a single-rigid-body plan.

        1. SRB TO: whole-body CoM c and pitch theta under the contact forces (the SRB model
           the ILC learns on), stance feet fixed where they stand, each stance foot within
           its leg's reach (from the knee's range), the same force limits and friction
           cone, ballistic flight, landing at the goal. The hips of legs in the air stay
           high enough over the terrain (and the box edge) for a tucked foot to clear it.
        2. Full body: the base placed so the home-pose CoM offset sits on c; legs by IK --
           stance feet on their contact points; a foot in the air moved from where it left
           the ground to its home spot under the hip (the landing pose), lifted by `tuck`
           in between and kept over the box edge (and, with a landing clearance, brought
           straight down onto its spot from that height above it).
        Velocities follow the TO's semi-implicit Euler; torques are the full-body dynamics'
        joint rows with the SRB forces, clipped to the plan's limits.
        returns dict(s, sd, tau, f) for init_trajopt(init_guess=...), and the SRB result
        """
        nu, Nc, Ndc, N = self.nu, self.Nc, self.Ndc, self.N
        p = self.p
        m, g = fb.total_mass, fb.g
        s0 = fb.standing_state()
        c0 = fb.com(s0).full().ravel()
        feet0 = fb.feet(s0).full().ravel().reshape(2, 2)
        I = float(fb.pitch_inertia(s0))
        c_off = c0 - s0[:2]                          # CoM - base origin, level home pose
        hips = [leg["hip"] for leg in fb.legs]       # in the base frame
        mu, fmin, fmax = margin * p["mu"], p["fmin"] / margin, margin * p["fmax"]
        radius = fb.foot_radius
        stance = ~swing_mask[:, 1::2]                                       # (Nc, 2)

        # hip-to-foot distance is set by the knee alone: |knee + R(-q_calf) foot|, so the
        # leg's reach runs between the knee's limits (kept joint_margin inside them)
        def leg_length(leg, q_calf):
            a = -q_calf
            v = np.array([np.cos(a) * leg["foot"][0] - np.sin(a) * leg["foot"][1],
                          np.sin(a) * leg["foot"][0] + np.cos(a) * leg["foot"][1]])
            return float(np.linalg.norm(leg["knee"] + v))
        reach_min, reach_max = [], []
        for i, leg in enumerate(fb.legs):
            lo, hi = fb.joint_range[2 * i + 1] + np.array([joint_margin, -joint_margin])
            lengths = [leg_length(leg, qc) for qc in np.linspace(lo, hi, 50)]
            reach_min.append(min(lengths) + 0.01)
            reach_max.append(max(lengths) - 0.01)

        def rot(a, v):
            return ca.vertcat(ca.cos(a) * v[0] - ca.sin(a) * v[1], ca.sin(a) * v[0] + ca.cos(a) * v[1])

        def R(a):
            return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

        def hermite(t, t0, t1, p0, p1, v0, v1):
            """Cubic Hermite point at time t between (t0, p0, v0) and (t1, p1, v1)."""
            h = t1 - t0
            u = (t - t0) / h
            return ((2 * u ** 3 - 3 * u ** 2 + 1) * p0 + (u ** 3 - 2 * u ** 2 + u) * h * v0
                    + (-2 * u ** 3 + 3 * u ** 2) * p1 + (u ** 3 - u ** 2) * h * v1)

        def solve_srb():
            opti = ca.Opti()
            C = opti.variable(N + 1, 2)
            Cd = opti.variable(N + 1, 2)
            Th = opti.variable(N + 1)
            Om = opti.variable(N + 1)
            F = opti.variable(Nc, nu)
            opti.subject_to(C[0, :].T == c0)
            opti.subject_to(Cd[0, :].T == 0)
            opti.subject_to(Th[0] == 0)
            opti.subject_to(Om[0] == 0)
            if landing_pitch_rate is not None:
                opti.subject_to(opti.bounded(-landing_pitch_rate, Om[N], landing_pitch_rate))
            if flight_pitch_rate is not None:
                for k in range(Nc, N + 1):
                    opti.subject_to(opti.bounded(-flight_pitch_rate, Om[k], flight_pitch_rate))
            if flight_min_pitch is not None:
                for k in range(Nc, N + 1):
                    opti.subject_to(Th[k] >= flight_min_pitch)
            for k in range(N):
                f = F[k, :].T if k < Nc else ca.DM.zeros(nu)
                ck = C[k, :].T
                force = f[0:2] + f[2:4] + ca.DM([0, -m * g])
                moment = 0
                for leg in range(2):
                    r = ca.DM(feet0[leg]) - ck
                    moment += r[0] * f[2 * leg + 1] - r[1] * f[2 * leg]
                opti.subject_to(m * (Cd[k + 1, :].T - Cd[k, :].T) / dt == force)
                opti.subject_to(I * (Om[k + 1] - Om[k]) / dt == moment)
                opti.subject_to(C[k + 1, :].T == ck + dt * Cd[k + 1, :].T)
                opti.subject_to(Th[k + 1] == Th[k] + dt * Om[k + 1])
            for t in range(Nc):
                for leg in range(2):
                    fx, fz = F[t, 2 * leg], F[t, 2 * leg + 1]
                    if not stance[t, leg]:
                        opti.subject_to(fx == 0)
                        opti.subject_to(fz == 0)
                    else:
                        opti.subject_to(opti.bounded(fmin, fz, fmax))
                        opti.subject_to(opti.bounded(-mu * fz, fx, mu * fz))
            for k in range(1, N + 1):
                base = C[k, :].T - rot(Th[k], c_off)
                for leg in range(2):
                    hip = base + rot(Th[k], hips[leg])
                    if k <= (Ndc if leg == 0 else Nc):
                        d = hip - ca.DM(feet0[leg])
                        opti.subject_to(opti.bounded(reach_min[leg] ** 2, ca.sumsqr(d),
                                                     reach_max[leg] ** 2))
                        opti.subject_to(hip[1] >= feet0[leg][1] + 0.5 * reach_min[leg])
                    elif k < N - 3:
                        lift = 0.6 * reach_max[leg]
                        # the box smoothed over 2 cm here: this is only a guess, and a
                        # sharp step in a hip constraint stalls the small solve
                        need = (self.box_profile(hip[0], 0.0, box, 0.02)
                                if box is not None else 0.0)
                        opti.subject_to(hip[1] - lift - radius
                                        >= need + (clearance if k > Nc else 0.0))
                        if box is not None and box_clearance > 0:
                            opti.subject_to(hip[1] - lift - radius >= self.box_edge(
                                hip[0], box, box_clearance, box_setback + 0.05, 0.02))
            cost = w_goal * ((C[N, 0] - goal[0]) ** 2 + 3 * (C[N, 1] - goal[1]) ** 2
                             + 3 * (Th[N] - theta_goal) ** 2)
            cost += w_force * ca.sumsqr(F) + w_smooth * ca.sumsqr(F[1:, :] - F[:-1, :])
            cost += w_omega * ca.sumsqr(Om)
            opti.minimize(cost)
            # a physical start: a smooth push to a takeoff state, then the ballistic arc
            # from there to the goal (from a straight line this solve stalls too)
            T_fl = (N - Nc) * dt
            c_to = c0 + np.array([0.3 * (goal[0] - c0[0]), 0.03])
            v_to = (np.asarray(goal) - c_to) / T_fl + np.array([0.0, 0.5 * g * T_fl])
            c_g, cd_g = np.zeros((N + 1, 2)), np.zeros((N + 1, 2))
            for k in range(N + 1):
                if k <= Nc:
                    c_g[k] = hermite(k * dt, 0.0, Nc * dt, c0, c_to, np.zeros(2), v_to)
                else:
                    t = (k - Nc) * dt
                    c_g[k] = c_to + v_to * t + np.array([0.0, -0.5 * g * t ** 2])
            cd_g[1:] = np.diff(c_g, axis=0) / dt
            opti.set_initial(C, c_g)
            opti.set_initial(Cd, cd_g)
            f_guess = np.zeros((Nc, nu))
            f_guess[:, 1::2] = stance * m * g / np.maximum(stance.sum(1, keepdims=True), 1)
            opti.set_initial(F, f_guess)
            opti.solver("ipopt", {"print_time": False},
                        {"print_level": 0, "sb": "yes", "max_iter": 1000})
            try:
                sol, ok = opti.solve(), True
            except RuntimeError:
                sol, ok = opti.debug, False
                if os.environ.get("ILC_SRB_DEBUG"):
                    opti.debug.show_infeasibilities(1e-4)
            return dict(success=ok, status=opti.stats().get("return_status", ""),
                        c=np.array(sol.value(C)).reshape(N + 1, 2),
                        cd=np.array(sol.value(Cd)).reshape(N + 1, 2),
                        theta=np.array(sol.value(Th)).ravel(),
                        omega=np.array(sol.value(Om)).ravel(),
                        f=np.array(sol.value(F)).reshape(Nc, nu))

        q_lo, q_hi = fb.joint_range[:, 0], fb.joint_range[:, 1]
        home_rel_body = [feet0[leg] - (s0[:2] + hips[leg]) for leg in range(2)]  # foot - hip, level

        def swing_path(leg, c, th):
            """World foot targets while a leg is in the air: from where it lifted off, moving
            with its hip at first (the leg leaves the ground with the body, the joints at
            rest; held back at the liftoff spot, the leg is dragged straight into the knee
            limit), through a via point, to its landing spot under the hip (home pose at N). The
            via point sits over the box edge when the foot crosses it, timed to when the
            hip gets there; otherwise mid-swing, tucked above both ends."""
            k_off = Ndc if leg == 0 else Nc
            p0 = feet0[leg]
            base_N = c[N] - R(th[N]) @ c_off
            pN = base_N + R(th[N]) @ (hips[leg] + home_rel_body[leg])
            hip_xz = np.array([c[k] - R(th[k]) @ c_off + R(th[k]) @ hips[leg]
                               for k in range(N + 1)])
            hip_x = hip_xz[:, 0]
            v0 = (hip_xz[k_off + 1] - hip_xz[k_off]) / dt
            if box is not None and p0[0] < box["x_front"] < pN[0]:
                past = np.where(hip_x[k_off:] >= box["x_front"] - 0.02)[0]
                k_via = k_off + (int(past[0]) if past.size else (N - k_off) // 2)
                via = np.array([box["x_front"],
                                box["height"] + box_clearance + radius + 0.01])
            else:
                k_via = (k_off + N) // 2
                via = np.array([0.5 * (p0[0] + pN[0]), max(p0[1], pN[1]) + tuck])
            # with a landing clearance, the foot hovers above its landing spot and comes
            # straight down onto it over the last n_landing_ramp samples
            k_end, p_end, v_end = N, pN, np.zeros(2)
            if landing_clearance is not None:
                k_end = N - n_landing_ramp
                p_end = pN + np.array([0.0, landing_clearance])
                v_end = np.array([0.0, -landing_clearance / (n_landing_ramp * dt)])
            k_via = int(np.clip(k_via, k_off + 3, k_end - 4))
            v_via = (p_end - p0) / ((k_end - k_off) * dt)
            path = {}
            for k in range(k_off + 1, N + 1):
                t = k * dt
                if k <= k_via:
                    path[k] = hermite(t, k_off * dt, k_via * dt, p0, via, v0, v_via)
                elif k <= k_end:
                    path[k] = hermite(t, k_via * dt, k_end * dt, via, p_end, v_via, v_end)
                else:
                    path[k] = p_end + (pN - p_end) * (k - k_end) / (N - k_end)
            return path

        def full_body(c, th):
            paths = [swing_path(leg, c, th) for leg in range(2)]
            s_g = np.zeros((N + 1, fb.ns))
            q = fb.q_home.copy()
            for k in range(N + 1):
                base = c[k] - R(th[k]) @ c_off
                targets = [paths[leg].get(k, feet0[leg]) for leg in range(2)]
                # IK: damped least squares on the planar model's feet, base fixed
                s_k = np.concatenate([base, [th[k]], q])
                for _ in range(60):
                    err = np.concatenate(targets) - fb.feet(s_k).full().ravel()
                    if np.abs(err).max() < 1e-5:
                        break
                    J = fb.Jc(s_k).full()[:, 3:]
                    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(4), err)
                    s_k[3:] = np.clip(s_k[3:] + dq, q_lo, q_hi)
                q = s_k[3:].copy()
                s_g[k] = s_k
            s_g[N, 3:] = fb.q_home
            return s_g

        srb = solve_srb()
        s_g = full_body(srb["c"], srb["theta"])

        sd_g = np.vstack([np.zeros(fb.ns), np.diff(s_g, axis=0) / dt])
        tau_max = margin * 2.0 * fb.torque_limit
        tau_g = np.zeros((N, fb.nj))
        for k in range(N):
            fk = srb["f"][k] if k < Nc else np.zeros(nu)
            gen = (fb.H(s_g[k]).full() @ (sd_g[k + 1] - sd_g[k]) / dt
                   + fb.h(s_g[k], sd_g[k]).full().ravel() - fb.Jc(s_g[k]).full().T @ fk)
            tau_g[k] = np.clip(gen[3:] - fb.tau_fric(sd_g[k, 3:]).full().ravel(), -tau_max, tau_max)
        return dict(s=s_g, sd=sd_g, tau=tau_g, f=srb["f"]), srb

    def init_trajopt(self, fb, goal, swing_mask, dt, theta_goal=0.0, box=None, clearance=0.02,
                     n_touchdown=3, qd_max=30.0, margin=0.8, joint_margin=0.1,
                     swing_qd=10.0, w_swing=1.0, w_goal=1e5, w_tau=1e-4, w_smooth=1e-4,
                     box_clearance=0.0, box_setback=0.0, max_iter=500, terrain_width=0.005,
                     init_guess=None, ipopt_options=None, guess="line",
                     landing_pitch_rate=None, flight_pitch_rate=None, flight_min_pitch=None,
                     flight_spin_change=None, leg_clearance=None, landing_clearance=None,
                     n_landing=15, n_landing_ramp=6, landing_capture_margin=None):
        """
        Reference for trial 1 and x_ref for every trial, from the full-body dynamics.

        fb         : PlanarQuadModel, the NOMINAL robot (the ILC corrects the mismatch)
        goal       : (c_x, c_z) whole-body CoM at sample N. The robot starts at
                     fb.standing_state() (ground at z = 0, CoM at x = 0), so standing on a box
                     of height h is c_z = c_z(0) + h
        swing_mask : (Nc, nu) contact schedule (Ephase); force u_t acts over [t, t+1]
        box        : None, or dict(x_front=..., height=...) in the same world frame
        clearance  : minimum foot height above the terrain during flight (m)
        box_clearance, box_setback : a foot off the ground -- the front one swinging during
                     rear-leg contact as well as both in flight -- clears the box's front
                     edge by box_clearance (m) whenever it is within box_setback (m) of it,
                     horizontally. Without it the swinging front foot may skim the edge,
                     and a jump that pushes a little late clips it and tumbles
        max_iter   : IPOPT iteration cap. Jumps well inside the robot's limits solved in
                     10-60 s; a task at or past them can grind for tens of minutes
                     before giving up, so the cap is kept low
        terrain_width : m over which the box's front face and edge zone are smoothed (tanh)
        init_guess : None, or dict(s, sd, tau, f) -- a previous solution to start from (see
                     JumpILC's to_steps); overrides `guess`
        guess      : what to start from without one: "line" (the standing pose slid toward
                     the goal) or "srb" (a single-rigid-body plan made full-body, srb_guess)
        flight_min_pitch : None, or the lowest trunk pitch (rad, nose up +) from takeoff to
                     touchdown. Nose-up pitch in flight is harmless; nose-down, the front feet
                     reach the box first and the robot tips over them
        flight_spin_change : None, or how far apart (rad/s) the trunk's fastest and slowest
                     pitch rates in flight may be. Free, plans swing the legs as a
                     reaction wheel to brake a 150-220 deg/s nose-down spin after takeoff to
                     the landing rate; the flown legs lag that swing, brake less, and the
                     robot lands 9 deg nose-down and tips over
        flight_pitch_rate : None, or the largest |pitch rate| (rad/s) of the trunk from
                     takeoff (sample Nc) to touchdown. Uncapped, a box plan pitches up 40-48
                     deg at takeoff and spins nose-down at ~200 deg/s, relying on the swinging
                     legs to brake it to landing_pitch_rate; the flown legs lag and brake less,
                     so the robot lands 11-14 deg nose-down and tips over
        leg_clearance : None, or the smallest x-distance (m) in the trunk frame between any
                     point of the front leg (hip, knee, foot) and any point of the rear leg.
                     The sagittal model has no self-collision: without this it may tuck the
                     front feet back until they catch on the rear hips and thighs in flight,
                     or cross the calves, and the front legs then cannot swing out to land
        landing_clearance, n_landing, n_landing_ramp : None, or the height (m) every foot
                     keeps above the surface under it over the last n_landing samples of
                     flight, ramped down to 0 over the final n_landing_ramp. Flown, the front
                     feet run 3-7 cm below the plan late in flight (leg tracking and pitch
                     lag), so a plan skimming the box top 2-5 cm up lands on the front feet
                     early and pitches the robot over
        landing_pitch_rate : None, or the largest |pitch rate| (rad/s) the trunk may have at
                     touchdown (sample N). Pitch itself is only pulled to theta_goal at N;
                     without this a plan can land level but spinning nose-down at 130-170
                     deg/s, and the robot then tips over on its landing
        ipopt_options : extra IPOPT options, e.g. {"mu_strategy": "adaptive"}
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
        if landing_pitch_rate is not None:
            opti.subject_to(opti.bounded(-landing_pitch_rate, Sd[N, 2], landing_pitch_rate))
        if flight_pitch_rate is not None:
            for k in range(Nc, N + 1):
                opti.subject_to(opti.bounded(-flight_pitch_rate, Sd[k, 2], flight_pitch_rate))
        if flight_min_pitch is not None:
            for k in range(Nc, N + 1):
                opti.subject_to(S[k, 2] >= flight_min_pitch)
        if flight_spin_change is not None:
            # every flight pitch rate inside one band [lo, lo + flight_spin_change]
            spin_lo = opti.variable()
            for k in range(Nc + 1, N + 1):
                opti.subject_to(opti.bounded(spin_lo, Sd[k, 2], spin_lo + flight_spin_change))
            opti.set_initial(spin_lo, -np.radians(90.0))

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

        # feet off the ground stay above the terrain (with a margin in flight); knees and
        # calf midpoints too
        for k in range(1, N + 1):
            feet = fb.feet(S[k, :].T)
            for foot in range(2):
                xz = feet[2 * foot:2 * foot + 2]
                if k in pinned[foot]:
                    opti.subject_to(xz == feet0[2 * foot:2 * foot + 2])
                    continue
                height = xz[1] - radius - self.box_profile(xz[0], ground_z, box, terrain_width)
                if box is not None and box_clearance > 0 and k < N - n_touchdown:
                    opti.subject_to(xz[1] - radius - ground_z >= self.box_edge(
                        xz[0], box, box_clearance, box_setback, terrain_width))
                if k <= Nc:                           # lifted off while the other foot pushes
                    opti.subject_to(height >= 0)
                elif k < N - n_touchdown:             # flight
                    opti.subject_to(height >= clearance)
                else:                                 # descending onto the landing surface
                    opti.subject_to(height >= 0)
                if landing_clearance is not None and k > max(Nc, N - n_landing):
                    opti.subject_to(height >= landing_clearance * min(1.0, (N - k) / n_landing_ramp))
            knees = fb.knees(S[k, :].T)
            for i in range(2):
                opti.subject_to(knees[2 * i + 1] >= self.box_profile(knees[2 * i], ground_z, box,
                                                                     terrain_width))
                mid = 0.5 * (knees[2 * i:2 * i + 2] + feet[2 * i:2 * i + 2])
                opti.subject_to(mid[1] >= self.box_profile(mid[0], ground_z, box, terrain_width))
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
        if landing_capture_margin is None:
            opti.subject_to(S[N, 3:].T == fb.q_home)
        else:
            # the front legs land reaching ahead of the landing's capture point, CoM x + vx
            # sqrt(h/g), by this margin: braking the landing's momentum then keeps the ground
            # force behind the front feet and the rear feet loaded. At the home pose the
            # front feet land right at it (~0.2 m ahead at ~1.35 m/s), so the rear feet lift
            # while the robot stops. The rear legs still land in the home pose
            opti.subject_to(S[N, 5:].T == fb.q_home[2:])
            cN_, cP_ = fb.com(S[N, :].T), fb.com(S[N - 1, :].T)
            footN = fb.feet(S[N, :].T)
            vxN = (cN_[0] - cP_[0]) / dt
            hN = cN_[1] - (footN[1] - radius)
            opti.subject_to(footN[0] - cN_[0] >= vxN * ca.sqrt(ca.fmax(hN, 0.05) / 9.81)
                            + landing_capture_margin)

        # legs keep clear of each other: along the trunk, every point of the front leg (hip,
        # knee, foot) stays leg_clearance ahead of every point of the rear leg, so the
        # thighs and calves of the two cannot cross or touch
        if leg_clearance is not None:
            hips = [leg["hip"] for leg in fb.legs]                  # trunk frame
            for k in range(1, N + 1):
                sk = S[k, :].T
                c, sn = ca.cos(sk[2]), ca.sin(sk[2])
                along = lambda pt: c * (pt[0] - sk[0]) + sn * (pt[1] - sk[1])   # trunk x
                feet, knees = fb.feet(sk), fb.knees(sk)
                front = [hips[0][0], along(knees[0:2]), along(feet[0:2])]
                rear = [hips[1][0], along(knees[2:4]), along(feet[2:4])]
                for i, a in enumerate(front):
                    for j, b in enumerate(rear):
                        if i or j:                          # hip to hip is fixed
                            opti.subject_to(a >= b + leg_clearance)

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
        srb_info = None
        if init_guess is None and guess == "srb":
            init_guess, srb_info = self.srb_guess(
                fb, goal, swing_mask, dt, box=box, margin=margin, clearance=clearance,
                box_clearance=box_clearance, box_setback=box_setback, theta_goal=theta_goal,
                landing_pitch_rate=landing_pitch_rate, flight_pitch_rate=flight_pitch_rate,
                flight_min_pitch=flight_min_pitch,
                landing_clearance=landing_clearance, n_landing_ramp=n_landing_ramp)
        elif guess not in ("line", "srb"):
            raise ValueError(f"guess must be line or srb, got {guess!r}")
        if init_guess is not None:
            opti.set_initial(S, init_guess["s"])
            opti.set_initial(Sd, init_guess["sd"])
            opti.set_initial(Tau, init_guess["tau"])
            opti.set_initial(F, init_guess["f"])

        opts = {"print_level": 0, "sb": "yes", "max_iter": int(max_iter)}
        opts.update(ipopt_options or {})
        opti.solver("ipopt", {"print_time": False}, opts)
        try:
            sol = opti.solve()
            ok = True
        except RuntimeError:
            sol, ok = opti.debug, False
        stats = opti.stats()
        # how far the returned point is from satisfying the dynamics (a converged plan: ~1e-9)
        dyn_res = 0.0
        for k in range(N):
            sk, sdk = np.array(sol.value(S[k, :])).ravel(), np.array(sol.value(Sd[k, :])).ravel()
            sdn = np.array(sol.value(Sd[k + 1, :])).ravel()
            f = np.array(sol.value(F[k, :])).ravel() if k < Nc else np.zeros(nu)
            tk = np.array(sol.value(Tau[k, :])).ravel()
            lhs = fb.H(sk).full() @ (sdn - sdk) / dt + fb.h(sk, sdk).full().ravel()
            rhs = fb.S @ (tk + fb.tau_fric(sdk[3:]).full().ravel()) + fb.Jc(sk).full().T @ f
            dyn_res = max(dyn_res, float(np.abs(lhs - rhs).max()))
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
        info = dict(success=ok, status=stats.get("return_status", ""), srb=srb_info,
                    iterations=int(stats.get("iter_count", -1)), dynamics_residual=dyn_res,
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
            if self.mdc == "go1":
                # Go1 envelope at this sample's measured speed: the limit drops only on
                # the side that drives the joint along qdot (motoring)
                tau_max = np.broadcast_to(np.asarray(p["tau_max"], float), (nj,))[idx]
                w_max = self.w_max[idx]
                w_knee = 0.5 * w_max
                drop = np.clip((w_max - np.abs(qd)) / (w_max - w_knee), 0.0, 1.0)
                lb = np.where(qd < 0, -tau_max * drop, -tau_max)
                ub = np.where(qd > 0, tau_max * drop, tau_max)
            else:
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



def _same_task(a, b):
    """Whether two JumpILC configs are the same task. A key one of them lacks counts as
    None there: options added since an older run was recorded default to off."""
    a, b = json.loads(json.dumps(a)), json.loads(json.dumps(b))
    return all(a.get(k) == b.get(k) for k in set(a) | set(b))


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
    box_clearance, box_setback : how far moving feet stay from the box's front edge
                (init_trajopt)
    terrain_width : smoothing of the box's face and edge zone in the TO (init_trajopt)
    to_steps  : solve a box plan by continuation: the box height and the jump's height
                gain grow over this many solves, each started from the one before, the
                last being the task itself (1: solve the task directly). Only how the plan
                is found -- the plan it converges to satisfies the same constraints. From
                the standing-pose guess alone, IPOPT failed every Go1 (50, 20) and (50, 15)
                box plan; in 4 steps each stage converged. The SRB guess (to_guess) now
                does the same in one solve, so this is off by default
    ipopt_options : extra IPOPT options for the TO
    to_guess  : the TO's starting point (init_trajopt's guess): "srb" (default), a
                single-rigid-body plan made full-body by IK (srb_guess), or "line", the
                standing pose slid toward the goal. From "srb", one solve found exact Go1
                (50, 20), (50, 15), (60, 10) and A1 (50, 20) box plans in 11-28 s, where
                "line" failed; flat jumps solve from either. With to_steps > 1 it seeds
                the first step only
    swing_qd, w_swing : soft speed limit on swing/flight legs and its weight (init_trajopt)
    landing_pitch_rate : largest |pitch rate| at touchdown in the plan, rad/s (init_trajopt)
    flight_pitch_rate : largest |pitch rate| from takeoff to touchdown, rad/s (init_trajopt)
    flight_min_pitch : lowest trunk pitch from takeoff to touchdown, rad (init_trajopt)
    flight_spin_change : how far the trunk's pitch rate may move in flight, rad/s (init_trajopt)
    leg_clearance : trunk-frame x-distance kept between every point of the front leg and
                every point of the rear leg in the plan, m (init_trajopt)
    landing_capture_margin : None, or plan the front feet to land this far (m) ahead of the
                landing's capture point, front legs free of the home pose (init_trajopt)
    landing_clearance : height every foot keeps above its landing surface late in flight
                until it comes straight down onto it, m (init_trajopt)
    Qe_diag, Qu_diag : ILC weights on the state error and the trial-to-trial force step
    Qu_stage3 : step weight for Stage III alone (None: Qu_diag). Stage III weighs only
                the landing state, a handful of rows against every force sample, so a
                step small enough for Stages I-II can overshoot there, and the unweighted
                directions drift trial to trial
    secant    : correct the SRB model's landing sensitivities from the robot (Broyden). Each
                trial is a step du from some trial; the landing changed by dx where the
                model said G_N du, so G_N gains the rank-one correction
                (dx - G_N du) du^T / |du|^2 -- exact along every direction stepped since.
                Stage III plans with the corrected G_N. On Go1 the SRB model gets the
                landing x right but its pitch 5-10x too sensitive to the forces, often with
                the wrong sign (the legs' own momentum, which it lacks), so uncorrected
                Stage III chases a pitch correction the forces do not make
    mu        : ground friction the plan, the ILC's friction cones and the landing assume
                (the paper's 0.6); the plan uses margin x mu of it at most
    safeguard : Stage III rolls a trial that landed worse than the best so far (or fell)
                back to the best trial's forces with a shorter step (see update). False:
                the paper's law, a step from every trial whatever it did -- the error is
                measured over the jump window [0, N] only, so a trial that fell off the box
                after landing is as good a measurement as any
    gain, gain_stage3 : ILC learning gain gamma, U <- U + gamma du* (1: the paper's law);
                Stage III uses gain_stage3 instead when > 0
    forget    : leaky ILC (anti-windup): U <- u_ref + forget (U - u_ref) + gamma du*, U the
                forces the step is taken from (after any safeguard rollback); 1: no leak
    tau_scale : the QP's per-motor torque limit tau_max x this (a weakened motor)
    fmin      : the QP's (and force clipping's) least normal force per stance pair, N
                (None: mdc_params'). Set after the TO, which keeps its own
    slack_weight : the QP's penalty on softened torque/MDC rows (QuadILCStageSolver)
    mdc, mdc_speed_scale : the QP's motor model, "legacy" (the paper's MDC) or "go1" (the
                Go1 torque-speed envelope at each sample's measured joint speed); see
                QuadILCStageSolver
    stage3_theta_mode : Stage III's landing pitch rows: "track" (aim at landing_pitch_target,
                pitch rate as planned), "hold" (target = the trial's own landing pitch and
                pitch rate: a pitch-neutral step in the model), "free" (theta and omega
                unweighted)
    landing_pitch_target : Stage III's target landing pitch, rad (the safeguard scores
                against it too; theta_err and convergence stay |pitch|)
    fall_pitch_bias : on a trial that fell, the pitch the step sees from the start of the
                free-flight tail (log["measured_until"], default N - 10) on is lowered by
                this, rad: as if it had landed more nose-down, so the step pushes nose-up
    converge_any_stage : converged may be declared in Stages I-II too (not only Stage III)
    converged_requires_no_fall : a trial that fell is never converged
    safeguard_growth : the Stage III step weight's growth on a rejection (backoff "qu")
    stage3_qu_scale_max : cap on that step weight's scale (None: none)
    stage3_backoff : what a rejection shortens: "qu" (the step weight, x safeguard_growth)
                or "alpha" (the ordinary step from the best trial, scaled by
                0.5 ** consecutive rejections; the step weight stays 1x)
    stage3_accept_rows : safeguard cost on the landing's "xzth" (x, z, pitch) or "xz" rows
    stage3_accept_tol : a trial is the best if its cost <= the best's x (1 + this)
    stage3_best_refresh : a trial flown with the best trial's forces again (within 1e-3 N)
                makes the best's cost the mean of every cost measured with them, rather
                than its (luckiest) first one
    stage3_fall_policy : a Stage III trial that fell "rollback"s to the best trial like any
                rejection, or is "step"ped from itself (still never the best)
    reference : None to run the TO, or a path written by save_reference() to reuse one

    A trial log may also carry R1, R2 (N, 2): measured lever arms (contact point - CoM,
    as the TO's), used instead of the plan's for that trial's step.
    """

    nx, nu = 6, 4                 # x = [px, pz, theta, vx, vz, omega], u = [f1x, f1z, f2x, f2z]

    def __init__(self, qm, jump=(0.50, 0.10), box=dict(x_front=0.25, height=0.10),
                 phases=(20, 20, 25), dt=0.01, n_stage1=2, n_stage2=3, pos_tol=0.01,
                 theta_tol=np.radians(1.0), margin=0.8, swing_qd=10.0, w_swing=1.0,
                 box_clearance=0.04, box_setback=0.03, terrain_width=0.005, to_steps=1,
                 to_guess="srb", landing_pitch_rate=np.radians(90.0),
                 flight_pitch_rate=None, flight_min_pitch=0.0,
                 flight_spin_change=np.radians(20.0), leg_clearance=0.10,
                 landing_clearance=0.06, landing_capture_margin=None,
                 ipopt_options=None,
                 mdc_params=None, Qe_diag=(1.0, 3.0, 3.0, 0.01, 0.01, 0.01), Qu_diag=1e-5,
                 Qu_stage3=None, safeguard=True, secant=False, mu=0.6, reference=None,
                 gain=1.0, gain_stage3=0.0, forget=1.0, tau_scale=1.0, fmin=None,
                 slack_weight=1e2, mdc="legacy", mdc_speed_scale=1.0,
                 stage3_theta_mode="track", landing_pitch_target=0.0, fall_pitch_bias=0.0,
                 converge_any_stage=False, converged_requires_no_fall=False,
                 safeguard_growth=4.0, stage3_qu_scale_max=None, stage3_backoff="qu",
                 stage3_accept_rows="xzth", stage3_accept_tol=0.0,
                 stage3_best_refresh=False, stage3_fall_policy="rollback",
                 _check_reference=True):
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
                                             mu=float(mu))
        self.config = dict(robot=qm.robot, jump=[float(v) for v in jump], box=self.box,
                           phases=[self.Ndc, self.Nsc, self.Nfl], dt=self.dt, margin=margin,
                           swing_qd=float(swing_qd), w_swing=float(w_swing),
                           fmax=float(self.mdc_params["fmax"]),
                           landing_pitch_rate=None if landing_pitch_rate is None
                           else float(landing_pitch_rate),
                           flight_pitch_rate=None if flight_pitch_rate is None
                           else float(flight_pitch_rate),
                           flight_min_pitch=None if flight_min_pitch is None
                           else float(flight_min_pitch),
                           flight_spin_change=None if flight_spin_change is None
                           else float(flight_spin_change),
                           leg_clearance=None if leg_clearance is None else float(leg_clearance),
                           landing_clearance=None if landing_clearance is None
                           else float(landing_clearance))
        if self.box is not None:            # only a box plan depends on these
            self.config.update(box_clearance=float(box_clearance), box_setback=float(box_setback))
            if terrain_width != 0.005:
                self.config["terrain_width"] = float(terrain_width)
        if landing_capture_margin is not None:    # only such a plan records it
            self.config["landing_capture_margin"] = float(landing_capture_margin)
        if self.mdc_params["mu"] != 0.6:    # a plan for other friction (0.6: older references)
            self.config["mu"] = float(self.mdc_params["mu"])

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
                                         fb.torque_map, slack_weight=slack_weight, mdc=mdc,
                                         mdc_speed_scale=mdc_speed_scale)
        if reference is None:
            to_kwargs = dict(margin=margin, swing_qd=swing_qd, w_swing=w_swing,
                             box_clearance=box_clearance, box_setback=box_setback,
                             terrain_width=terrain_width, ipopt_options=ipopt_options,
                             guess=to_guess, landing_pitch_rate=landing_pitch_rate,
                             flight_pitch_rate=flight_pitch_rate,
                             flight_min_pitch=flight_min_pitch,
                             flight_spin_change=flight_spin_change,
                             leg_clearance=leg_clearance,
                             landing_clearance=landing_clearance,
                             landing_capture_margin=landing_capture_margin)
            steps = int(to_steps) if self.box is not None else 1
            c_home = fb.com(self.s_home).full().ravel()
            guess, self.to_trace = None, []
            for lam in np.arange(1, steps + 1) / steps:
                box_l = None if self.box is None else dict(self.box,
                                                           height=lam * self.box["height"])
                goal_l = c_home + np.array([jump[0], lam * jump[1]])
                self.x_ref, self.u_ref, self.to_info = self.solver.init_trajopt(
                    fb, goal_l, self.swing_mask, self.dt, box=box_l, init_guess=guess,
                    **to_kwargs)
                info = self.to_info
                guess = dict(s=info["s"], sd=info["sd"], tau=info["tau"], f=self.u_ref)
                self.to_trace.append(dict(fraction=float(lam), success=info["success"],
                                          status=info["status"], iterations=info["iterations"],
                                          dynamics_residual=info["dynamics_residual"],
                                          goal_miss=info["goal_miss"]))
        else:
            self._load_reference(reference, check=_check_reference)
        # the ILC's own limits, set after the TO so the plan does not depend on them
        self.mdc_params = dict(self.mdc_params,
                               tau_max=np.asarray(self.mdc_params["tau_max"], float)
                               * float(tau_scale))
        if fmin is not None:
            self.mdc_params["fmin"] = float(fmin)
        self.solver.p = self.mdc_params
        # time-varying lever arms (contact point - CoM) from the TO
        self.R1, self.R2 = self.to_info["R1"], self.to_info["R2"]

        self.U = self.u_ref.copy()                # trial 1 flies the TO forces (eq. 29)
        # Stage III safeguard (see update): the best Stage III trial so far, and how much
        # the step weight is scaled up after steps that made the landing worse
        self.safeguard = bool(safeguard)
        self.secant = bool(secant)
        self.G_corr = np.zeros((nx, self.Nc * nu))    # learned correction to G's landing rows
        # a correction given up front (set_secant_prior): fitted offline from many trials on
        # this robot, held fixed (no Broyden updates unless `secant`)
        self.secant_prior = False
        self._origin = None                           # what the last step started from
        self.best3 = None
        self.qu3_scale = 1.0
        self.n_rejected = 0                           # consecutive Stage III rejections
        for name, value, options in (("stage3_theta_mode", stage3_theta_mode, ("track", "hold", "free")),
                                     ("stage3_backoff", stage3_backoff, ("qu", "alpha")),
                                     ("stage3_accept_rows", stage3_accept_rows, ("xzth", "xz")),
                                     ("stage3_fall_policy", stage3_fall_policy, ("rollback", "step"))):
            if value not in options:
                raise ValueError(f"{name} must be one of {options}, got {value!r}")
        self.gain, self.gain_stage3, self.forget = float(gain), float(gain_stage3), float(forget)
        self.theta_mode = stage3_theta_mode
        self.theta_goal = float(landing_pitch_target)
        self.fall_pitch_bias = float(fall_pitch_bias)
        self.converge_any_stage = bool(converge_any_stage)
        self.converged_requires_no_fall = bool(converged_requires_no_fall)
        self.safeguard_growth = float(safeguard_growth)
        self.qu3_scale_max = np.inf if stage3_qu_scale_max is None else float(stage3_qu_scale_max)
        self.backoff = stage3_backoff
        self.accept_rows = 2 if stage3_accept_rows == "xz" else 3
        self.accept_tol = float(stage3_accept_tol)
        self.best_refresh = bool(stage3_best_refresh)
        self.fall_policy = stage3_fall_policy
        self.trial = 0                            # trials flown so far
        self.history = []

    # -- reference persistence (the TO takes tens of seconds) ------------------------------
    _INFO_KEYS = ("s", "sd", "tau", "q_ref", "qd_ref", "R1", "R2", "feet_rel", "landing",
                  "foot_radius", "ground_z", "success", "goal_miss", "min_clearance")

    def reference_arrays(self):
        """The reference as arrays: what save_reference writes and trial files carry."""
        return dict(x_ref=self.x_ref, u_ref=self.u_ref, config=json.dumps(self.config),
                    **{f"info_{k}": self.to_info[k] for k in self._INFO_KEYS})

    def save_reference(self, path):
        np.savez(path, **self.reference_arrays())

    def _load_reference(self, src, check=True):
        """src: a path written by save_reference, or a mapping of the same arrays (a
        TrialRecord's .reference). check: refuse one made for another task/TO setting."""
        data = np.load(src) if isinstance(src, (str, os.PathLike)) else src
        saved = json.loads(str(data["config"]))
        if check and not _same_task(saved, self.config):
            raise ValueError(f"reference {src if isinstance(src, str) else ''} was made for "
                             f"{saved}, not {self.config}")
        self.x_ref, self.u_ref = np.asarray(data["x_ref"]), np.asarray(data["u_ref"])
        self.to_info = {k: np.asarray(data[f"info_{k}"]) for k in self._INFO_KEYS}
        for k in ("foot_radius", "ground_z", "goal_miss", "min_clearance"):
            self.to_info[k] = float(self.to_info[k])
        self.to_info["success"] = bool(self.to_info["success"])

    # -- resuming and transferring (trial files from trial_log.TrialRecorder) --------------
    def restore(self, rec):
        """Continue learning from a recorded trial of this same task: the next trial flies
        that trial's U_next, and the stage schedule and history carry on from it."""
        if not _same_task(rec["config"], self.config):
            raise ValueError(f"{rec.get('path', 'trial')} is from task {rec['config']}, not "
                             f"{self.config}; use transfer_from for another task")
        self.U = np.asarray(rec["U_next"], float).copy()
        self.trial = int(rec["trial"])
        self.history = list(rec["history"])
        # the safeguard's step scale carries on; its best trial does not (not recorded),
        # so the next Stage III trial becomes the best
        self.qu3_scale = float(self.history[-1].get("qu3_scale", 1.0)) if self.history else 1.0
        self.n_rejected = int(self.history[-1].get("n_rejected", 0)) if self.history else 0

    def transfer_from(self, rec):
        """
        Start this task from what another task learned ("retarget"): this task keeps its
        own TO reference, and trial 1 flies its TO forces plus the correction the other
        task learned on top of its TO forces, U_s - u_ref,s. That correction is mostly
        what the model gets wrong about the robot (force it does not produce, lever arms),
        which carries over between jumps. It is resampled phase by phase onto this task's
        contact schedule (all-leg contact, then rear-leg contact), and the result clipped
        into the force limits and friction cone.
        """
        cfg = rec["config"]
        if cfg["robot"] != self.config["robot"]:
            raise ValueError(f"{rec.get('path', 'trial')} was flown by {cfg['robot']}, "
                             f"not {self.config['robot']}")
        src_dc, src_sc = int(cfg["phases"][0]), int(cfg["phases"][1])
        dU_src = np.asarray(rec["U_next"], float) - np.asarray(rec["u_ref"], float)
        dU = np.zeros_like(self.u_ref)
        for (s0, n_src), (d0, n_dst) in (((0, src_dc), (0, self.Ndc)),
                                           ((src_dc, src_sc), (self.Ndc, self.Nsc))):
            t_src = (np.arange(n_src) + 0.5) / n_src
            t_dst = (np.arange(n_dst) + 0.5) / n_dst
            for j in range(self.nu):
                dU[d0:d0 + n_dst, j] = np.interp(t_dst, t_src, dU_src[s0:s0 + n_src, j])
        self.U = self._clip_forces(self.u_ref + dU)
        self.transferred_from = rec.get("path")

    @classmethod
    def from_trial(cls, qm, rec, jump, box=None, **kwargs):
        """
        The paper's transfer (Sec. II-D3): keep the simple task's reference -- its joint
        profile, TO torque and forces -- and its learned forces, aim at a new target, and
        learn in Stage III only. Nothing is re-planned; call prime() with the simple
        task's last trial before the first new one. `box` is the new task's box, which
        the reference itself was not planned around.
        """
        cfg = rec["config"]
        kwargs.setdefault("n_stage1", 0)
        kwargs.setdefault("n_stage2", 0)
        ilc = cls(qm, jump=jump, box=box, phases=cfg["phases"], dt=cfg["dt"],
                  margin=cfg["margin"], swing_qd=cfg["swing_qd"], w_swing=cfg["w_swing"],
                  landing_pitch_rate=cfg.get("landing_pitch_rate"),
                  flight_pitch_rate=cfg.get("flight_pitch_rate"),
                  flight_min_pitch=cfg.get("flight_min_pitch"),
                  flight_spin_change=cfg.get("flight_spin_change"),
                  leg_clearance=cfg.get("leg_clearance"),
                  landing_clearance=cfg.get("landing_clearance"),
                  reference=rec.reference, _check_reference=False, **kwargs)
        ilc.config["reference_from"] = dict(path=rec.get("path"), jump=cfg["jump"],
                                            box=cfg["box"])
        ilc.U = np.asarray(rec["U_next"], float).copy()
        ilc.transferred_from = rec.get("path")
        return ilc

    def prime(self, log):
        """
        One Stage III step before the first trial of a transferred task (eq. 22-23): the
        last trial of the simple task (its recorded log, flown with self.U) scored against
        this task's goal, and U moved to close it. Counts no trial.
        """
        du, info = self._step(log, stage=3)
        self.U = self.U + du
        self.U[self.swing_mask] = 0.0
        return info

    def _clip_forces(self, U):
        """U projected into the contact limits and friction cone, swing forces zero."""
        p = self.mdc_params
        U = U.copy()
        for pair in range(self.nu // 2):
            fx, fz = U[:, 2 * pair], U[:, 2 * pair + 1]
            fz[:] = np.clip(fz, p["fmin"], p["fmax"])
            fx[:] = np.clip(fx, -p["mu"] * fz, p["mu"] * fz)
        U[self.swing_mask] = 0.0
        return U

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
              fell   (optional) the trial failed its landing: never the best, never converged
                     if converged_requires_no_fall
              R1, R2 (optional, N x 2) measured lever arms, used instead of the plan's
              measured_until (optional) first sample of the free-flight tail (fall_pitch_bias)
        returns dict(trial, stage, pos_err, theta_err, clearance, converged, gain, forget,
                     du_norm, dU_ref_norm[, solver, qu3_scale, n_rejected, rejected])
        """
        X = np.asarray(log["X"], float)
        stage = self.stage()
        self._secant_update(X[-1])
        pos_err, th_err = self.landing_error(X[-1])
        fell = bool(log.get("fell", False))
        result = dict(trial=self.trial + 1, stage=stage, pos_err=pos_err, theta_err=th_err,
                      clearance=self.clearance(X),
                      converged=bool((stage == 3 or self.converge_any_stage)
                                     and pos_err < self.pos_tol and th_err < self.theta_tol
                                     and not (self.converged_requires_no_fall and fell)))
        self.trial += 1
        self.history.append(result)
        gain = self.gain_stage3 if stage == 3 and self.gain_stage3 > 0 else self.gain
        if result["converged"]:
            result.update(gain=gain, forget=self.forget, du_norm=0.0,
                          dU_ref_norm=float(np.linalg.norm(self.U - self.u_ref)))
            return result

        # Safeguard. Stage III steps on the SRB model's sensitivities of the landing state
        # alone; where they are off, every step can make the landing a little worse, and
        # with nothing to notice it drifts away trial after trial (1-10 cm over 10 trials
        # on Go1 boxes). So every trial is scored by the landing cost Stage III minimizes
        # (a trial that fell never counts as best), and in Stage III a trial worse than the
        # best so far -- from any stage: Stage II, tracking the whole reference, often ends
        # with a worse landing than one of its earlier trials -- sends the next trial back
        # to the best trial's forces with a 4x (safeguard_growth) more heavily weighted
        # (shorter) step from there; each improvement halves the weight back.
        # (safeguard=False: none of it, as in the paper -- every trial is stepped from)
        # Variants: the cost on x, z alone (accept_rows), a tolerance on "worse", the best's
        # cost re-averaged when its forces are flown again (best_refresh), a shorter step
        # by scaling the ordinary one (backoff "alpha"), and falls stepped from (fall_policy)
        e = X[-1, :3] - np.array([self.goal[0], self.goal[1], self.theta_goal])
        r = self.accept_rows
        cost = np.inf if log.get("fell") else float(e[:r] @ (self.Qe_diag[:r] * e[:r]))
        costs = [cost]
        if (self.best_refresh and self.best3 is not None and np.isfinite(cost)
                and np.abs(self.U - self.best3["U"]).max() < 1e-3):
            costs = self.best3["costs"] + [cost]          # the best's forces, flown again
            self.best3["costs"] = costs
            self.best3["cost"] = float(np.mean(costs))
            result["best_refreshed"] = True
        is_best = np.isfinite(cost) and (self.best3 is None
                                         or cost <= self.best3["cost"] * (1.0 + self.accept_tol))
        if is_best:
            self.best3 = dict(cost=cost if len(costs) == 1 else float(np.mean(costs)),
                              U=self.U.copy(), log=log, costs=costs)
        alpha = None
        if stage == 3 and self.safeguard:
            if is_best:
                self.n_rejected = 0
                if self.backoff == "qu":
                    self.qu3_scale = max(1.0, 0.5 * self.qu3_scale)
            elif self.best3 is not None and not (fell and self.fall_policy == "step"):
                self.U = self.best3["U"].copy()
                log = self.best3["log"]
                self.n_rejected += 1
                if self.backoff == "qu":
                    self.qu3_scale = min(self.qu3_scale * self.safeguard_growth,
                                         self.qu3_scale_max)
                else:
                    alpha = 0.5 ** self.n_rejected
                result["rejected"] = True
            result["qu3_scale"] = self.qu3_scale
            result["n_rejected"] = self.n_rejected
        du, info = self._step(log, stage)
        result["solver"] = info
        if alpha is not None:
            du = alpha * du
            result["alpha"] = alpha
        if gain != 1.0:
            du = gain * du
        if self.forget == 1.0:
            self.U = self.U + du                          # eq. (15): u_{k+1} = u_k + du*
        else:                                             # leaky: decay toward u_ref
            self.U = self.u_ref + self.forget * (self.U - self.u_ref) + du
        self.U[self.swing_mask] = 0.0
        result.update(gain=gain, forget=self.forget, du_norm=float(np.linalg.norm(du)),
                      dU_ref_norm=float(np.linalg.norm(self.U - self.u_ref)))
        return result

    def set_secant_prior(self, C):
        """Stage III plans with G_N + C: C (nx x Nc*nu, raw units) the landing-sensitivity
        correction a ridge fit over earlier trials' pairs gives (dilc_train.secant_fit)."""
        C = np.asarray(C, float)
        assert C.shape == self.G_corr.shape, (C.shape, self.G_corr.shape)
        self.G_corr = C.copy()
        self.secant_prior = True

    def _secant_update(self, x_N):
        """Broyden: the landing x_N of the trial just flown (self.U) against what the model
        predicted for the step that led to it (see `secant`)."""
        o = self._origin
        if not self.secant or o is None:
            return
        du = (self.U - o["U"]).reshape(-1)
        if np.abs(du).max() < 1e-3:
            return
        r = (x_N - o["x_N"]) - (o["G_N"] + self.G_corr) @ du
        self.G_corr += np.outer(r, du) / float(du @ du)

    def _step(self, log, stage):
        """The stage's QP on the trial in `log` (flown with self.U): the force step du*."""
        X = np.asarray(log["X"], float)
        nx, nu, N, Nc = self.nx, self.nu, self.N, self.Nc
        U_full = np.vstack([self.U, np.zeros((self.Nfl, nu))])
        # the trial's measured lever arms, when the log has them
        R1 = np.asarray(log["R1"], float) if "R1" in log else self.R1
        R2 = np.asarray(log["R2"], float) if "R2" in log else self.R2
        A_list, B_list = self.nominal.linearize_along_trial(X, U_full, R1, R2, self.dt)
        G = build_lifted_G(A_list, B_list, N, Nc, nx, nu, flatten=True)
        self._origin = dict(U=self.U.copy(), x_N=X[-1].copy(), G_N=G[-nx:].copy())
        if stage == 3 and (self.secant or self.secant_prior):
            G = G.copy()
            G[-nx:] += self.G_corr
        if self.fall_pitch_bias != 0.0 and log.get("fell"):
            # a fall: the step sees a more nose-down flight tail than measured
            X = X.copy()
            X[int(log.get("measured_until", N - 10)):, 2] -= self.fall_pitch_bias
        # Stages I-II track the reference; Stage III aims at the goal itself (eq. 21-23:
        # the desired landing position and pose), which is where the plan lands only when
        # the TO reached it -- and never after a transfer to a new target
        target = self.x_ref
        Qe = self.Qe_diag
        if stage == 3:
            target = self.x_ref.copy()
            target[-1, :2] = self.goal
            target[-1, 2] = self.theta_goal
            if self.theta_mode == "hold":           # no pitch correction in the model
                target[-1, 2], target[-1, 5] = X[-1, 2], X[-1, 5]
            elif self.theta_mode == "free":         # pitch rows unweighted
                Qe = Qe.copy()
                Qe[..., [2, 5]] = 0.0
        return self.solver.solve(G, X, target, self.U, log["q"], log["theta"],
                                 log["qdot"], log["tau_pd"], self.swing_mask,
                                 Qe,
                                 self.Qu_stage3 * self.qu3_scale if stage == 3 else self.Qu_diag,
                                 stage)

    def describe_reference(self):
        info = self.to_info
        extra = ""
        if "status" in info:
            extra = (f" [{info['status']}, {info['iterations']} it, dynamics residual "
                     f"{info['dynamics_residual']:.1e}]")
        return (f"TO: success={info['success']}{extra}, planned landing {np.round(info['landing'], 3)} "
                f"(goal {np.round(self.goal, 3)}), min foot clearance "
                f"{info['min_clearance']*100:.1f} cm, peak |f| {np.abs(self.u_ref).max():.0f} N, "
                f"peak |tau| per motor {np.round(np.abs(info['tau']).max(0) / 2, 1)} N·m")


def describe_result(r):
    line = (f"trial {r['trial']:2d} | stage {r['stage']} | landing miss {r['pos_err']*100:5.2f} cm | "
            f"pitch {np.degrees(r['theta_err']):4.1f} deg | min foot clearance "
            f"{r['clearance']*100:5.1f} cm" + ("  <-- feet hit the ground/box" if r["clearance"] < 0 else "")
            + ("  <-- FELL" if r.get("fell") else ""))
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
