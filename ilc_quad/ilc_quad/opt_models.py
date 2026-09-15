"""
Iterative Learning Control (ILC) for quadruped target jumping, in CasADi + IPOPT.

Implements the framework described in the paper:
  - Planar single-rigid-body (SRB) dynamics (eq. 24)
  - LTV error model via trial-to-trial linearization (eq. 8-12)
  - Lifted sensitivity matrix G_k (eq. 19)
  - Three-stage goal-priority QP (eq. 16, 20-21), solved per trial with IPOPT
  - Motor dynamic constraints (MDC), eq. (1)-(4), (17)-(18)

This is a reference / scaffolding implementation. Replace the leg
kinematics (contact points, Jacobian) and the trial-execution callback
with your robot's real kinematics / hardware or simulator interface.
"""

import numpy as np
import casadi as ca


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
# 2) Build the lifted sensitivity matrix G_k  (eq. 19-20)
# --------------------------------------------------------------------------
def build_lifted_G(A_list, B_list, N, Nc, nx, nu):
    """
    G[m, n] block (m = 1..N, n = 1..Nc), 0-indexed here as m=0..N-1, n=0..Nc-1:
        G_{m,n} = B_{m}                      if m == n and m < Nc
        G_{m,n} = A_{m-1} A_{m-2} ... A_{n+1} B_n   if m > n
        G_{m,n} = 0                          otherwise (m < n, or m>=Nc & m==n handled by m>n case)
    Flight phase (m >= Nc) still accumulates the full A-chain product,
    matching eq. (12): no new control after Nc, only propagation.
    """
    G = np.zeros((N * nx, Nc * nu))
    # precompute cumulative A products: cumA[m] = A_{m-1}...A_{0} (identity if m==0)
    cumA = [np.eye(nx)]
    for t in range(N):
        cumA.append(A_list[t] @ cumA[-1]) if t < len(A_list) else cumA.append(cumA[-1])

    for m in range(N):
        for n in range(Nc):
            if m == n and m < Nc:
                block = B_list[n]
            elif m > n:
                # product A_{m-1} ... A_{n+1}  =  cumA[m] @ inv-style chain
                # build directly to avoid inverses:
                prod = np.eye(nx)
                for j in range(n + 1, m):
                    prod = A_list[j] @ prod
                block = prod @ B_list[n]
            else:
                block = np.zeros((nx, nu))
            G[m * nx:(m + 1) * nx, n * nu:(n + 1) * nu] = block
    return G


# --------------------------------------------------------------------------
# 3) Three-stage ILC QP solved with CasADi + IPOPT  (eq. 16, 20-21)
# --------------------------------------------------------------------------
class ILCStageSolver:
    def __init__(self, N, Nc, Ndc, nx, nu, mdc_params, torque_map_fn):
        """
        mdc_params: dict with rho, sigma, Vmax, Vmin, tau_max, fmin, fmax, mu
        torque_map_fn(u, qdot_t) -> T_t (numeric Jacobian^T*R^T map, contact force -> joint torque)
                                     supplied per your robot's kinematics.
        """
        self.N, self.Nc, self.Ndc = N, Nc, Ndc
        self.nx, self.nu = nx, nu
        self.p = mdc_params
        self.torque_map_fn = torque_map_fn

    def stage_rows(self, stage):
        """Row (time-step) indices included in the cost for each stage."""
        if stage == 1:
            return list(range(0, self.Nc))                 # contact-priority
        elif stage == 2:
            return list(range(self.Ndc, self.N))            # hybrid (rear-contact + flight)
        elif stage == 3:
            return [self.N - 1]                              # goal-priority: final sample only
        else:
            raise ValueError("stage must be 1, 2 or 3")

    def solve(self, e_k, G, u_k, qdot_k, swing_mask, Qe_diag, Qu_diag, stage):
        """
        e_k        : (N, nx) actual trajectory error for the trial just executed
        G          : lifted matrix from build_lifted_G
        u_k        : (Nc, nu) applied control this trial
        qdot_k     : (Nc,) joint velocities this trial (for MDC), per-joint list acceptable
        swing_mask : (Nc, nu) boolean, True where that force component must stay 0 (Ephase, eq 16b)
        Qe_diag    : (nx,) weights on trajectory error
        Qu_diag    : (nu,) weights on control offset
        stage      : 1, 2, or 3
        returns delta_u_star: (Nc, nu)
        """
        nx, nu, Nc, N = self.nx, self.nu, self.Nc, self.N
        rows = self.stage_rows(stage)

        opti = ca.Opti()
        du = opti.variable(Nc, nu)                # decision var: delta_u_{k,t}
        du_flat = ca.reshape(du.T, Nc * nu, 1)     # match G's column ordering

        e_flat = e_k.reshape(-1, 1)                # (N*nx, 1)
        e_next_flat = e_flat - G @ du_flat          # eq. (19): e_{k+1} = e_k - G delta_u

        # ---- objective: sum over stage rows of e^T Qe e + delta_u^T Qu delta_u  (eq. 16/20) ----
        cost = 0
        for t in rows:
            et = e_next_flat[t * nx:(t + 1) * nx]
            cost += et.T @ np.diag(Qe_diag) @ et
        for t in range(Nc):
            dut = du[t, :].T
            cost += dut.T @ np.diag(Qu_diag) @ dut
        opti.minimize(cost)

        # ---- constraints ----
        p = self.p
        for t in range(Nc):
            u_next = ca.reshape(u_k[t, :], nu, 1) + du[t, :].T

            # Ephase (16b): zero force on swing legs
            for j in range(nu):
                if swing_mask[t, j]:
                    opti.subject_to(u_next[j] == 0)

            # Icontact (friction cone + force limits), per foot pair (fx, fz)
            # -- only enforced for feet in stance this timestep; swing feet are
            #    already pinned to zero by the Ephase equality constraint above.
            for foot in range(nu // 2):
                if swing_mask[t, 2 * foot] or swing_mask[t, 2 * foot + 1]:
                    continue
                fx = u_next[2 * foot]
                fz = u_next[2 * foot + 1]
                opti.subject_to(fz >= p["fmin"])
                opti.subject_to(fz <= p["fmax"])
                opti.subject_to(fx <= p["mu"] * fz)
                opti.subject_to(-fx <= p["mu"] * fz)

            # Map contact force -> joint torque:  tau = T_t (u + delta_u)   (eq. 14)
            T_t = self.torque_map_fn(u_k[t, :], qdot_k[t])   # numeric matrix, shape (n_joints, nu)
            tau_t = T_t @ u_next

            # Isat (torque saturation)
            opti.subject_to(opti.bounded(-p["tau_max"], tau_t, p["tau_max"]))

            # Imdc (motor dynamic constraint), eq. (18):
            #   (Vmin - sigma*qdot)/rho <= tau <= (Vmax - sigma*qdot)/rho
            qd = qdot_k[t]
            lb = (p["Vmin"] - p["sigma"] * qd) / p["rho"]
            ub = (p["Vmax"] - p["sigma"] * qd) / p["rho"]
            opti.subject_to(opti.bounded(lb, tau_t, ub))

        opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes"})
        sol = opti.solve()
        return sol.value(du)


# --------------------------------------------------------------------------
# 4) Trial loop (offline sim demo standing in for hardware/simulator I/O)
# --------------------------------------------------------------------------
def run_ilc_demo():
    nx, nu = 6, 4
    Ndc, Nsc, Nfl = 10, 10, 20
    Nc = Ndc + Nsc
    N = Nc + Nfl
    dt = 0.01

    srb = SRBModel(mass=12.0, inertia=0.3)

    # target state at landing (final row of reference trajectory)
    x_target = np.array([0.60, 0.0, 0.0, 0.0, 0.0, 0.0])  # 60 cm forward jump, flat landing

    # crude MDC / actuator params (Unitree-A1-like, see Table III)
    mdc_params = dict(rho=0.35, sigma=0.02, Vmax=21.5, Vmin=-21.5,
                       tau_max=33.5, fmin=5.0, fmax=200.0, mu=0.6)

    def torque_map_fn(u, qdot_t):
        # placeholder: replace with J(q)^T R^T for your robot (eq. 14).
        # here: identity-like 2-joint-per-foot mapping for illustration.
        return np.eye(4) * 0.05

    solver = ILCStageSolver(N, Nc, Ndc, nx, nu, mdc_params, torque_map_fn)

    # initial control guess from trajectory optimization (stand-in: simple force profile)
    U = np.zeros((Nc, nu))
    U[:Ndc, [1, 3]] = 120.0     # vertical push, both feet, during all-leg contact
    U[Ndc:, 3] = 100.0          # rear foot only during rear-contact phase

    swing_mask = np.zeros((Nc, nu), dtype=bool)
    swing_mask[Ndc:, 0:2] = True   # front foot is swing during rear-contact phase

    R1 = [np.array([0.15, -0.3])] * Nc + [np.array([0.15, -0.3])] * Nfl
    R2 = [np.array([-0.15, -0.3])] * Nc + [np.array([-0.15, -0.3])] * Nfl
    qdot_k = [0.0] * Nc  # placeholder joint velocities for MDC

    x0 = np.zeros(nx)
    Qe_diag = np.array([1.0, 3.0, 3.0, 0.01, 0.01, 0.01])
    Qu_diag = 1e-5 * np.ones(nu)

    stage_schedule = [1] * 5 + [2] * 5 + [3] * 15   # stage per trial index, eq. schedule in paper

    for k, stage in enumerate(stage_schedule):
        U_full = np.vstack([U, np.zeros((Nfl, nu))])
        X = srb.simulate(x0, U_full, R1, R2, dt)

        x_ref = np.tile(x_target, (N, 1))
        e_k = x_ref - X[1:]  # error per eq. (11), shape (N, nx)

        A_list, B_list = srb.linearize_along_trial(X, U_full, R1, R2, dt)
        G = build_lifted_G(A_list, B_list, N, Nc, nx, nu)

        du = solver.solve(e_k, G, U, qdot_k, swing_mask, Qe_diag, Qu_diag, stage)
        U = U + du  # eq. (15): u_{k+1} = delta_u* + u_k

        final_err = np.linalg.norm(e_k[-1, :3])
        print(f"trial {k+1:2d} | stage {stage} | final pos/ori error norm: {final_err:.4f}")

        if stage == 3 and final_err < 0.01:
            print("Target reached.")
            break


if __name__ == "__main__":
    run_ilc_demo()