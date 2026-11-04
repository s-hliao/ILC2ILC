"""fullbody_lin.py (2026-11-01): the planar full-body model with planted feet, linearized along a measured trial; a copy of the class in the fixbox overlay's ilc_gen.py, for deploy.py --jac-at fb."""
import casadi as ca
import numpy as np


class FullBodyLin:
    """(2026-11-01) The planar full-body model with its planted feet, linearized along the MEASURED trial: the lifted G of the ILC in place of the SRB's. The SRB has no
    legs, and measured force changes do not act on the Go1 as it predicts (a rear +20 N over 0.3 s: +0.002 m/s at take-off, SRB +0.47; this model +0.006).
    State z = (s, sd), 14; x = (c, theta, cd, omega), 6. Contact to Ndc (front), Nc (rear), then free flight: the CoM ballistic, omega the body's momentum / inertia."""

    def __init__(self, fb, Ndc, Nc, N, dt, kp_c, kd_c, kp_j=30.0, kd_j=1.0, nsub=5):
        self.fb, self.Ndc, self.Nc, self.N, self.dt, self.nsub = fb, Ndc, Nc, N, dt, nsub
        self.gains = (float(kp_c), float(kd_c), float(kp_j), float(kd_j))
        ns = fb.ns
        s, sd, u, front = ca.MX.sym("s", ns), ca.MX.sym("sd", ns), ca.MX.sym("u", 4), ca.MX.sym("front")
        z = ca.vertcat(s, sd)
        tau_to, u_ref, q_d, qd_d, kp, kd = (ca.MX.sym(n, 4) for n in ("tau_to", "u_ref", "q_d", "qd_d", "kp", "kd"))
        H, h, J = fb.H(s), fb.h(s, sd), fb.Jc(s)
        Jd_sd = ca.jtimes(ca.mtimes(J, sd), s, sd)
        c = ca.DM(fb.damping)
        tau = tau_to - ca.mtimes(J[:, 3:].T, u - u_ref) + 2.0 * kp * (q_d - s[3:]) + 2.0 * kd * (qd_d - sd[3:]) - c * sd[3:]
        rhs = ca.vertcat(ca.MX.zeros(3), tau) - h
        w = ca.vertcat(front, front, 1, 1)
        K = ca.blockcat([[H, J.T], [ca.diag(w) @ J, ca.diag(1 - w)]])
        sdd = ca.solve(K + 1e-9 * ca.DM.eye(ns + 4), ca.vertcat(rhs, -w * Jd_sd), "qr")[:ns]
        f = ca.vertcat(sd, sdd)
        self.jac = ca.Function("fb_jac", [s, sd, u, tau_to, u_ref, q_d, qd_d, kp, kd, front], [ca.jacobian(f, z), ca.jacobian(f, u)])
        com, Jcom = fb.com(s), fb.Jcom(s)
        v = ca.mtimes(Jcom, sd)
        ptheta = ca.mtimes(H[2, :], sd)
        Lc = ptheta - fb.total_mass * ((com[0] - s[0]) * v[1] - (com[1] - s[1]) * v[0])
        om = Lc / fb.pitch_inertia(s)
        self.C = ca.Function("fb_C", [s, sd], [ca.jacobian(ca.vertcat(com, s[2], v, sd[2]), z), ca.jacobian(om, z)])

    def state(self, x, q, qd):
        """z from x = (c, theta, cd, omega) and the joints: the base from the CoM."""
        fb = self.fb
        s = np.concatenate([[0.0, 0.0, x[2]], q])
        s[:2] = x[:2] - np.asarray(fb.com(s)).ravel()
        Jc_ = np.asarray(fb.Jcom(s))
        sd = np.concatenate([[0.0, 0.0, x[5]], qd])
        sd[:2] = x[3:5] - Jc_[:, 2:] @ sd[2:]
        return np.concatenate([s, sd])

    def lifted(self, X, U, u_ref, q, qd, tau, q_ref, qd_ref, A_srb):
        """G (N * 6, Nc * 4), time-major; rows to Nc from this model, past it the SRB's free flight (eq. 12)."""
        Nc, N, nu = self.Nc, self.N, 4
        kpc, kdc, kpj, kdj = self.gains
        G = np.zeros((N * 6, Nc * nu))
        Z = np.zeros((14, Nc * nu)); h = self.dt / self.nsub
        for k in range(Nc):
            z = self.state(X[k], q[k], qd[k])
            fr = 1.0 if k < self.Ndc else 0.0
            kp = np.array([kpc if fr else kpj] * 2 + [kpc] * 2); kd = np.array([kdc if fr else kdj] * 2 + [kdc] * 2)
            Ac, Bc = self.jac(z[:7], z[7:], U[k], tau[k], u_ref[k], q_ref[k], qd_ref[k], kp, kd, fr)
            Ac, Bc = np.asarray(Ac), np.asarray(Bc)
            Ad = np.eye(14); Bd = np.zeros((14, nu))
            for _ in range(self.nsub):
                Bd = Bd + h * Ad @ Bc; Ad = Ad @ (np.eye(14) + h * Ac)
            Z = Ad @ Z; Z[:, k * nu:(k + 1) * nu] += Bd
            if k + 1 < Nc:
                z1 = self.state(X[k + 1], q[k + 1], qd[k + 1])
                C = np.asarray(self.C(z1[:7], z1[7:])[0]); G[(k * 6):(k + 1) * 6] = C @ Z
        zN = self.state(X[Nc], q[-1], qd[-1])
        Cx, Co = self.C(zN[:7], zN[7:]); Cx, Co = np.asarray(Cx), np.asarray(Co)
        xg = Cx @ Z; xg[5] = Co @ Z
        G[(Nc - 1) * 6:Nc * 6] = xg
        for t in range(Nc, N):
            xg = A_srb @ xg; G[t * 6:(t + 1) * 6] = xg
        return G
