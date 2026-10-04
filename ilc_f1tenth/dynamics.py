import casadi as ca
import numpy as np

from params import F110


def continous_dynamics_model(x, u, p, params_car, exact = False):
    # x = ca.MX.sym('x', 9)   # state: x, y, phi, vx, vy, omega, omega_w, current, delta
    # u = ca.MX.sym('u', 2)   # control rate: curent slew rate, steer rate
    # p = ca.MX.sym('p', 5)   # parameters vector: Cf, Cr, muf, mur, Cro

    Cf, Cr, muf, mur, Cro = [p[i] for i in range(5)]

    omega_w = x[6]
    current = x[7]
    delta = x[8]

    mass = params_car['mass']
    Iz   = params_car['Iz']
    lf   = params_car['lf']
    lr   = params_car['lr']
    rw   = params_car['rw']     # rear wheel radius [m]
    Iw   = params_car['Iw']     # rear driveline rotational inertia [kg m^2]
    Im = params_car['Im']
    pole_pairs  = params_car['pole_pairs']
    gear_ratio  = params_car['gear_ratio']
    lam = params_car['lambda']
    g    = 9.81
    hcg, L = params_car['h'], lf + lr

    if not exact:
        eps = 0.1
        vx_dyn = ca.sqrt(x[3]**2 + eps)

        Ffz = mass*g*lr/L
        Frz = mass*g*lf/L
        Ffmax = muf*Ffz
        Frmax = mur*Frz          # coupling lives in sigma_r, no friction-circle derate

        # --- slip angles (no guard) ---
        alphaf = delta - ca.atan2(x[5]*lf + x[4], vx_dyn)
        alphar = ca.atan2(x[5]*lr - x[4], vx_dyn)

        vx_front = x[3]*ca.cos(delta) + (x[4] + lf*x[5])*ca.sin(delta)
        vx_dyn_f = ca.sqrt(vx_front**2 + eps)

        kappa_r = (rw*omega_w - x[3]) / vx_dyn
        kappa_f = (rw*omega_w - vx_front) / vx_dyn_f

        # --- combined slips ---
        sigma_f = ca.sqrt(ca.tan(alphaf)**2 + kappa_f**2 + 1e-2)
        sigma_r = ca.sqrt(ca.tan(alphar)**2 + kappa_r**2 + 1e-2)

        sigmaf_max = 3*Ffmax/Cf
        sigmar_max = 3*Frmax/Cr


        def brush(C, s, Fmax):
            Cs = C*s
            return Cs - Cs**2/(3*Fmax) + Cs**3/(27*Fmax**2)


        Ff_total = ca.if_else(
            sigma_f < sigmaf_max,
            brush(Cf, sigma_f, Ffmax),
            Ffmax
        )

        Fr_total = ca.if_else(
            sigma_r < sigmar_max,
            brush(Cr, sigma_r, Frmax),
            Frmax
        )

        # --- front: lateral only ---
        Ffy = Ff_total*ca.tan(alphaf)/sigma_f
        Ffx = Ff_total*kappa_f/sigma_f

        # --- rear: lateral and longitudinal share Fr_total via sigma_r ---
        v_eps = 0.5
        F_roll = Cro * Frz * ca.tanh(x[3] / v_eps)

        Frx = Fr_total*kappa_r/sigma_r - F_roll
        Fry = Fr_total*ca.tan(alphar)/sigma_r

        # --- drive torque on rear wheel ---
        tau_drive = gear_ratio * 1.5 * pole_pairs * lam * current

        # --- dynamics ---
        dx0 = x[3]*ca.cos(x[2]) - x[4]*ca.sin(x[2])
        dx1 = x[3]*ca.sin(x[2]) + x[4]*ca.cos(x[2])
        dx2 = x[5]

        dx3 = (
            Frx
            + Ffx*ca.cos(delta)
            - Ffy*ca.sin(delta)
        )/mass + x[4]*x[5]

        dx4 = (
            Fry
            + Ffy*ca.cos(delta)
            + Ffx*ca.sin(delta)
        )/mass - x[3]*x[5]

        dx5 = (
            lf*(Ffy*ca.cos(delta) + Ffx*ca.sin(delta))
            - lr*Fry
        )/Iz

        dx6 = (
            tau_drive
            - Frx*rw
            - Ffx*rw
        )/(Iw + Im)

        dx7 = u[0]
        dx8 = u[1]


    else:
        Ffz = mass*g*lr/L
        Frz = mass*g*lf/L
        Ffmax = muf*Ffz
        Frmax = mur*Frz          # coupling lives in sigma_r, no friction-circle derate

        # --- slip angles (no guard) ---
        alphaf = delta - ca.atan2(x[5]*lf + x[4], x[3])
        alphar = ca.atan2(x[5]*lr - x[4], x[3])

        # --- rear longitudinal slip ratio from wheel speed (no guard) ---
        kappa_r = (rw*omega_w - x[3]) / x[3]

        vx_front = x[3]*ca.cos(delta) + (x[4] + lf*x[5])*ca.sin(delta)
        kappa_f = (rw*omega_w - vx_front) / vx_front

        # --- combined slips ---
        sigma_f = ca.sqrt(ca.tan(alphaf)**2 + kappa_f**2 + 1e-9)
        sigma_r = ca.sqrt(ca.tan(alphar)**2 + kappa_r**2 + 1e-9)

        sigmaf_max = 3*Ffmax/Cf
        sigmar_max = 3*Frmax/Cr


        def brush(C, s, Fmax):
            Cs = C*s
            return Cs - Cs**2/(3*Fmax) + Cs**3/(27*Fmax**2)


        Ff_total = ca.if_else(
            sigma_f < sigmaf_max,
            brush(Cf, sigma_f, Ffmax),
            Ffmax
        )

        Fr_total = ca.if_else(
            sigma_r < sigmar_max,
            brush(Cr, sigma_r, Frmax),
            Frmax
        )

        Ffy = Ff_total*ca.tan(alphaf)/sigma_f
        Ffx = Ff_total*kappa_f/sigma_f

        # --- rear: lateral and longitudinal share Fr_total via sigma_r ---
        v_eps = 0.5
        F_roll = Cro * Frz * ca.tanh(x[3] / v_eps)

        Frx = Fr_total*kappa_r/sigma_r - F_roll
        Fry = Fr_total*ca.tan(alphar)/sigma_r

        # --- drive torque on rear wheel ---
        tau_drive = gear_ratio * 1.5 * pole_pairs * lam * current

        # --- dynamics ---
        dx0 = x[3]*ca.cos(x[2]) - x[4]*ca.sin(x[2])
        dx1 = x[3]*ca.sin(x[2]) + x[4]*ca.cos(x[2])
        dx2 = x[5]

        dx3 = (
            Frx
            + Ffx*ca.cos(delta)
            - Ffy*ca.sin(delta)
        )/mass + x[4]*x[5]

        dx4 = (
            Fry
            + Ffy*ca.cos(delta)
            + Ffx*ca.sin(delta)
        )/mass - x[3]*x[5]

        dx5 = (
            lf*(Ffy*ca.cos(delta) + Ffx*ca.sin(delta))
            - lr*Fry
        )/Iz

        dx6 = (
            tau_drive
            - Frx*rw
            - Ffx*rw
        )/(Iw + Im)

        dx7 = u[0]
        dx8 = u[1]


    x_derivative = ca.vertcat(
        dx0,
        dx1,
        dx2,
        dx3,
        dx4,
        dx5,
        dx6,
        dx7,
        dx8
    )

    return x_derivative


def rk4(x, u, p, params_car, exact, dt):
    k1 = continous_dynamics_model(
        x,
        u,
        p,
        params_car,
        exact
    )

    k2 = continous_dynamics_model(
        x + dt/2 * k1,
        u,
        p,
        params_car,
        exact
    )

    k3 = continous_dynamics_model(
        x + dt/2 * k2,
        u,
        p,
        params_car,
        exact
    )

    k4 = continous_dynamics_model(
        x + dt*k3,
        u,
        p,
        params_car,
        exact
    )

    return x + dt/6 * (
        k1
        + 2*k2
        + 2*k3
        + k4
    )


def discrete_dynamics_model(x, u, p, params_car, exact, dt):
    return rk4(
        x,
        u,
        p,
        params_car,
        exact,
        dt
    )