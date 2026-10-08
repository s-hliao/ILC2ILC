import casadi as ca
from acados_template import AcadosModel,  AcadosOcp, AcadosOcpSolver
import numpy as np
import os
from pathlib import Path
from scipy.linalg import block_diag

from params import F110

def linear_model(x, u, p, params_car, exact = False):

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

    Cf, Cr, muf, mur, Cro = [p[i] for i in range(5)]
    # state: x, y, phi, vx, vy, omega, omega_w, current, delta   (10 states)
    omega_w, current, delta = x[6], x[7], x[8]

    eps = 0.1
    vx_dyn = ca.sqrt(x[3]**2 + eps)
    
    '''
    # --- slip angles (no guard) ---
    alphaf = delta - ca.atan2(x[5]*lf + x[4], vx_dyn)
    alphar = ca.atan2(x[5]*lr - x[4], vx_dyn)
    '''

    v0 = 1.0
    alphaf = delta - (x[4] + lf*x[5]) / v0
    alphar = (lr*x[5] - x[4]) / v0

    # --- front: lateral only ---
    Ffy = Cf*alphaf
    Ffx = 0

    # --- drive torque on rear wheel ---
    tau_drive = gear_ratio * 1.5 * pole_pairs * lam *current
    Frx = tau_drive/rw
    Fry = Cr*alphar
    

    # --- realized longitudinal chassis force drives load transfer ---
    Fx_chassis = Frx + Ffx*ca.cos(delta) - Ffy*ca.sin(delta)

    # --- dynamics ---
    dx0 = x[3]*ca.cos(x[2]) - x[4]*ca.sin(x[2])
    dx1 = x[3]*ca.sin(x[2]) + x[4]*ca.cos(x[2])
    dx2 = x[5]
    dx3 = (Frx + Ffx*ca.cos(delta) - Ffy*ca.sin(delta))/mass + x[4]*x[5]
    dx4 = (Fry + Ffy*ca.cos(delta) + Ffx*ca.sin(delta))/mass - x[3]*x[5]
    dx5 = (lf *(Ffy*ca.cos(delta) + Ffx*ca.sin(delta)) - lr*Fry)/Iz
    dx6 = dx3/rw
    dx7 = u[0]
    dx8 = u[1]

    xdot= ca.vertcat(
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
    
    return xdot


def nonlinear_model(x, u, p, params_car, exact = False):

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


    Cf, Cr, muf, mur, Cro = [p[i] for i in range(5)]
    # state: x, y, phi, vx, vy, omega, omega_w, current, delta   (10 states)
    omega_w, current, delta = x[6], x[7], x[8]
    

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

        kappa_r = (rw*omega_w- x[3]) / vx_dyn
        kappa_f = (rw*omega_w - vx_front) / vx_dyn_f

        # --- combined slips ---
        sigma_f = ca.sqrt(ca.tan(alphaf)**2 + kappa_f**2 + 1e-2)               # front free-rolling
        sigma_r = ca.sqrt(ca.tan(alphar)**2 + kappa_r**2 + 1e-2)
        
        sigmaf_max = 3*Ffmax/Cf
        sigmar_max = 3*Frmax/Cr


        def brush(C, s, Fmax):
            Cs = C*s
            return Cs - Cs**2/(3*Fmax) + Cs**3/(27*Fmax**2)

        Ff_total = ca.if_else(sigma_f < sigmaf_max, brush(Cf, sigma_f, Ffmax), Ffmax)
        Fr_total = ca.if_else(sigma_r < sigmar_max, brush(Cr, sigma_r, Frmax), Frmax)

        # --- front: lateral only ---
        Ffy = Ff_total*ca.tan(alphaf)/sigma_f
        Ffx = Ff_total*kappa_f       /sigma_f
        
        # --- rear: lateral and longitudinal share Fr_total via sigma_r ---
        v_eps = 0.5
        F_roll = Cro * Frz * ca.tanh(x[3] / v_eps)
        Frx = Fr_total*kappa_r/sigma_r - F_roll
        Fry =  Fr_total*ca.tan(alphar)/sigma_r

        # --- drive torque on rear wheel ---
        tau_drive = gear_ratio * 1.5 * pole_pairs * lam *current

        # --- realized longitudinal chassis force drives load transfer ---
        Fx_chassis = Frx + Ffx*ca.cos(delta) - Ffy*ca.sin(delta)

        # --- dynamics ---
        dx0 = x[3]*ca.cos(x[2]) - x[4]*ca.sin(x[2])
        dx1 = x[3]*ca.sin(x[2]) + x[4]*ca.cos(x[2])
        dx2 = x[5]
        dx3 = (Frx + Ffx*ca.cos(delta) - Ffy*ca.sin(delta))/mass + x[4]*x[5]
        dx4 = (Fry + Ffy*ca.cos(delta) + Ffx*ca.sin(delta))/mass - x[3]*x[5]
        dx5 = (lf *(Ffy*ca.cos(delta) + Ffx*ca.sin(delta)) - lr*Fry)/Iz
        dx6 = (tau_drive - Frx*rw - Ffx *rw)/(Iw + Im)
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
        sigma_f = ca.sqrt(ca.tan(alphaf)**2 + kappa_f**2 + 1e-9)               # front free-rolling
        sigma_r = ca.sqrt(ca.tan(alphar)**2 + kappa_r**2 + 1e-9)

        sigmaf_max = 3*Ffmax/Cf
        sigmar_max = 3*Frmax/Cr

        def brush(C, s, Fmax):
            Cs = C*s
            return Cs - Cs**2/(3*Fmax) + Cs**3/(27*Fmax**2)

        Ff_total = ca.if_else(sigma_f < sigmaf_max, brush(Cf, sigma_f, Ffmax), Ffmax)
        Fr_total = ca.if_else(sigma_r < sigmar_max, brush(Cr, sigma_r, Frmax), Frmax)

        Ffy = Ff_total*ca.tan(alphaf)/sigma_f
        Ffx = Ff_total*kappa_f       /sigma_f

        # --- rear: lateral and longitudinal share Fr_total via sigma_r ---
        v_eps = 0.5
        F_roll = Cro * Frz * ca.tanh(x[3] / v_eps)
        Frx = Fr_total*kappa_r/sigma_r - F_roll
        Fry =  Fr_total*ca.tan(alphar)/sigma_r

        # --- drive torque on rear wheel ---
        tau_drive = gear_ratio * 1.5 * pole_pairs *lam* current
        
        # --- realized longitudinal chassis force drives load transfer ---
        Fx_chassis = Frx + Ffx*ca.cos(delta) - Ffy*ca.sin(delta)

        # --- dynamics ---
        dx0 = x[3]*ca.cos(x[2]) - x[4]*ca.sin(x[2])
        dx1 = x[3]*ca.sin(x[2]) + x[4]*ca.cos(x[2])
        dx2 = x[5]
        dx3 = (Frx + Ffx*ca.cos(delta) - Ffy*ca.sin(delta))/mass + x[4]*x[5]
        dx4 = (Fry + Ffy*ca.cos(delta) + Ffx*ca.sin(delta))/mass - x[3]*x[5]
        dx5 = (lf *(Ffy*ca.cos(delta) + Ffx*ca.sin(delta)) - lr*Fry)/Iz
        dx6 = (tau_drive - Frx*rw - Ffx *rw)/(Iw + Im)
        dx7 = u[0]
        dx8 = u[1]


    xdot = ca.vertcat(
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
    
    return xdot


def blend_model(x, u, p, params_car, exact=False):
    """Shared continuous dynamics for NMPC and blend iLQR."""
    w = 0.5 * (1.0 + ca.tanh((x[3] - 1.0) / 0.2))
    return ((1.0 - w) * linear_model(x, u, p, params_car, exact)
            + w * nonlinear_model(x, u, p, params_car, exact))


def export_model(params_car, exact):
    model = AcadosModel()
    model.name = "f1tenthblend"

    x = ca.MX.sym('x', 9)   # state: x, y, phi, vx, vy, omega, omega_w, current, delta
    u = ca.MX.sym('u', 2)   # control rate: curent slew rate, steer rate
    p = ca.MX.sym('p', 5)
    x_ref = ca.MX.sym('x_ref', 6)

    model.f_expl_expr = blend_model(x, u, p, params_car, exact)
    model.x = x
    model.u = u
    model.p = ca.vertcat(p, x_ref)

    return model


def create_ocp(model, params_car, steps, horizon):
    ocp = AcadosOcp()
    ocp.model = model

    N = steps #steps
    Tf = horizon # total time horizon

    ocp.solver_options.tf = Tf #set solver settings
    
    nx = model.x.size()[0]  
    nu = model.u.size()[0]  # This is 2
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.solver_options.tf = Tf #set solver settings

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'


    # fig8
    # w_x = 40.0 #20  40  40
    # w_y = 40.0 # 20  40 40
    # w_xe = 40.0 # 20  40 0
    # w_ye = 40.0 # 20 40  0
    # w_theta = 0
    # w_vx = 10  #  10 10 

    # w_current = 0.01
    # w_steer = 0.01 # 0.1 0.01 0.8
    # w_slew = 0.0
    # w_steer_v = 0.1 # 0.01 0.1 0.8
    # w_omega = 0

    # blevel
    # w_x = 40.0 #20  40  40
    # w_y = 40.0 # 20  40 40
    # w_xe = 40.0 # 20  40 0
    # w_ye = 40.0 # 20 40  0
    # w_theta = 0
    # w_vx = 1.0  #  10 10 

    # w_current = 0.01
    # w_steer = 0.5 # 0.1 0.01 0.8
    # w_slew = 0.0
    # w_steer_v = 0.5 # 0.01 0.1 0.8
    # w_omega = 0.5

    # oval
    # w_x = 40.0 #20  40  40
    # w_y = 40.0 # 20  40 40
    # w_xe = 40.0 # 20  40 0
    # w_ye = 40.0 # 20 40  0
    # w_theta = 0
    # w_vx = 1.0 #  10 10 
    # w_omega = 0.0

    # w_current = 0.01
    # w_steer = 0.5 # 0.1 0.01 0.8
    # w_slew = 0.0
    # w_steer_v = 0.5 

    # switch
    # w_x = 40.0 #20  40  40
    # w_y = 40.0 # 20  40 40
    # w_xe = 40.0 # 20  40 0
    # w_ye = 40.0 # 20 40  0
    # w_theta = 0
    # w_vx = 0  #  10 10 

    # w_current = 0.01
    # w_steer = 0.8 # 0.1 0.01 0.8
    # w_slew = 0.0
    # w_steer_v = 0.8 # 0.01 0.1 0.8
    
    # FAST
    w_x = 40.0 #20  40  40
    w_y = 40.0 # 20  40 40
    w_xe = 0.0 # 20  40 0
    w_ye = 0.0 # 20 40  0
    w_theta = 0
    w_vx = 0.1  #  10 10 


    w_omega = 0.
    w_current = 0.01
    w_steer = 0.5 # 0.1 0.01 0.8
    w_slew = 0.0
    w_steer_v = 0.5 # 0.01 0.1 0.8

    # CRAZY
    # w_x = 40.0 #20  40  40
    # w_y = 40.0 # 20  40 40
    # w_xe = 0.0 # 20  40 0
    # w_ye = 0.0 # 20 40  0
    # w_theta = 0
    # w_vx = 1.0  #  10 10 


    # w_omega = 1.0
    # w_current = 0.01
    # w_steer = 0.1 # 0.1 0.01 0.8
    # w_slew = 0.0
    # w_steer_v = 0.1 # 0.01 0.1 0.8

    # w_x = 40.0 #20  40  40
    # w_y = 40.0 # 20  40 40
    # w_xe = 40.0 # 20  40 0
    # w_ye = 40.0 # 20 40  0
    # w_theta = 0
    # w_vx = 10.0  #  10 10 

    # w_omega = 0.1
    # w_current = 0.01
    # w_steer = 0.1 # 0.1 0.01 0.8
    # w_slew = 0.0
    # w_steer_v = 0.1 # 0.01 0.1 0.8
    
    
    
      
    Q_flat = [w_x, w_y, w_theta, w_vx, 0.0, w_omega,  w_current, w_steer]

    # vx, vy, omega
    R_flat = [w_slew, w_steer_v]

    Q = np.diag(Q_flat)
    R = np.diag(R_flat)
    Qf = np.diag([w_xe, w_ye, 0, 0, 0, 0, 0, 0])

    ocp.cost.W = np.diag(np.concatenate((Q_flat, R_flat)))
    ocp.cost.W_e = Qf

    x_ref = model.p[-6:]  # last 6 parameters
    x = model.x
    u = model.u

    ny = 10 # running dimensions
    ny_e = 8 #terminal dimension

    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e
    
    ocp.cost.yref = np.zeros(ny) # running objective function reference
    ocp.cost.yref_e = np.zeros(ny_e) # terminal objective function reference

    yaw_err = x[2] - x_ref[2]
    yaw_err_wrapped = ca.atan2(ca.sin(yaw_err), ca.cos(yaw_err))


    ocp.model.cost_y_expr = ca.vertcat(
        x[0] - x_ref[0],   # x
        x[1] - x_ref[1],   # y
        yaw_err_wrapped,    # yaw (wrapped)
        x[3] - x_ref[3],   # vx
        x[4] - x_ref[4],   # vy
        x[5] - x_ref[5],   # omega
        x[7],   # current
        x[8],   # steer
        u
    )
    ocp.model.cost_y_expr_e = ca.vertcat(
        x[0] - x_ref[0],   # x
        x[1] - x_ref[1],   # y
        yaw_err_wrapped,    # yaw (wrapped)
        x[3] - x_ref[3],   # vx
        x[4] - x_ref[4],   # vy
        x[5] - x_ref[5],   # omega
        x[7],   # current
        x[8],   # steer
    )
    
    ocp.model.p = model.p  # Combine with existing parameters
    ocp.dims.np = model.p.size()[0]
    ocp.parameter_values = np.zeros((ocp.dims.np, 1))

    ocp.constraints.idxbx = np.array([3, 4,5, 7,8])
    ocp.constraints.lbx = np.array([-0.5, 
                                -4,
                                - 2*np.pi,
                                -25, 
                               params_car['min_steer']])

    ocp.constraints.ubx = np.array([params_car['max_v'], 
                                    4,
                                    2* np.pi,
                                    50, 
                                    params_car['max_steer']])
    
    ocp.constraints.lbu = np.array([-300, -params_car['max_steer_vel']])
    ocp.constraints.ubu = np.array([300, params_car['max_steer_vel']])
    ocp.constraints.idxbu = np.array([0, 1]) # 0 is slew rate, 1 is steer_vel

    # slack on constraints
    w_slack = 100.0       # L2 slack penalty (quadratic)
    w_slack_l1 = 10.0    # L1 slack penalty (linear)
    
    nsbx = 5
    ocp.dims.nsbx = nsbx
    ocp.constraints.idxsbx = np.arange(nsbx)   # slack all idxbx entries

    ocp.cost.zl  = w_slack_l1 * np.ones(nsbx)  # lower L1 weight
    ocp.cost.zu  = w_slack_l1 * np.ones(nsbx)  # upper L1 weight
    ocp.cost.Zl  = w_slack    * np.ones(nsbx)  # lower L2 weight
    ocp.cost.Zu  = w_slack    * np.ones(nsbx)  # upper L2 weight
    #####################################
    
    ocp.constraints.idxbx_0 = np.arange(9) # IMPORTANT FOR RUNTIME
    ocp.constraints.lbx_0 = np.zeros(9)           # placeholder
    ocp.constraints.ubx_0 = np.zeros(9)           # placeholder

    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = ocp.constraints.idxbx
    
    # slack on terminal constraints
    nsbx_e = 5
    ocp.dims.nsbx_e = nsbx_e
    ocp.constraints.idxsbx_e = np.arange(nsbx_e)

    ocp.cost.zl_e  = w_slack_l1 * np.ones(nsbx_e)
    ocp.cost.zu_e  = w_slack_l1 * np.ones(nsbx_e)
    ocp.cost.Zl_e  = w_slack    * np.ones(nsbx_e)
    ocp.cost.Zu_e  = w_slack    * np.ones(nsbx_e)
    
    ###############################################

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'

    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    
    # DROPPED FROM 10 to 2 (This makes the solver ~5x faster)
    ocp.solver_options.sim_method_num_steps = 5

    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 20  # 2-3 iterations
    # ocp.solver_options.globalization = 'FIXED_STEP'
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_warm_start = 1    
    
    # STABILITY FIXES
    ocp.solver_options.levenberg_marquardt = 1e-4  # Increased damping
    ocp.solver_options.regularize_hessian = 1e-6   # Prevent singular Hessian crashes
    # ocp.solver_options.qp_solver_cond_N = N        # Enable full condensing for small horizons
    ocp.solver_options.hpipm_mode = 'SPEED'       # Failsafe against stiff Pacejka matrices

    return ocp


def get_solver_directory(solver_config = "default"):
    # package_dir = Path(get_package_share_directory('llampc'))
    package_dir = Path(__file__).parent.resolve()
    solvers_dir = package_dir / 'solvers' / solver_config

    solvers_dir.mkdir(parents=True, exist_ok=True)
    
    return solvers_dir


def setup_mpc(steps, horizon, json_file='f1tenth_acados_ocp.json', solver_config ="default", build=True, params_car=F110):
    
    solver_dir = get_solver_directory(solver_config)
    full_json_path = solver_dir / json_file
    
    original_cwd = os.getcwd()
    p_car = params_car()
    try:
        os.chdir(solver_dir)
        print(f"Generating solver in: {solver_dir}")
        
        f1tenth_model = export_model(p_car, exact = False)
        ocp = create_ocp(f1tenth_model, p_car, steps, horizon)
        solver = AcadosOcpSolver(ocp, json_file=json_file, build=build)
        
        print(f"Solver generated successfully: {full_json_path}")
        return solver
    except Exception as e:
        print(f"Error generating solver: {e}")
        raise
    finally:
        os.chdir(original_cwd) 

def _get_tire_params(p_car):
    """
    Pull [Cf, Cr, muf, mur, Cro] to match the unpacking order used in
    export_model: `Cf, Cr, muf, mur, Cro = [p[i] for i in range(5)]`.
    Uses the same dict-style access (`params_car['key']`) as the rest
    of this file, with fallbacks in case a key is missing from F110.
    """
    def _get(key, default):
        try:
            return p_car[key]
        except (KeyError, TypeError):
            return default

    Cf  = _get('Cf', 5.0)
    Cr  = _get('Cr', 5.0)
    muf = _get('muf', 1.0)
    mur = _get('mur', 1.0)
    Cro = _get('Cro', 0.01)
    return np.array([Cf, Cr, muf, mur, Cro])

def build_smoke_reference(N, dt, v_target=2.0):
    """
    Straight-line accelerating reference.

    vx: 0 -> v_target
    y = 0
    phi = 0
    vy = 0
    omega = 0
    """

    t = np.arange(N + 1) * dt
    T = N * dt

    a_ref = v_target / T

    vx_ref = a_ref * t
    x_ref = 0.5 * a_ref * t**2

    y_ref = np.zeros(N + 1)
    phi_ref = np.zeros(N + 1)
    vy_ref = np.zeros(N + 1)
    omega_ref = np.zeros(N + 1)

    return np.stack(
        [
            x_ref,
            y_ref,
            phi_ref,
            vx_ref,
            vy_ref,
            omega_ref
        ],
        axis=1
    )


def main():

    # --------------------------------------------------
    # Solver setup
    # --------------------------------------------------

    dt = 0.05
    horizon = 2.0
    N = int(horizon / dt)

    solver = setup_mpc(
        steps=N,
        horizon=horizon,
        solver_config="blend_smoke",
        build=True
    )

    p_car = F110()

    tire_params = _get_tire_params(p_car)

    # --------------------------------------------------
    # Reference
    #
    # vx goes:
    # 0 -> 2 m/s
    #
    # so it crosses the 1 m/s blending region
    # --------------------------------------------------

    ref_traj = build_smoke_reference(
        N,
        dt,
        v_target=5.0
    )

    # --------------------------------------------------
    # Stage parameters
    #
    # model.p =
    # [Cf, Cr, muf, mur, Cro, x_ref(6)]
    # --------------------------------------------------

    for k in range(N + 1):

        p_k = np.concatenate(
            [
                tire_params,
                ref_traj[k]
            ]
        )

        solver.set(k, "p", p_k)

    # --------------------------------------------------
    # Initial state
    #
    # [x, y, phi, vx, vy, omega,
    #  omega_w, current, delta]
    # --------------------------------------------------

    x0 = np.zeros(9)

    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    # --------------------------------------------------
    # Warm start
    # --------------------------------------------------

    for k in range(N + 1):

        x_guess = np.zeros(9)

        x_guess[0:6] = ref_traj[k]

        # no-slip wheel speed guess
        x_guess[6] = (
            ref_traj[k, 3]
            / p_car['rw']
        )

        solver.set(
            k,
            "x",
            x_guess
        )

    for k in range(N):

        solver.set(
            k,
            "u",
            np.zeros(2)
        )

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------

    status = solver.solve()

    print("\n==============================")
    print("      BLEND SMOKE TEST")
    print("==============================")

    print(f"Solver status: {status}")

    if status != 0:

        print("\nSolver failed.")
        solver.print_statistics()
        return

    print("Solve succeeded.")

    # --------------------------------------------------
    # Get result
    # --------------------------------------------------

    X = np.array(
        [
            solver.get(k, "x")
            for k in range(N + 1)
        ]
    )

    U = np.array(
        [
            solver.get(k, "u")
            for k in range(N)
        ]
    )

    cost = solver.get_cost()

    print(f"Cost: {cost:.4f}")

    # --------------------------------------------------
    # Basic numerical checks
    # --------------------------------------------------

    print(
        "States finite:",
        np.all(np.isfinite(X))
    )

    print(
        "Controls finite:",
        np.all(np.isfinite(U))
    )

    print(
        f"vx range: "
        f"{X[:, 3].min():.3f} "
        f"to "
        f"{X[:, 3].max():.3f} m/s"
    )

    print(
        f"max |current slew|: "
        f"{np.max(np.abs(U[:, 0])):.3f}"
    )

    print(
        f"max |steer rate|: "
        f"{np.max(np.abs(U[:, 1])):.3f}"
    )

    # --------------------------------------------------
    # Print blending behavior
    # --------------------------------------------------

    print("\n")
    print(
        " k     t       x       vx       w      "
        "omega_w    current"
    )

    print(
        "-------------------------------------------------------"
    )

    for k in range(0, N + 1, 4):

        vx = X[k, 3]

        w = 0.5 * (
            1.0
            + np.tanh(
                (vx - 1.0) / 0.2
            )
        )

        print(
            f"{k:2d}  "
            f"{k*dt:5.2f}  "
            f"{X[k,0]:7.3f}  "
            f"{vx:7.3f}  "
            f"{w:7.3f}  "
            f"{X[k,6]:8.3f}  "
            f"{X[k,7]:8.3f}"
        )

    # --------------------------------------------------
    # Find first point above switch speed
    # --------------------------------------------------

    above_switch = np.where(
        X[:, 3] >= 1.0
    )[0]

    if len(above_switch) > 0:

        k_switch = above_switch[0]

        print(
            "\nFirst state above 1 m/s:"
        )

        print(
            f"k = {k_switch}, "
            f"t = {k_switch * dt:.2f} s, "
            f"vx = {X[k_switch,3]:.3f} m/s"
        )

    else:

        print(
            "\nWARNING: optimized trajectory "
            "never reached 1 m/s."
        )


if __name__ == "__main__":
    main()
