"""iLQR backward and forward passes using the shared NMPC blend dynamics."""

import numpy as np
import casadi as ca
from blend_solver import blend_model

Q = np.diag([10, 10, 5, 2, 0.5, 0.5, 0, 0, 1050])
Q_f = np.diag([75, 75, 20, 5, 1, 1, 0, 0, 1])
# Retain the original baseline control penalties.
R = np.diag([0.1, 10.0])

alpha_values = [1.0, 0.5, 0.25, 0.125, 0.0625]


def state_difference(state, reference):
    """Subtract states using the shortest signed heading difference."""
    difference = np.asarray(state) - np.asarray(reference)
    difference = np.array(difference, dtype=float, copy=True)
    heading = difference[..., 2]
    difference[..., 2] = np.arctan2(np.sin(heading), np.cos(heading))
    return difference


def discrete_blend_dynamics_model(x, u, p, params_car, exact, dt):
    """Match NMPC's five RK4 substeps, holding u over the control interval."""
    num_substeps = 5  # blend_solver.create_ocp: sim_method_num_steps
    h = dt / num_substeps
    x_next = x
    for _ in range(num_substeps):
        k1 = blend_model(x_next, u, p, params_car, exact)
        k2 = blend_model(x_next + h / 2 * k1, u, p, params_car, exact)
        k3 = blend_model(x_next + h / 2 * k2, u, p, params_car, exact)
        k4 = blend_model(x_next + h * k3, u, p, params_car, exact)
        x_next = x_next + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def backward(ref_trajectory, cur_trajectory, cur_controls, p, params_car, exact, dt):
    x_sym = ca.MX.sym('x', 9)   # state: x, y, phi, vx, vy, omega, omega_w, current, delta
    u_sym = ca.MX.sym('u', 2)   # control rate: curent slew rate, steer rate
    p_sym = ca.MX.sym('p', 5)   # parameters vector: Cf, Cr, muf, mur, Cro

    #next state
    x_next_sym = discrete_blend_dynamics_model(x_sym, u_sym, p_sym, params_car, exact, dt)

    #set up jacobian computation
    A_jacobian_expr = ca.jacobian(x_next_sym, x_sym)
    B_jacobian_expr = ca.jacobian(x_next_sym, u_sym)
    A_func = ca.Function("A_func", [x_sym,u_sym,p_sym], [A_jacobian_expr])
    B_func = ca.Function("B_func", [x_sym,u_sym,p_sym], [B_jacobian_expr])

    #track error
    error = state_difference(cur_trajectory, ref_trajectory)

    #cost
    l_terminal = 0.5 * error[-1].T @ Q_f @ error[-1]

    #initialize terminal value
    V_x = Q_f @ error[-1]
    V_xx = Q_f

    total_cost = 0

    k_list = []
    K_list = []

    
    for i in reversed(range(cur_trajectory.shape[0] - 1)):
        x = cur_trajectory[i]
        u = cur_controls[i]

        #cost at each step
        l_k = 0.5 * error[i].T @ Q @ error[i] + 0.5 * u.T @ R @ u
        total_cost += l_k

        #linearize dynamics
        A_k = np.array(A_func(x,u,p))
        B_k = np.array(B_func(x,u,p))

        ''' used for debug
        A_idx = np.unravel_index(
            np.argmax(np.abs(A_k)),
            A_k.shape
        )

        B_idx = np.unravel_index(
            np.argmax(np.abs(B_k)),
            B_k.shape
        )

        state_names = [
            "x", "y", "phi", "vx", "vy",
            "omega", "omega_w", "current", "delta"
        ]

        control_names = [
            "current_slew_rate",
            "steer_rate"
        ]

        print(
            f"\nbackward step {i}"
        )

        print(
            f"max A = A{A_idx} = {A_k[A_idx]:.3e}"
        )

        print(
            f"meaning: d({state_names[A_idx[0]]}_next) / "
            f"d({state_names[A_idx[1]]})"
        )

        print(
            f"max B = B{B_idx} = {B_k[B_idx]:.3e}"
        )

        print(
            f"meaning: d({state_names[B_idx[0]]}_next) / "
            f"d({control_names[B_idx[1]]})"
        )

        print("state =", x)

        print(
            f"backward step {i}: "
            f"max|A|={np.max(np.abs(A_k)):.3e}, "
            f"max|B|={np.max(np.abs(B_k)):.3e}, "
            f"max|Vxx|={np.max(np.abs(V_xx)):.3e}"
        )
        '''

        #quadratic approximation
        l_x = Q @ error[i]
        l_u = R @ u
        l_xx = Q
        l_uu = R
        l_ux = 0

        #construct Q function derivatives
        Q_x = l_x + A_k.T @ V_x
        Q_u = l_u + B_k.T @ V_x
        Q_xx = l_xx + A_k.T @ V_xx @ A_k
        Q_uu = l_uu + B_k.T @ V_xx @ B_k

        print(
            f"    max|Quu|={np.max(np.abs(Q_uu)):.3e}, "
            f"cond(Quu)={np.linalg.cond(Q_uu):.3e}"
        )
        Q_ux = l_ux + B_k.T @ V_xx @ A_k

        #solve for optimal control rule
        k = -np.linalg.solve(Q_uu, Q_u)
        K = -np.linalg.solve(Q_uu, Q_ux)
        k_list.append(k)
        K_list.append(K)

        #update V for time step i-1
        V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
        V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K

        if not np.all(np.isfinite(V_xx)):
            print("V_xx exploded at backward step:", i)
            print("state =", x)
            break

    total_cost += l_terminal

    k_list.reverse()
    K_list.reverse()

    return np.array(k_list), np.array(K_list), total_cost


def forward(k_list, K_list, ref_trajectory, cur_trajectory, cur_controls, p, params_car, exact, dt, alpha):
    x_sym = ca.MX.sym('x', 9)   # state: x, y, phi, vx, vy, omega, omega_w, current, delta
    u_sym = ca.MX.sym('u', 2)   # control rate: curent slew rate, steer rate
    p_sym = ca.MX.sym('p', 5)   # parameters vector: Cf, Cr, muf, mur, Cro
    dynamics_expr = discrete_blend_dynamics_model(x_sym, u_sym, p_sym, params_car, exact, dt)
    dynamics_func = ca.Function("dynamics_func", [x_sym, u_sym, p_sym], [dynamics_expr])

    x_new = cur_trajectory[0]
    new_trajectory = [x_new]
    new_controls = []

    total_cost = 0

    for i in range(cur_trajectory.shape[0] - 1):
        x = cur_trajectory[i]
        u = cur_controls[i]
        
        delta_x = state_difference(x_new, x)

        feedforward = alpha * k_list[i]
        feedback = K_list[i] @ delta_x

        '''
        if i <= 1:
            print(f"\nSTEP {i}")
            print("x_new =", x_new)
            print("x_nominal =", x)
            print("delta_x =", delta_x)
            print("largest delta index =", np.argmax(np.abs(delta_x)))
    
        print(
            f"step {i}: "
            f"max|dx|={np.max(np.abs(delta_x)):.3e}, "
            f"max|ff|={np.max(np.abs(feedforward)):.3e}, "
            f"max|fb|={np.max(np.abs(feedback)):.3e}"
        )
        '''

        delta_u = feedforward + feedback

        '''
        delta_x = x_new - x
        delta_u = alpha * k_list[i] + K_list[i] @ delta_x
        print("max |delta_x|:", np.max(np.abs(delta_x)))
        print("max |alpha*k|:", np.max(np.abs(alpha * k_list[i])))
        print("max |K*delta_x|:", np.max(np.abs(K_list[i] @ delta_x)))
        '''
        
        u_new = u + delta_u
        # Predict using the physical rate limits applied by the ROS caller.
        u_new = np.clip(u_new,
                        [-300.0, -params_car['max_steer_vel']],
                        [300.0, params_car['max_steer_vel']])
        if not np.all(np.isfinite(u_new)):
            raise RuntimeError(f'Nonfinite blend forward control at step {i}')

        if i < 5:
            print(f"\nFORWARD STEP {i}")
            print("u nominal =", u)
            print("feedforward =", feedforward)
            print("feedback =", feedback)
            print("u new =", u_new)
        

        #cost at each step
        error_new = state_difference(x_new, ref_trajectory[i])
        l_k = 0.5 * error_new.T @ Q @ error_new + 0.5 * u_new.T @ R @ u_new
        total_cost += l_k

        x_new = np.array(dynamics_func(x_new, u_new, p)).reshape(-1)
        if not np.all(np.isfinite(x_new)):
            raise RuntimeError(f'Nonfinite blend forward prediction at step {i}')
        
        new_controls.append(u_new)
        new_trajectory.append(x_new)

    #terminal cost
    error_terminal = state_difference(x_new, ref_trajectory[-1])
    l_terminal = 0.5 * error_terminal.T @ Q_f @ error_terminal
    total_cost += l_terminal

    return np.array(new_controls), np.array(new_trajectory), total_cost
