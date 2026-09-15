import numpy as np
import casadi as ca
from dynamics import discrete_dynamics_model

Q = np.diag([10, 10, 5, 2, 0.5, 0.5, 0, 0, 0.1])
Q_f = np.diag([50, 50, 20, 5, 1, 1, 0, 0, 1])
R = 0.1 * np.eye(2)

alpha_values = [1.0, 0.5, 0.25, 0.125, 0.0625]



def backward(ref_trajectory, cur_trajectory, cur_controls, p, params_car, exact, dt):
    x_sym = ca.MX.sym('x', 9)   # state: x, y, phi, vx, vy, omega, omega_w, current, delta
    u_sym = ca.MX.sym('u', 2)   # control rate: curent slew rate, steer rate
    p_sym = ca.MX.sym('p', 5)   # parameters vector: Cf, Cr, muf, mur, Cro

    #next state
    x_next_sym = discrete_dynamics_model(x_sym, u_sym, p_sym, params_car, exact, dt)

    #set up jacobian computation
    A_jacobian_expr = ca.jacobian(x_next_sym, x_sym)
    B_jacobian_expr = ca.jacobian(x_next_sym, u_sym)
    A_func = ca.Function("A_func", [x_sym,u_sym,p_sym], [A_jacobian_expr])
    B_func = ca.Function("B_func", [x_sym,u_sym,p_sym], [B_jacobian_expr])

    #track error
    error = cur_trajectory - ref_trajectory

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
        Q_ux = l_ux + B_k.T @ V_xx @ A_k

        #solve for optimal control rule
        k = -np.linalg.solve(Q_uu, Q_u)
        K = -np.linalg.solve(Q_uu, Q_ux)
        k_list.append(k)
        K_list.append(K)

        #update V for time step i-1
        V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
        V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K

    total_cost += l_terminal

    k_list.reverse()
    K_list.reverse()

    return np.array(k_list), np.array(K_list), total_cost


def forward(k_list, K_list, ref_trajectory, cur_trajectory, cur_controls, p, params_car, exact, dt, alpha):
    x_sym = ca.MX.sym('x', 9)   # state: x, y, phi, vx, vy, omega, omega_w, current, delta
    u_sym = ca.MX.sym('u', 2)   # control rate: curent slew rate, steer rate
    p_sym = ca.MX.sym('p', 5)   # parameters vector: Cf, Cr, muf, mur, Cro
    dynamics_expr = discrete_dynamics_model(x_sym, u_sym, p_sym, params_car, exact, dt)
    dynamics_func = ca.Function("dynamics_func", [x_sym, u_sym, p_sym], [dynamics_expr])

    x_new = cur_trajectory[0]
    new_trajectory = [x_new]
    new_controls = []

    total_cost = 0

    for i in range(cur_trajectory.shape[0] - 1):
        x = cur_trajectory[i]
        u = cur_controls[i]

        delta_x = x_new - x
        delta_u = alpha * k_list[i] + K_list[i] @ delta_x

        u_new = u + delta_u

        #cost at each step
        error_new = x_new - ref_trajectory[i]
        l_k = 0.5 * error_new.T @ Q @ error_new + 0.5 * u_new.T @ R @ u_new
        total_cost += l_k

        x_new = np.array(dynamics_func(x_new, u_new, p)).reshape(-1)
        
        new_controls.append(u_new)
        new_trajectory.append(x_new)

    #terminal cost
    error_terminal = x_new - ref_trajectory[-1]
    l_terminal = 0.5 * error_terminal.T @ Q_f @ error_terminal
    total_cost += l_terminal

    return np.array(new_controls), np.array(new_trajectory), total_cost


def alpha_search(k_list, K_list, ref_trajectory, cur_trajectory, cur_controls, current_cost, p, params_car, exact, dt):

    for alpha in alpha_values:

        new_controls, new_trajectory, new_cost = forward(
            k_list,
            K_list,
            ref_trajectory,
            cur_trajectory,
            cur_controls,
            p,
            params_car,
            exact,
            dt,
            alpha
        )

        if new_cost < current_cost:
            return alpha, new_controls, new_trajectory, new_cost

    return 0.0, cur_controls, cur_trajectory, current_cost