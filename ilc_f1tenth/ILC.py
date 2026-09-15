import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from params import F110
from trajectory_generator import generate_s_curve
from dynamics import discrete_dynamics_model
from iLQR import backward, alpha_search


def _get_tire_params(p_car):
    """
    Pull [Cf, Cr, muf, mur, Cro] to match the unpacking order used in
    dynamics: `Cf, Cr, muf, mur, Cro = [p[i] for i in range(5)]`.
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


def rollout_model(controls, x0, p, params_car, exact, dt):
    x_sym = ca.MX.sym('x', 9)   # state: x, y, phi, vx, vy, omega, omega_w, current, delta
    u_sym = ca.MX.sym('u', 2)   # control rate: curent slew rate, steer rate
    p_sym = ca.MX.sym('p', 5)   # parameters vector: Cf, Cr, muf, mur, Cro

    dynamics_expr = discrete_dynamics_model(
        x_sym,
        u_sym,
        p_sym,
        params_car,
        exact,
        dt
    )

    dynamics_func = ca.Function(
        "rollout_dynamics_func",
        [x_sym, u_sym, p_sym],
        [dynamics_expr]
    )

    trajectory = np.zeros((controls.shape[0] + 1, 9))
    trajectory[0] = x0

    for i in range(controls.shape[0]):
        x = trajectory[i]
        u = controls[i]

        x_next = np.array(
            dynamics_func(x, u, p)
        ).reshape(-1)

        trajectory[i + 1] = x_next

    return trajectory


def main():

    #simulation parameters
    dt = 0.05
    exact = False

    #ILC parameters
    epochs = 100

    #car parameters
    params_car = F110()
    tire_params = _get_tire_params(params_car)

    #reference trajectory
    ref_trajectory = generate_s_curve(
        params_car=params_car,
        path_length=8.0,
        vx_ref=3.0,
        dt=dt,
        kappa_max=0.20,
        hard = True
    )

    ref_num = ref_trajectory.shape[0]

    #initial controls
    cur_controls = np.zeros((ref_num - 1, 2))

    #initial state: x, y, phi, vx, vy, omega, omega_w, current, delta
    x0 = ref_trajectory[0].copy()

    #first rollout
    cur_trajectory = rollout_model(
        cur_controls,
        x0,
        tire_params,
        params_car,
        exact,
        dt
    )

    cost_history = []
    position_error_history = []

    for i in range(epochs):

        #backward pass
        k, K, cur_cost = backward(
            ref_trajectory,
            cur_trajectory,
            cur_controls,
            tire_params,
            params_car,
            exact,
            dt
        )

        #forward pass with alpha search
        alpha, new_controls, predicted_trajectory, predicted_cost = alpha_search(
            k,
            K,
            ref_trajectory,
            cur_trajectory,
            cur_controls,
            cur_cost,
            tire_params,
            params_car,
            exact,
            dt
        )

        if alpha == 0.0:
            print("No alpha reduces the cost.")
            break

        #apply the new controls in f1tenth and get a new trajectory
        #temporary: use the nonlinear dynamics model as the plant
        new_trajectory = rollout_model(
            new_controls,
            x0,
            tire_params,
            params_car,
            exact,
            dt
        )

        #then update cur_trajectory and cur_control
        cur_controls = new_controls
        cur_trajectory = new_trajectory

        position_error = np.mean(
            np.linalg.norm(
                cur_trajectory[:, :2] - ref_trajectory[:, :2],
                axis=1
            )
        )

        cost_history.append(cur_cost)
        position_error_history.append(position_error)

        print(
            f"Epoch {i + 1}: "
            f"cost = {cur_cost:.4f}, "
            f"predicted cost = {predicted_cost:.4f}, "
            f"alpha = {alpha}, "
            f"position error = {position_error:.4f}"
        )


    #plot graphs to see the progress
    plt.figure()
    plt.plot(
        ref_trajectory[:, 0],
        ref_trajectory[:, 1],
        label="Reference"
    )
    plt.plot(
        cur_trajectory[:, 0],
        cur_trajectory[:, 1],
        label="ILC trajectory"
    )
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("ILC Trajectory Tracking")
    plt.legend()
    plt.grid(True)
    plt.savefig("ilc_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close()


    plt.figure()
    plt.plot(cost_history)
    plt.xlabel("ILC Epoch")
    plt.ylabel("Cost")
    plt.title("ILC Cost")
    plt.grid(True)
    plt.savefig("ilc_cost.png", dpi=300, bbox_inches="tight")
    plt.close()


    plt.figure()
    plt.plot(position_error_history)
    plt.xlabel("ILC Epoch")
    plt.ylabel("Mean Position Error [m]")
    plt.title("ILC Tracking Error")
    plt.grid(True)
    plt.savefig("ilc_error.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()