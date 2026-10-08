import numpy as np
import matplotlib.pyplot as plt

from params import F110


def generate_s_curve(
    params_car,
    path_length=8.0,      # S curve total arc length [m]
    vx_ref=3.0,           # target longitudinal speed [m/s]
    dt=0.05,              # time step [s]
    kappa_max=0.20,       # maximum curvature [1/m]
    hard=False,
    accel_time=1.0        # time to accelerate from 0 to vx_ref [s]
):
    """
    Generate S-curve reference trajectory for F1TENTH.

    Vehicle starts from rest with zero current and accelerates to vx_ref.
    This is a kinematic reference, not an exact blend-model solution.
    Current uses the same torque-to-acceleration conversion as ROS rollout.

    Each row:
    [x_ref, y_ref, phi_ref, vx_ref, vy_ref, omega_ref,
     omega_w_ref, current_ref, delta_ref]
    """

    lf = params_car['lf']
    lr = params_car['lr']
    rw = params_car['rw']

    L = lf + lr
    if path_length <= 0 or vx_ref <= 0 or dt <= 0 or accel_time <= 0:
        raise ValueError('path_length, vx_ref, dt and accel_time must be positive')

    # -----------------------------------------
    # Hard trajectory parameters
    # -----------------------------------------

    if hard:
        kappa_max = 0.45

    # -----------------------------------------
    # 1. Generate longitudinal velocity profile
    #    and corresponding arc length
    # -----------------------------------------

    s_list = [0.0]
    vx_list = [0.0]

    t = 0.0
    s_current = 0.0

    while s_current < path_length:

        t += dt

        # Smooth velocity ramp: zero acceleration at rest and at cruise.
        # This avoids an instantaneous nonzero current at the initial state.
        if t < accel_time:
            ramp = t / accel_time
            vx_current = vx_ref * ramp**2 * (3.0 - 2.0 * ramp)
        else:
            vx_current = vx_ref

        ds = 0.5 * (vx_list[-1] + vx_current) * dt

        s_current += ds

        # Do not go beyond path_length
        if s_current > path_length:
            s_current = path_length

        s_list.append(s_current)
        vx_list.append(vx_current)

    s = np.array(s_list)
    vx = np.array(vx_list)

    # -----------------------------------------
    # 2. Define S-curve curvature
    # -----------------------------------------

    if not hard:

        kappa = kappa_max * np.sin(
            2.0 * np.pi * s / path_length
        )

    else:

        kappa = (
            kappa_max * np.sin(
                2.0 * np.pi * s / path_length
            )
            +
            0.35 * kappa_max * np.sin(
                4.0 * np.pi * s / path_length
            )
        )

    # -----------------------------------------
    # 3. Curvature -> heading
    #
    # d(phi) / ds = kappa
    # -----------------------------------------

    phi = np.zeros_like(s)

    for k in range(1, len(s)):

        ds = s[k] - s[k - 1]

        phi[k] = (
            phi[k - 1]
            +
            0.5
            * (kappa[k - 1] + kappa[k])
            * ds
        )

    # -----------------------------------------
    # 4. Reference lateral velocity
    # -----------------------------------------

    omega = vx * kappa

    if not hard:

        # Body-frame CoG lateral velocity for a no-slip rear axle:
        # vy - lr * omega = 0.
        vy = lr * omega

    else:

        # Drift-like lateral velocity.
        # Scale it by vx / vx_ref so vy also starts at zero.
        speed_scale = vx / vx_ref

        vy = (
            -0.6
            * np.tanh(5.0 * kappa)
            * speed_scale
        )

    # -----------------------------------------
    # 5. Yaw rate
    #
    # omega = vx * curvature
    # -----------------------------------------

    omega = vx * kappa

    # -----------------------------------------
    # 6. Integrate position
    # -----------------------------------------

    x = np.zeros_like(s)
    y = np.zeros_like(s)

    for k in range(1, len(s)):

        phi_mid = 0.5 * (
            phi[k - 1] + phi[k]
        )

        # Approximate body velocities during interval
        vx_mid = 0.5 * (
            vx[k - 1] + vx[k]
        )

        vy_mid = 0.5 * (
            vy[k - 1] + vy[k]
        )

        # Body frame -> world frame
        vx_world = (
            vx_mid * np.cos(phi_mid)
            -
            vy_mid * np.sin(phi_mid)
        )

        vy_world = (
            vx_mid * np.sin(phi_mid)
            +
            vy_mid * np.cos(phi_mid)
        )

        x[k] = (
            x[k - 1]
            +
            vx_world * dt
        )

        y[k] = (
            y[k - 1]
            +
            vy_world * dt
        )

    # -----------------------------------------
    # 7. Additional F1TENTH states
    # -----------------------------------------

    # Wheel angular velocity
    # Starts from zero because vx starts from zero.
    omega_w = vx / rw

    # rollout uses I[k+1] to update its velocity command over interval k.
    # Match that discrete convention exactly, with an initially zero current.
    acceleration = np.zeros_like(vx)
    acceleration[1:] = np.diff(vx) / dt
    torque_per_amp = (params_car['gear_ratio'] * 1.5
                      * params_car['pole_pairs'] * params_car['lambda'])
    current = params_car['mass'] * rw * acceleration / torque_per_amp

    # Bicycle-model steering angle
    delta = np.arctan(
        L * kappa
    )

    # -----------------------------------------
    # 8. Combine reference trajectory
    # -----------------------------------------

    ref_traj = np.column_stack([
        x,
        y,
        phi,
        vx,
        vy,
        omega,
        omega_w,
        current,
        delta
    ])

    return ref_traj


if __name__ == "__main__":

    params_car = F110()

    ref_traj = generate_s_curve(
        params_car=params_car,
        path_length=8.0,
        vx_ref=3.0,
        dt=0.05,
        kappa_max=0.20,
        hard=True,
        accel_time=1.0
    )

    print(
        "Reference trajectory shape:",
        ref_traj.shape
    )

    print(
        "\nFirst 10 points "
        "[x, y, phi, vx, vy, omega, "
        "omega_w, current, delta]:"
    )

    print(
        ref_traj[:10]
    )

    # -----------------------------------------
    # Save CSV
    # -----------------------------------------

    np.savetxt(
        "s_curve_ref.csv",
        ref_traj,
        delimiter=",",
        header=(
            "x,y,phi,vx,vy,omega,"
            "omega_w,current,delta"
        ),
        comments=""
    )

    # -----------------------------------------
    # Plot path
    # -----------------------------------------

    plt.figure()

    plt.plot(
        ref_traj[:, 0],
        ref_traj[:, 1]
    )

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(
        "F1TENTH S-Curve Reference Trajectory"
    )
    plt.grid(True)

    plt.savefig(
        "s_curve_ref.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------------------
    # Plot vx reference
    # -----------------------------------------

    time = (
        np.arange(ref_traj.shape[0])
        * 0.05
    )

    plt.figure()

    plt.plot(
        time,
        ref_traj[:, 3]
    )

    plt.xlabel("Time [s]")
    plt.ylabel("vx [m/s]")
    plt.title(
        "Reference Longitudinal Velocity"
    )
    plt.grid(True)

    plt.savefig(
        "vx_ref.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
