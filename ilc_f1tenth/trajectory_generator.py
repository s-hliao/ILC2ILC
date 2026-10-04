import numpy as np
import matplotlib.pyplot as plt

from params import F110


def generate_s_curve(
    params_car,
    path_length=8.0,      # S curve total arc length [m]
    vx_ref=3.0,           # desired longitudinal speed [m/s]
    dt=0.05,              # time step [s]
    kappa_max=0.20,       # maximum curvature [1/m]
    hard=False
):
    """
    Generate S-curve reference trajectory for F1TENTH.

    Each row:
    [x_ref, y_ref, phi_ref, vx_ref, vy_ref, omega_ref,
     omega_w_ref, current_ref, delta_ref]
    """

    lf = params_car['lf']
    lr = params_car['lr']
    rw = params_car['rw']

    L = lf + lr

    #hard trajectory parameters
    if hard:
        kappa_max = 0.45

    # Since vx is constant:
    # ds = v * dt
    ds_nominal = vx_ref * dt

    # Arc-length samples
    s = np.arange(0.0, path_length, ds_nominal)

    if len(s) == 0 or s[-1] < path_length:
        s = np.append(s, path_length)

    # 1. Define S-curve curvature
    if not hard:
        kappa = kappa_max * np.sin(
            2.0 * np.pi * s / path_length
        )

    else:
        #more aggressive S curve
        kappa = (
            kappa_max * np.sin(
                2.0 * np.pi * s / path_length
            )
            + 0.35 * kappa_max * np.sin(
                4.0 * np.pi * s / path_length
            )
        )

    # 2. curvature -> heading
    #
    # d(phi)/ds = kappa
    phi = np.zeros_like(s)

    for k in range(1, len(s)):
        ds = s[k] - s[k - 1]

        # trapezoidal integration
        phi[k] = phi[k - 1] + 0.5 * (
            kappa[k - 1] + kappa[k]
        ) * ds

    # 3. reference velocities
    vx = np.full_like(s, vx_ref)

    if not hard:
        # No desired lateral velocity in body frame
        vy = np.zeros_like(s)

    else:
        #add lateral velocity for drift-like behavior
        vy = -0.6 * np.tanh(5.0 * kappa)

    # omega = v * curvature
    omega = vx_ref * kappa

    # 4. heading -> x, y position
    x = np.zeros_like(s)
    y = np.zeros_like(s)

    for k in range(1, len(s)):
        ds = s[k] - s[k - 1]

        phi_mid = 0.5 * (
            phi[k - 1] + phi[k]
        )

        if not hard:
            x[k] = x[k - 1] + np.cos(phi_mid) * ds
            y[k] = y[k - 1] + np.sin(phi_mid) * ds

        else:
            #account for lateral velocity in hard mode
            vx_world = (
                vx[k] * np.cos(phi_mid)
                - vy[k] * np.sin(phi_mid)
            )

            vy_world = (
                vx[k] * np.sin(phi_mid)
                + vy[k] * np.cos(phi_mid)
            )

            x[k] = x[k - 1] + vx_world * dt
            y[k] = y[k - 1] + vy_world * dt

    # 5. F1TENTH additional states

    # wheel angular velocity assuming approximately zero longitudinal slip
    omega_w = vx / rw

    # current reference
    current = np.zeros_like(s)

    # bicycle-model steering angle corresponding to curvature
    delta = np.arctan(L * kappa)

    # 6. Combine into x_ref
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
        hard=True
    )

    print("Reference trajectory shape:", ref_traj.shape)

    print(
        "\nFirst 5 points "
        "[x, y, phi, vx, vy, omega, omega_w, current, delta]:"
    )
    print(ref_traj[:5])

    # Save for later use
    np.savetxt(
        "s_curve_ref.csv",
        ref_traj,
        delimiter=",",
        header="x,y,phi,vx,vy,omega,omega_w,current,delta",
        comments=""
    )

    # Plot path
    plt.figure()
    plt.plot(ref_traj[:, 0], ref_traj[:, 1])
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("F1TENTH S-Curve Reference Trajectory")
    plt.grid(True)
    plt.savefig("s_curve_ref.png", dpi=300, bbox_inches="tight")
    plt.close()