import numpy as np
import matplotlib.pyplot as plt


STATE_NAMES = [
    "x",
    "y",
    "phi",
    "vx",
    "vy",
    "omega",
    "omega_w",
    "current",
    "delta",
]


def calculate_errors(ref_trajectory, actual_trajectory):
    ref_trajectory = np.asarray(ref_trajectory)
    actual_trajectory = np.asarray(actual_trajectory)

    if ref_trajectory.shape != actual_trajectory.shape:
        raise ValueError(
            f"Shape mismatch: ref={ref_trajectory.shape}, "
            f"actual={actual_trajectory.shape}"
        )

    errors = actual_trajectory - ref_trajectory

    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)

    position_error = np.sqrt(
        errors[:, 0] ** 2 +
        errors[:, 1] ** 2
    )

    return errors, rmse, mae, position_error


def print_error_summary(ref_trajectory, actual_trajectory):
    errors, rmse, mae, position_error = calculate_errors(
        ref_trajectory,
        actual_trajectory
    )

    print("\n===== Tracking Error Summary =====")

    for i, name in enumerate(STATE_NAMES):
        print(
            f"{name:>8s}: "
            f"RMSE = {rmse[i]:.6f}, "
            f"MAE = {mae[i]:.6f}"
        )

    print("\nPosition tracking error:")
    print(f"Mean = {np.mean(position_error):.6f} m")
    print(f"Max  = {np.max(position_error):.6f} m")
    print(f"RMSE = {np.sqrt(np.mean(position_error ** 2)):.6f} m")

    return errors, rmse, mae, position_error


def plot_xy_trajectory(
    ref_trajectory,
    actual_trajectory,
    save_path="trajectory_comparison.png"
):
    plt.figure(figsize=(8, 6))

    plt.plot(
        ref_trajectory[:, 0],
        ref_trajectory[:, 1],
        label="Reference",
        linewidth=2
    )

    plt.plot(
        actual_trajectory[:, 0],
        actual_trajectory[:, 1],
        label="Actual",
        linewidth=2
    )

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Reference vs Actual Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved trajectory comparison to: {save_path}")


def plot_position_error(
    ref_trajectory,
    actual_trajectory,
    dt,
    save_path="position_error.png"
):
    _, _, _, position_error = calculate_errors(
        ref_trajectory,
        actual_trajectory
    )

    time = np.arange(len(position_error)) * dt

    plt.figure(figsize=(8, 5))

    plt.plot(
        time,
        position_error
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    plt.title("Position Tracking Error")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved position error plot to: {save_path}")


def plot_state_errors(
    ref_trajectory,
    actual_trajectory,
    dt,
    save_prefix="state_error"
):
    errors, _, _, _ = calculate_errors(
        ref_trajectory,
        actual_trajectory
    )

    time = np.arange(
        errors.shape[0]
    ) * dt

    for i, name in enumerate(STATE_NAMES):
        plt.figure(figsize=(8, 5))

        plt.plot(
            time,
            errors[:, i]
        )

        plt.xlabel("Time [s]")
        plt.ylabel(f"{name} error")
        plt.title(f"{name} Tracking Error")
        plt.grid(True)
        plt.tight_layout()

        save_path = (
            f"{save_prefix}_{name}.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved {name} error plot to: "
            f"{save_path}"
        )


def compare_trajectories(
    ref_trajectory,
    actual_trajectory,
    dt,
    prefix="trajectory"
):
    """
    Print tracking errors and save plots.

    Files:
        <prefix>_comparison.png
        <prefix>_position_error.png
    """

    print_error_summary(
        ref_trajectory,
        actual_trajectory
    )

    plot_xy_trajectory(
        ref_trajectory,
        actual_trajectory,
        save_path=f"{prefix}_comparison.png"
    )

    plot_position_error(
        ref_trajectory,
        actual_trajectory,
        dt,
        save_path=f"{prefix}_position_error.png"
    )


if __name__ == "__main__":
    print(
        "Import this file and call "
        "compare_trajectories("
        "ref_trajectory, actual_trajectory, dt)."
    )
