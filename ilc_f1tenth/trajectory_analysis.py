"""
Plots of ILC runs over trials (headless-safe: figures are saved; plt.show() only with show=True).

    plot_trials(ref, trials, dt)              every trial's path and states, coloured by trial, plus the errors per trial
    compare_trajectories(ref, trajectory, dt) one trajectory against the reference
    save_trials / load_trials                 the run's trials as an npz, to re-plot without rerunning

Trajectories are (N+1, 9) arrays of [x, y, phi, vx, vy, omega, omega_w, current, delta].
"""
import os

import matplotlib
import numpy as np

if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")                                   # no display: save files only
import matplotlib.pyplot as plt  # noqa: E402

STATES = ((3, "vx [m/s]"), (4, "vy [m/s]"), (5, "yaw rate [rad/s]"), (8, "steer [rad]"))


def position_errors(ref, trials):
    """(n_trials, N+1) distance from the reference at each step"""
    trials = np.asarray(trials)
    return np.linalg.norm(trials[:, :, :2] - np.asarray(ref)[None, :, :2], axis=2)


# where every run's trials go (one timestamped file per run), whatever directory the script is started from
RUN_DIR = os.path.expanduser("~/ilc_ws/log/f1tenth")


def save_trials(name, ref, trials, controls, dt, costs=None):
    """saves RUN_DIR/<name>_<timestamp>.npz (name: e.g. "ilc", "f1tenth_ilc") and returns its path"""
    import time
    os.makedirs(RUN_DIR, exist_ok=True)
    path = os.path.join(RUN_DIR, f"{os.path.splitext(os.path.basename(name))[0]}_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    np.savez(path, ref=ref, trials=np.asarray(trials), controls=np.asarray(controls), dt=dt,
             costs=np.asarray(costs if costs is not None else []))
    print(f"saved {path}")
    return path


def latest_run():
    import glob
    runs = sorted(glob.glob(os.path.join(RUN_DIR, "*.npz")), key=os.path.getmtime)
    if not runs:
        raise SystemExit(f"no runs in {RUN_DIR} yet: run ILC.py or f1tenth_ilc.py first")
    return runs[-1]


def load_trials(path):
    d = np.load(path)
    return d["ref"], d["trials"], d["controls"], float(d["dt"]), d["costs"]


def plot_trials(ref, trials, dt, costs=None, out="ilc_trials.png", show=False):
    """Every trial over the reference (dark = early, yellow = late): the path, the per-trial mean and max position
    error (and cost, if given), and vx, vy, yaw rate and steer over time."""
    ref, trials = np.asarray(ref), np.asarray(trials)
    n = len(trials)
    cols = plt.cm.viridis(np.linspace(0, 1, max(n, 2)))
    t = np.arange(ref.shape[0]) * dt
    err = position_errors(ref, trials)

    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    a = ax[0, 0]
    a.plot(ref[:, 0], ref[:, 1], "k--", lw=2, label="reference")
    for i, tr in enumerate(trials):
        a.plot(tr[:, 0], tr[:, 1], color=cols[i], lw=1.2 if i in (0, n - 1) else 0.7,
               label=f"trial {i}" if i in (0, n - 1) else None)
    a.set_aspect("equal")
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")
    a.set_title("path")
    a.legend()

    ax[0, 1].plot(err.mean(1), "o-", label="mean")
    ax[0, 1].plot(err.max(1), "s-", label="max")
    ax[0, 1].set_xlabel("trial")
    ax[0, 1].set_ylabel("position error [m]")
    ax[0, 1].set_title("position error per trial")
    ax[0, 1].legend()

    a = ax[0, 2]
    for i in range(n):
        a.plot(t, err[i], color=cols[i], lw=0.8)
    a.set_xlabel("t [s]")
    a.set_ylabel("position error [m]")
    a.set_title("position error over time")

    a = ax[0, 3]
    if costs is not None and len(costs):
        a.plot(np.arange(len(costs)), costs, "o-")
        a.set_yscale("log")
        a.set_title("cost per trial")
        a.set_xlabel("trial")
    else:
        a.axis("off")

    for j, (k, name) in enumerate(STATES):
        a = ax[1, j]
        a.plot(t, ref[:, k], "k--", lw=2)
        for i, tr in enumerate(trials):
            a.plot(t, tr[:, k], color=cols[i], lw=0.8)
        a.set_xlabel("t [s]")
        a.set_title(name)

    for a in ax.flat:
        a.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, max(n - 1, 1)))
    fig.colorbar(sm, ax=ax, label="trial", fraction=0.02)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out} ({n} trials; mean position error {err.mean(1)[0]:.3f} -> {err.mean(1)[-1]:.3f} m)")
    if show:
        plt.show()
    plt.close(fig)


def compare_trajectories(ref_trajectory, trajectory, dt, out="ilc_compare.png", show=False):
    """One trajectory against the reference: the path, the position error and the states over time."""
    plot_trials(ref_trajectory, [trajectory], dt, out=out, show=show)


if __name__ == "__main__":
    # python trajectory_analysis.py [RUN.npz]   (default: the newest run in RUN_DIR); the PNG goes next to the run
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else latest_run()
    ref, trials, _, dt, costs = load_trials(path)
    plot_trials(ref, trials, dt, costs=costs, out=os.path.splitext(path)[0] + ".png", show=True)
