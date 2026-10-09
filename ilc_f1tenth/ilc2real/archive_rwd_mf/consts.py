"""Constants shared by the JAX sim (sim_jax) and its numpy twin (sim_jax_np): kept jax-free so the CPU worker processes
of the real cars never initialize a GPU backend."""
import numpy as np

PREVIEW = np.array([0.0, 0.2, 0.4, 0.7, 1.0, 1.5])
DEV_SCALE = np.array([0.1, 0.2, 0.5, 0.5, 1.0, 0.5, 1.0, 0.1])     # e_y e_psi vx vy r wf*Rw wr*Rw delta
ERR_SCALE = np.array([0.05, 0.10, 0.30, 0.30, 0.50])               # ILC error rows: e_y e_psi vx vy r
FAIL_EY, FAIL_EPSI = 0.6, 1.2
WIN = np.arange(-20, 61)
