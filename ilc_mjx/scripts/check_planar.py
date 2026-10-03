#!/usr/bin/env python3
"""check_planar.py: the JAX planar kinematics against ilc_quad's CasADi model, at random states."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.model import quad_model  # noqa: E402
from ilc_mjx.planar import Planar  # noqa: E402
from ilc_quad.ilc_gen import PlanarQuadModel  # noqa: E402

fb = PlanarQuadModel(quad_model())
pl = Planar(fb)
rng = np.random.default_rng(0)
err = {k: 0.0 for k in ("feet", "com", "Jc", "Jcom", "H", "torque_map")}
for _ in range(50):
    s = fb.standing_state() + rng.normal(0, 0.2, 7)
    js = jnp.asarray(s)
    err["feet"] = max(err["feet"], np.abs(np.asarray(pl.feet(js)) - fb.feet(s).full().ravel()).max())
    err["com"] = max(err["com"], np.abs(np.asarray(pl.com(js)) - fb.com(s).full().ravel()).max())
    err["Jc"] = max(err["Jc"], np.abs(np.asarray(pl.feet_jac(js)) - fb.Jc(s).full()).max())
    err["Jcom"] = max(err["Jcom"], np.abs(np.asarray(pl.com_jac(js)) - fb.Jcom(s).full()).max())
    err["H"] = max(err["H"], np.abs(np.asarray(pl.mass_matrix(js)) - fb.H(s).full()).max())
    err["torque_map"] = max(err["torque_map"], np.abs(np.asarray(pl.torque_map(js[3:], js[2]))
                                                      - fb.torque_map(s[3:], s[2])).max())
print("max |JAX - CasADi| over 50 random states: " + "  ".join(f"{k} {v:.1e}" for k, v in err.items()))
assert max(err.values()) < 1e-6, "the JAX planar model disagrees with CasADi"
print("ok")
