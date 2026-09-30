"""sens_time.py RUN PLAN K: for the 60 cm flat jump, predicted vs actual change of pitch,
pitch rate and CoM velocity along the trajectory from trial K to K+1 of RUN."""
import sys, numpy as np
from ilc_quad.sim_quad_model import QuadModel
from ilc_quad.ilc_gen import JumpILC, build_lifted_G
run, ref, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
ilc = JumpILC(QuadModel("go1"), jump=(0.6, 0.0), box=None, phases=(30, 30, 30), margin=0.85,
              reference=ref, _check_reference=False)
N, Nc, nx, nu = ilc.N, ilc.Nc, ilc.nx, ilc.nu
a = np.load(f"{run}/trial_{k:03d}.npz", allow_pickle=True); b = np.load(f"{run}/trial_{k+1:03d}.npz", allow_pickle=True)
du = b["U_flown"] - a["U_flown"]; X = a["log_X"]
U_full = np.vstack([a["U_flown"], np.zeros((N - Nc, nu))])
A, B = ilc.nominal.linearize_along_trial(X, U_full, ilc.R1, ilc.R2, ilc.dt)
G = build_lifted_G(A, B, N, Nc, nx, nu, flatten=True)
pred = np.vstack([np.zeros(nx), (G @ du.reshape(-1)).reshape(N, nx)])
act = b["log_X"] - X
print("du mean front fx,fz rear fx,fz (dc):", du[:30].mean(0).round(2), " (sc):", du[30:].mean(0).round(2))
for kk in (10, 20, 30, 40, 50, 60, 70, 80, 90):
    print(f"k={kk:2d}  th pred {np.degrees(pred[kk,2]):+6.2f} act {np.degrees(act[kk,2]):+6.2f} deg | "
          f"w pred {np.degrees(pred[kk,5]):+6.1f} act {np.degrees(act[kk,5]):+6.1f} deg/s | "
          f"vx pred {100*pred[kk,3]:+5.1f} act {100*act[kk,3]:+5.1f} | vz pred {100*pred[kk,4]:+5.1f} act {100*act[kk,4]:+5.1f} cm/s")
