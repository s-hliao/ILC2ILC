"""
F1TENTH-scale parameters for the fork's multi-body (MB) model, which ships only full-scale
ones (CommonRoad vehicle 2, BMW 320i). Engineering estimates, not an identification:

* anchored on the F1TENTH values that exist -- the gym's F1TENTH single-track set (mass,
  yaw inertia, axle distances, CoG height, friction, cornering stiffness per load, steering
  and speed limits, width) and ilc_f1tenth/params.py (wheel radius and spin inertia);
* everything with no F1TENTH counterpart (suspension, roll/pitch inertia, compliant joints,
  tire vertical/lateral compliance) Froude-scaled from the full-scale set: mass ratio
  k_m = m / m_full, length ratio k_L = wheelbase / wheelbase_full, gravity fixed, so time
  scales as sqrt(k_L), force as k_m. Spring rates x k_m / k_L, dampers x k_m / sqrt(k_L),
  torsion x k_m k_L, compliances x k_L / k_m, rad/m terms / k_L. That keeps suspension
  damping ratios and body/wheel-hop modes at the same Froude-scaled frequencies (sprung
  corner ~3.8 Hz, vs ~1.4 Hz full-scale), and static tire deflection ~2 mm;
* inertias other than yaw kept at the full-scale ratio to the yaw inertia;
* Pacejka MF shape, curvature and combined-slip coefficients (dimensionless) from the
  full-scale tire; peak friction p_dy1 = mu (= the full-scale value, 1.0489) and the slip
  stiffness per load set to the F1TENTH set's mu * (C_Sf + C_Sr) / 2 (MF has one tire for
  all four corners), longitudinal at the full-scale K_x / K_y ratio;
* drive and brake torque split 50/50 front/rear (the F1TENTH Traxxas chassis is 4WD with
  motor braking through the same driveline; the full-scale set is front-drive, T_se = 1).
"""
from fork_import import F1TENTH_VEHICLE_PARAMETERS as F1, FULLSCALE_VEHICLE_PARAMETERS as FS

K_M = F1.m / FS.m                                  # 0.00305
K_L = (F1.lf + F1.lr) / (FS.lf + FS.lr)            # 0.138
K_T = K_L ** 0.5                                   # time

RW = 0.051                                         # params.py rw
IW_WHEEL = 0.9 * 0.1 * 0.043 ** 2                  # params.py Iw = 0.9 mw r^2, mw = 4 x 0.1
K_Y = F1.mu * 0.5 * (F1.C_Sf + F1.C_Sr)            # cornering stiffness / load, 1/rad

m_uf = FS.m_uf * K_M
m_ur = FS.m_ur * K_M

F1TENTH_MB_PARAMETERS = F1.with_updates(
    kappa_dot_max=FS.kappa_dot_max, kappa_dot_dot_max=FS.kappa_dot_dot_max,
    j_max=FS.j_max, j_dot_max=FS.j_dot_max,
    # masses: unsprung at the full-scale share, sprung the rest of the F1TENTH mass
    m_s=F1.m - m_uf - m_ur, m_uf=m_uf, m_ur=m_ur,
    # inertias: yaw from the F1TENTH set, roll/pitch/unsprung at full-scale ratios to it
    I_z_body=F1.I, I_xz_s=0.0,
    I_Phi_s=F1.I * FS.I_Phi_s / FS.I_z_body,
    I_y_s=F1.I * FS.I_y_s / FS.I_z_body,
    I_uf=F1.I * FS.I_uf / FS.I_z_body,
    I_ur=F1.I * FS.I_ur / FS.I_z_body,
    # suspension (Froude)
    K_sf=FS.K_sf * K_M / K_L, K_sr=FS.K_sr * K_M / K_L,
    K_sdf=FS.K_sdf * K_M / K_T, K_sdr=FS.K_sdr * K_M / K_T,
    K_ras=FS.K_ras * K_M / K_L, K_rad=FS.K_rad * K_M / K_T,
    K_tsf=FS.K_tsf * K_M * K_L, K_tsr=FS.K_tsr * K_M * K_L,
    K_zt=FS.K_zt * K_M / K_L,
    K_lt=FS.K_lt * K_L / K_M,
    D_f=FS.D_f / K_L, D_r=FS.D_r / K_L, E_f=0.0, E_r=0.0,
    # geometry: track at the full-scale track/width ratio of the F1TENTH width; heights
    # from the F1TENTH CoG height
    T_f=F1.width * FS.T_f / FS.width, T_r=F1.width * FS.T_r / FS.width,
    h_cg_mb=F1.h, h_s=F1.h * FS.h_s / FS.h_cg_mb, h_raf=0.0, h_rar=0.0,
    # wheels
    R_w=RW, I_y_w=IW_WHEEL,
    T_sb=0.5, T_se=0.5,
    # tire (Pacejka MF)
    tire_p_cx1=FS.tire_p_cx1, tire_p_dx1=FS.tire_p_dx1, tire_p_dx3=FS.tire_p_dx3,
    tire_p_ex1=FS.tire_p_ex1,
    tire_p_kx1=FS.tire_p_kx1 / abs(FS.tire_p_ky1) * K_Y,
    tire_p_hx1=FS.tire_p_hx1, tire_p_vx1=FS.tire_p_vx1,
    tire_r_bx1=FS.tire_r_bx1, tire_r_bx2=FS.tire_r_bx2, tire_r_cx1=FS.tire_r_cx1,
    tire_r_ex1=FS.tire_r_ex1, tire_r_hx1=FS.tire_r_hx1,
    tire_p_cy1=FS.tire_p_cy1, tire_p_dy1=F1.mu, tire_p_dy3=FS.tire_p_dy3,
    tire_p_ey1=FS.tire_p_ey1,
    tire_p_ky1=-K_Y,                               # full-scale sign convention
    tire_p_hy1=FS.tire_p_hy1, tire_p_hy3=FS.tire_p_hy3,
    tire_p_vy1=FS.tire_p_vy1, tire_p_vy3=FS.tire_p_vy3,
    tire_r_by1=FS.tire_r_by1, tire_r_by2=FS.tire_r_by2, tire_r_by3=FS.tire_r_by3,
    tire_r_cy1=FS.tire_r_cy1, tire_r_ey1=FS.tire_r_ey1, tire_r_hy1=FS.tire_r_hy1,
    tire_r_vy1=FS.tire_r_vy1, tire_r_vy3=FS.tire_r_vy3, tire_r_vy4=FS.tire_r_vy4,
    tire_r_vy5=FS.tire_r_vy5, tire_r_vy6=FS.tire_r_vy6,
)

if __name__ == '__main__':
    from dataclasses import fields
    for f in fields(F1TENTH_MB_PARAMETERS):
        print(f"{f.name:24s} {getattr(F1TENTH_MB_PARAMETERS, f.name):.6g}")
