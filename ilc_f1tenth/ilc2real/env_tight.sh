# the tight-track setup (user, 2026-10-09 afternoon): mu 0.2 (plastic tires), square2fast shortened 0.5 m and figfast 0.25 m at each end
# of their long direction (make_tight_tracks.py), the room's safety envelope (a crash beyond it): 0.3 m until 2026-10-28, now 0.5 m (user: "0.5 m is fine in mocap";
# = net_runner.py's safety_ey on the real car). source me.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export F1T_MU=0.2 F1T_PLANS=$D/plans_mu02_tight F1T_TRACKS=$D/../lifted_linear_tire_20261009/tracks_tight F1T_SAFETY_EY=0.5
