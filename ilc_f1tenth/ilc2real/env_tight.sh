# the tight-track setup (user, 2026-10-09 afternoon): mu 0.2 (plastic tires), square2fast shortened 0.5 m and figfast 0.25 m at each end
# of their long direction (make_tight_tracks.py), the room's 0.3 m safety envelope (a crash beyond it). source me.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export F1T_MU=0.2 F1T_PLANS=$D/plans_mu02_tight F1T_TRACKS=$D/../lifted_linear_tire_20261009/tracks_tight F1T_SAFETY_EY=0.3
