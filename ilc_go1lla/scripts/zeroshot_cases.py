"""The perturbation cases (LLA-MPPI-style); see ilc_go1lla/walkenv.py for their meaning."""
CASES = {
    'nominal': {},
    'weakFR50': dict(weak_leg=('FR', 0.5)),
    'weakFR35': dict(weak_leg=('FR', 0.35)),
    'weakFR20': dict(weak_leg=('FR', 0.2)),
    'weakFR12': dict(weak_leg=('FR', 0.12)),
    'lockFR': dict(weak_leg=('FR', 0.0)),
    'lockRL': dict(weak_leg=('RL', 0.0)),
    'payload2R': dict(payload=(2.0, -0.10, 0.08)),
    'payload3R': dict(payload=(3.0, -0.10, 0.08)),
    'payload3L': dict(payload=(3.0, 0.10, 0.08)),
    'payload4R': dict(payload=(4.0, -0.10, 0.08)),
    'payload5R': dict(payload=(5.0, -0.10, 0.08)),
    'foam': dict(foam=(0.08, 0.10)),
    'foam_mid': dict(foam=(0.11, 0.12)),
    'foam_soft': dict(foam=(0.15, 0.15)),
    'combo': dict(weak_leg=('FR', 0.5), payload=(3.0, -0.10, 0.08), foam=(0.08, 0.10)),
}


def parse_case(c):
    """A CASES name, or a spec: foam:tc,width[,dampratio] | payload:m,dy,dz | weak:leg,scale."""
    if c in CASES:
        return dict(CASES[c])
    kind, vals = c.split(':')
    if kind == 'weak':
        leg, sc = vals.split(',')
        return dict(weak_leg=(leg, float(sc)))
    v = tuple(float(x) for x in vals.split(','))
    if kind == 'foamE':                                         # (11-24) foam + elliptic friction cones (goalstage.GoalSimWalker only)
        return dict(foam=v, cone='elliptic')
    return {'foam': dict(foam=v), 'payload': dict(payload=v)}[kind]
