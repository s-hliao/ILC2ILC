"""Import the dynamics of the f1tenth_gym fork (s-hliao/f1tenth_gym, ~/docker_workspaces/
ilc_f1tenth/src/f1tenth_gym) without its package __init__, which needs gymnasium: the parent
packages are registered as bare namespace modules so only envs/dynamic_models loads."""
import sys
import types
from pathlib import Path

FORK = Path('/workspaces/ilc_f1tenth/src/f1tenth_gym')


def _stub(name, path):
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m


_stub('f1tenth_gym', FORK / 'f1tenth_gym')
_stub('f1tenth_gym.envs', FORK / 'f1tenth_gym' / 'envs')

from f1tenth_gym.envs.dynamic_models import (  # noqa: E402,F401
    F1TENTH_VEHICLE_PARAMETERS, FULLSCALE_VEHICLE_PARAMETERS, DynamicModel, pid_steer, pid_accl)
from f1tenth_gym.envs.dynamic_models.multi_body import vehicle_dynamics_mb, init_mb  # noqa: E402,F401

from dataclasses import astuple, fields  # noqa: E402
import numpy as np  # noqa: E402

# The fork's MB code (vehicle_dynamics_mb, init_mb, tire_model) indexes the parameter vector
# without VehicleParameters' collision_body_center_x/y (fields 18, 19, added later), so
# VehicleParameters.to_array(DynamicModel.MB) hands it every MB parameter two slots early
# (m_s <- j_max, ...). mb_vector drops those two fields, which matches every index the MB
# code reads, and keeps float64 (to_array casts to float32).
_SKIP = ('collision_body_center_x', 'collision_body_center_y')


def mb_vector(params):
    names = [f.name for f in fields(params)]
    values = astuple(params)
    return np.array([v for n, v in zip(names, values) if n not in _SKIP], dtype=np.float64)


def mb_index(name, params=F1TENTH_VEHICLE_PARAMETERS):
    return [f.name for f in fields(params) if f.name not in _SKIP].index(name)
