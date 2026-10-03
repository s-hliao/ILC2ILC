"""The Go1 as ilc_quad builds it (QuadModel: torque actuators, canonical joint order), on the GPU."""
from __future__ import annotations

import numpy as np

from . import MENAGERIE


def quad_model(robot="go1"):
    from ilc_quad.sim_quad_model import QuadModel
    return QuadModel(robot, MENAGERIE)


def mjx_model(qm, feet_only_contacts=False, iterations=None, ls_iterations=None):
    """The compiled model on the device (and the CPU twin it was made from). feet_only_contacts: collisions
    only between the foot spheres and the floor (the jump's contacts; the trunk, thighs and calves never
    meet the ground in a jump that lands) -- the GPU's collision cost scales with the candidate pairs.
    iterations / ls_iterations: the constraint solver's budget (MJX runs every iteration of every env)."""
    from mujoco import mjx
    m = _copy(qm.model)
    if iterations:
        m.opt.iterations = int(iterations)
    if ls_iterations:
        m.opt.ls_iterations = int(ls_iterations)
    if feet_only_contacts:
        keep = set(int(g) for g in qm.foot_geom_ids) | {int(qm.floor_geom_id)}
        for g in range(m.ngeom):
            if g not in keep:
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
    return m, mjx.put_model(m)


def _copy(m):
    import copy
    return copy.deepcopy(m)


def joint_index(qm):
    """qpos / qvel addresses of the 12 joints and the actuator of each, canonical order."""
    return np.asarray(qm.qpos_adr), np.asarray(qm.qvel_adr), np.asarray(qm.act_idx)
