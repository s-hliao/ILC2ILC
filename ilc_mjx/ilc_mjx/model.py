"""The Go1 as ilc_quad builds it (QuadModel: torque actuators, canonical joint order), on the GPU."""
from __future__ import annotations

import numpy as np

from . import MENAGERIE


def quad_model(robot="go1", box=None):
    """box: dict(x_front, height[, length, width]) -- a static box in the world (QuadModel's, as the CPU robots
    have it); JumpEnv moves and resizes it per jump."""
    from ilc_quad.sim_quad_model import QuadModel
    return QuadModel(robot, MENAGERIE, box=box)


def mjx_model(qm, feet_only_contacts=False, iterations=None, ls_iterations=None, legs_hit_box=False):
    """The compiled model on the device (and the CPU twin it was made from). feet_only_contacts: collisions
    only between the foot spheres and the terrain (the floor, and the box if the model has one: the jump's
    contacts; the trunk, thighs and calves never meet the ground in a jump that lands) -- the GPU's collision
    cost scales with the candidate pairs.
    iterations / ls_iterations: the constraint solver's budget (MJX runs every iteration of every env).
    legs_hit_box (with feet_only_contacts and a box): the thigh and calf capsules and the trunk's box also collide
    with the box -- not with the floor or each other -- as on the CPU robots, whose full collision model makes the
    box's edge something a leg can catch."""
    from mujoco import mjx
    m = _copy(qm.model)
    if iterations:
        m.opt.iterations = int(iterations)
    if ls_iterations:
        m.opt.ls_iterations = int(ls_iterations)
    if feet_only_contacts:
        keep = set(int(g) for g in qm.foot_geom_ids) | set(int(g) for g in qm.terrain_geom_ids)
        import mujoco
        box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
        for g in range(m.ngeom):
            if g in keep:
                continue
            leg = legs_hit_box and box >= 0 and m.geom_contype[g] and (
                m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
                or (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_bodyid[g] == qm.base_body_id))
            # bit 2: meets only the box (whose conaffinity takes it); the feet and the floor stay bit 1
            m.geom_contype[g] = 2 if leg else 0
            m.geom_conaffinity[g] = 0
        if legs_hit_box and box >= 0:
            m.geom_conaffinity[box] = 3
    return m, mjx.put_model(m)


def _copy(m):
    import copy
    return copy.deepcopy(m)


def joint_index(qm):
    """qpos / qvel addresses of the 12 joints and the actuator of each, canonical order."""
    return np.asarray(qm.qpos_adr), np.asarray(qm.qvel_adr), np.asarray(qm.act_idx)
