"""MuJoCo model layer for the Menagerie Unitree quadrupeds, normalized for ILC.

The point of this module is that the rest of the package never has to know which
robot it is driving. Two Menagerie models that are structurally identical --
`nq=19 nv=18 nu=12`, 2 ms timestep, a `home` keyframe at 0.27 m -- still differ in
three ways that will silently corrupt a controller:

    actuators   Go2 carries `motor` actuators (direct torque, limits in
                `ctrlrange`); Go1 carries `position` servos with kp=100 and its
                torque limits in `forcerange`. Commanding one as if it were the
                other is the difference between a torque and an angle.
    leg order   Go2's actuators run FL, FR, RL, RR; Go1's run FR, FL, RR, RL. So
                `data.ctrl[0]` is a different leg on the two robots, and a
                controller written against raw indices swaps the front legs when
                you change robot -- which looks like a tuning problem, not a
                indexing bug.
    base body   Go2 calls it `base`, Go1 calls it `trunk`.

`QuadModel` resolves all three by reading the compiled model, so there are no
per-robot constant tables here to drift out of date. Everything it exposes is in
one canonical order (`CANONICAL_JOINT_NAMES`) that both robots are mapped into.

Actuators are normalized to direct torque, Go1's position servos included: ILC on
a jump needs authority over the joint torque, and a servo in the loop is a second
controller fighting the one you are trying to learn. The conversion happens on the
compiled `MjModel` rather than in the MJCF, so the Menagerie checkout stays
read-only and untouched.

The joint limits differ too (Go1's knee stops at -0.888 where Go2's reaches
-0.838, and its hip travel is 0.863 against Go2's 1.047), which is why
`joint_range` is exposed: a jump trajectory written as fixed angles for one robot
will drive the other into its limits, where it collapses on landing. Clip against
this, not against numbers in a trajectory file.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

# The order every (12,) array in this package is in. Legs front/rear x left/right,
# and within a leg the chain from the body out: hip abduction, thigh, knee.
CANONICAL_LEGS = ("FL", "FR", "RL", "RR")
CANONICAL_JOINTS = ("hip", "thigh", "calf")
CANONICAL_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint" for leg in CANONICAL_LEGS for joint in CANONICAL_JOINTS
)
NU = len(CANONICAL_JOINT_NAMES)

SUPPORTED_ROBOTS = ("go2", "go1")

# Whichever of these the model defines is the floating base. Go2 uses the first,
# Go1 the second.
_BASE_BODY_NAMES = ("base", "trunk")

# The foot collision geoms are named for the leg alone -- `FL`, not `FL_foot` --
# on both robots: the condim=6 priority=1 spheres at the end of each calf.
_FLOOR_GEOM = "floor"

_MENAGERIE_ENV = "MUJOCO_MENAGERIE_PATH"
_MENAGERIE_FALLBACK = "/mujoco_menagerie"


def default_menagerie_root() -> str:
    """Where to look for the Menagerie checkout absent an explicit path."""
    return os.environ.get(_MENAGERIE_ENV, _MENAGERIE_FALLBACK)


class QuadModel:
    """A Menagerie Unitree quadruped with torque actuators and canonical ordering.

    Parameters
    ----------
    robot : "go2" or "go1".
    menagerie_root : path to a mujoco_menagerie checkout. Defaults to
        `$MUJOCO_MENAGERIE_PATH`, then `/mujoco_menagerie`.

    Attributes
    ----------
    model, data : the usual MuJoCo pair. `data` starts at the `home` keyframe.
    act_idx, qpos_adr, qvel_adr : (12,) int, canonical order -> model indices.
        Index with these rather than slicing, so Go1's leg order is handled.
    joint_range : (12, 2) float, the MJCF joint limits in canonical order.
    torque_limit : (12,) float, the symmetric per-joint torque bound.
    base_body_id : the floating base body.
    foot_geom_ids : (4,) int, the foot spheres in `CANONICAL_LEGS` order.
    """

    def __init__(self, robot: str = "go2", menagerie_root: str | None = None):
        if robot not in SUPPORTED_ROBOTS:
            raise ValueError(
                f"unsupported robot {robot!r}; expected one of {SUPPORTED_ROBOTS}"
            )
        self.robot = robot
        self.menagerie_root = menagerie_root or default_menagerie_root()

        self.scene_path = os.path.join(
            self.menagerie_root, f"unitree_{robot}", "scene.xml"
        )
        if not os.path.exists(self.scene_path):
            raise FileNotFoundError(
                f"no MJCF at {self.scene_path}. Point menagerie_root (or "
                f"${_MENAGERIE_ENV}) at a mujoco_menagerie checkout."
            )

        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        torque_limit_model_order = self._to_torque_actuators()
        self._build_index_maps()
        # Out of model order into canonical order, so every (12,) array this
        # class hands out agrees.
        self.torque_limit = torque_limit_model_order[self.act_idx]
        self.reset_home()

    # -- setup ---------------------------------------------------------------

    def _to_torque_actuators(self) -> np.ndarray:
        """Rewrite every actuator into a direct torque motor, in place.

        A MuJoCo actuator's force is `gain(ctrl) * ctrl + bias(...)`. A position
        servo is the affine-bias case: `kp * ctrl - kp * q - kv * v`. Zeroing the
        bias and fixing the gain at 1 leaves `force = ctrl`, which is a torque
        motor -- the same thing Go2's MJCF declares directly.

        Returns the symmetric per-actuator torque bound, in *model* order. Where
        the actuator was a servo the bound comes from `forcerange` (ctrlrange
        being an angle there); where it was already a motor, from `ctrlrange`.
        """
        m = self.model
        limits = np.zeros(m.nu)

        for i in range(m.nu):
            is_servo = m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE

            if is_servo:
                lo, hi = m.actuator_forcerange[i]
            else:
                lo, hi = m.actuator_ctrlrange[i]
            if hi <= lo:
                raise ValueError(
                    f"actuator {i} ({_act_name(m, i)}) declares no usable torque "
                    f"limit: forcerange={tuple(m.actuator_forcerange[i])} "
                    f"ctrlrange={tuple(m.actuator_ctrlrange[i])}"
                )
            limits[i] = min(abs(lo), abs(hi))

            m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_gainprm[i] = 0.0
            m.actuator_gainprm[i, 0] = 1.0
            m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
            m.actuator_biasprm[i] = 0.0
            m.actuator_ctrllimited[i] = 1
            m.actuator_ctrlrange[i] = (-limits[i], limits[i])

        self._assert_torque_actuators()
        return limits

    def _assert_torque_actuators(self) -> None:
        """Fail loudly if the normalization did not take.

        Cheap, and the failure it catches is otherwise invisible: a servo left in
        the loop tracks angles, so the robot still moves and still looks roughly
        plausible -- it is just no longer the plant the controller thinks it is.
        """
        m = self.model
        for i in range(m.nu):
            if m.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_NONE:
                raise AssertionError(
                    f"actuator {_act_name(m, i)} still has bias type "
                    f"{int(m.actuator_biastype[i])}, expected mjBIAS_NONE"
                )
            if m.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED:
                raise AssertionError(
                    f"actuator {_act_name(m, i)} still has gain type "
                    f"{int(m.actuator_gaintype[i])}, expected mjGAIN_FIXED"
                )
            if m.actuator_gainprm[i, 0] != 1.0:
                raise AssertionError(
                    f"actuator {_act_name(m, i)} has gain "
                    f"{m.actuator_gainprm[i, 0]}, expected 1.0"
                )

    def _build_index_maps(self) -> None:
        """Resolve canonical order into model indices, by name."""
        m = self.model

        act_idx, qpos_adr, qvel_adr, joint_range = [], [], [], []
        for joint_name in CANONICAL_JOINT_NAMES:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                raise ValueError(f"{self.robot} has no joint {joint_name!r}")
            qpos_adr.append(m.jnt_qposadr[jid])
            qvel_adr.append(m.jnt_dofadr[jid])
            joint_range.append(m.jnt_range[jid])

            # Actuators are named for the joint without the `_joint` suffix:
            # joint `FL_hip_joint` is driven by actuator `FL_hip`.
            act_name = joint_name.removesuffix("_joint")
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if aid < 0:
                raise ValueError(f"{self.robot} has no actuator {act_name!r}")
            act_idx.append(aid)

        self.act_idx = np.array(act_idx, dtype=int)
        self.qpos_adr = np.array(qpos_adr, dtype=int)
        self.qvel_adr = np.array(qvel_adr, dtype=int)
        self.joint_range = np.array(joint_range, dtype=float)

        self.base_body_id = self._find_base_body()
        self.foot_geom_ids = self._find_foot_geoms()
        self.floor_geom_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, _FLOOR_GEOM
        )
        if self.floor_geom_id < 0:
            raise ValueError(
                f"{self.scene_path} defines no geom {_FLOOR_GEOM!r}; foot contact "
                "forces need the ground geom to pair against"
            )

    def _find_base_body(self) -> int:
        for name in _BASE_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.base_body_name = name
                return bid
        raise ValueError(
            f"{self.robot} has no floating base body; looked for {_BASE_BODY_NAMES}"
        )

    def _find_foot_geoms(self) -> np.ndarray:
        ids = []
        for leg in CANONICAL_LEGS:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, leg)
            if gid < 0:
                raise ValueError(f"{self.robot} has no foot geom {leg!r}")
            ids.append(gid)
        return np.array(ids, dtype=int)

    # -- state ---------------------------------------------------------------

    def reset_home(self) -> None:
        """Reset to the MJCF `home` keyframe with no actuation."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def substeps_for(self, rate_hz: float) -> int:
        """Physics steps per control tick at `rate_hz`, which must be a whole number.

        A control period that is not an exact multiple of the model timestep makes
        the reference index and the physics drift apart over a trial. With a 2 ms
        timestep, 200 Hz wants 2.5 steps and is rejected; 250 Hz wants 2.
        """
        if rate_hz <= 0:
            raise ValueError(f"control rate must be positive, got {rate_hz}")
        exact = 1.0 / (rate_hz * self.dt)
        substeps = int(round(exact))
        if substeps < 1 or abs(exact - substeps) > 1e-9:
            options = ", ".join(f"{1.0 / (k * self.dt):.0f}" for k in (1, 2, 4, 5, 10))
            raise ValueError(
                f"control rate {rate_hz} Hz needs {exact:.4f} physics steps per "
                f"tick, which is not a whole number. With a {self.dt * 1e3:.1f} ms "
                f"timestep, use one of: {options} Hz."
            )
        return substeps

    @property
    def home_qpos(self) -> np.ndarray:
        """The `home` joint configuration, canonical order, (12,)."""
        return self.model.key_qpos[0][self.qpos_adr].copy()

    @property
    def total_mass(self) -> float:
        return float(self.model.body_subtreemass[0])

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.qvel_adr].copy()

    def applied_torques(self) -> np.ndarray:
        """The torque MuJoCo actually developed, canonical order, (12,).

        This is what a controller should report as its realized input: it is
        post-clamp, so it reflects a saturated actuator rather than the command
        that asked for more than the motor has.
        """
        return self.data.actuator_force[self.act_idx].copy()

    def set_torques(self, tau: np.ndarray) -> np.ndarray:
        """Command joint torques, canonical order. Returns the clipped command."""
        tau = np.clip(np.asarray(tau, dtype=float), -self.torque_limit, self.torque_limit)
        self.data.ctrl[self.act_idx] = tau
        return tau

    def base_position(self) -> np.ndarray:
        return self.data.xpos[self.base_body_id].copy()

    def base_quat(self) -> np.ndarray:
        """Base orientation as (w, x, y, z), MuJoCo's convention."""
        return self.data.qpos[3:7].copy()

    def base_velocity(self) -> np.ndarray:
        """Free-joint velocity, (6,), exactly as MuJoCo stores it.

        Mind the frames: MuJoCo keeps a free joint's **linear velocity in the
        world frame** and its **angular velocity in the body frame**. Anything
        that wants a consistent frame (`nav_msgs/Odometry` wants both in the
        child frame) has to rotate one half, and only one half.
        """
        return self.data.qvel[:6].copy()

    def base_velocity_body(self) -> np.ndarray:
        """Free-joint velocity, (6,), both halves in the base frame.

        This is the convention every consumer in this package uses, because it is
        the one `nav_msgs/Odometry` wants and the ROS path has to go through
        Odometry anyway. Rotating only the linear half is the whole content of the
        conversion -- MuJoCo already keeps the angular half body-relative, and
        rotating it a second time yields a twist that looks plausible until the
        robot pitches.
        """
        vel = self.base_velocity()
        conj = np.zeros(4)
        lin_body = np.zeros(3)
        mujoco.mju_negQuat(conj, self.base_quat())
        mujoco.mju_rotVecQuat(lin_body, vel[:3], conj)
        return np.concatenate([lin_body, vel[3:]])

    def foot_normal_forces(self) -> np.ndarray:
        """Ground reaction normal force per foot, canonical leg order, (4,).

        Only foot-against-floor contacts count; a foot resting on another part of
        the robot is not ground contact, and a knee on the floor is not a foot.
        """
        forces = np.zeros(len(CANONICAL_LEGS))
        foot_to_leg = {gid: i for i, gid in enumerate(self.foot_geom_ids)}
        wrench = np.zeros(6)

        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            g1, g2 = contact.geom1, contact.geom2
            if g1 == self.floor_geom_id and g2 in foot_to_leg:
                leg = foot_to_leg[g2]
            elif g2 == self.floor_geom_id and g1 in foot_to_leg:
                leg = foot_to_leg[g1]
            else:
                continue
            mujoco.mj_contactForce(self.model, self.data, c, wrench)
            # wrench[0] is the normal component, in the contact frame.
            forces[leg] += abs(wrench[0])

        return forces

    def __repr__(self) -> str:
        return (
            f"QuadModel({self.robot!r}, base={self.base_body_name!r}, "
            f"mass={self.total_mass:.2f}kg, dt={self.dt * 1e3:.1f}ms, "
            f"act_idx={self.act_idx.tolist()})"
        )


def _act_name(model: mujoco.MjModel, i: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"<{i}>"
