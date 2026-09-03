"""Official unitree_mujoco R1 state and control adapter."""

from pathlib import Path

import numpy as np

from .foothold import FixedFootholdGenerator, FootPose
from .math_utils import quat_rotate, quat_rotate_inverse


LEG_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

LEG_ACTUATORS = tuple(name.removesuffix("_joint") for name in LEG_JOINTS)


class R1MujocoAdapter:
    def __init__(self, scene_path, config):
        import mujoco

        self.mujoco = mujoco
        self.scene_path = Path(scene_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.config = config
        simulation = config["simulation"]
        self.model.opt.timestep = float(simulation["sim_dt"])
        self.decimation = int(round(simulation["policy_dt"] / simulation["sim_dt"]))
        if self.decimation < 1:
            raise ValueError("policy_dt must be no smaller than sim_dt")

        control = config["control"]
        self.default_q = np.asarray(control["default_q"], dtype=np.float64)
        self.kp = np.asarray(control["kp"], dtype=np.float64)
        self.kd = np.asarray(control["kd"], dtype=np.float64)
        self.torque_limits = np.asarray(control["torque_limits"], dtype=np.float64)
        self.action_scale = float(control["action_scale"])
        self.action_clip = float(control["action_clip"])
        for value in (self.default_q, self.kp, self.kd, self.torque_limits):
            if value.shape != (12,):
                raise ValueError("All leg control arrays must have 12 entries")

        self.leg_qpos, self.leg_dofs = self._joint_addresses(LEG_JOINTS)
        self.leg_actuators = self._actuator_ids(LEG_ACTUATORS)
        self.upper_joint_names = tuple(
            self._name(mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(1, self.model.njnt)
            if self._name(mujoco.mjtObj.mjOBJ_JOINT, index) not in LEG_JOINTS
        )
        self.upper_qpos, self.upper_dofs = self._joint_addresses(self.upper_joint_names)
        self.upper_actuators = self._actuator_ids(
            tuple(self._joint_actuator_name(name) for name in self.upper_joint_names)
        )
        self.upper_home = np.zeros(len(self.upper_joint_names), dtype=np.float64)
        self.upper_kp = float(control.get("upper_body_kp", 30.0))
        self.upper_kd = float(control.get("upper_body_kd", 2.0))
        self.left_foot_body = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )
        self.right_foot_body = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
        )
        self.foot_offset = np.asarray(config["robot"]["foot_position_offset"])
        self.previous_action = np.zeros(12, dtype=np.float32)
        foothold = config["foothold"]
        self.goal_generator = FixedFootholdGenerator(
            foothold["half_cycle_steps"],
            foothold["step_length"],
            foothold["step_width"],
        )
        self.reset()

    def _name(self, object_type, index):
        return self.mujoco.mj_id2name(self.model, object_type, index)

    def _id(self, object_type, name):
        index = self.mujoco.mj_name2id(self.model, object_type, name)
        if index < 0:
            raise ValueError(f"Official R1 model is missing {name}")
        return index

    def _joint_addresses(self, names):
        qpos, dofs = [], []
        for name in names:
            index = self._id(self.mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos.append(int(self.model.jnt_qposadr[index]))
            dofs.append(int(self.model.jnt_dofadr[index]))
        return np.asarray(qpos), np.asarray(dofs)

    def _actuator_ids(self, names):
        return np.asarray(
            [self._id(self.mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
        )

    @staticmethod
    def _joint_actuator_name(joint_name):
        aliases = {
            "waist_roll_joint": "waist_roll",
            "waist_yaw_joint": "waist_yaw",
            "left_shoulder_pitch_joint": "left_shoulder_pitch",
            "left_shoulder_roll_joint": "left_shoulder_roll",
            "left_shoulder_yaw_joint": "left_shoulder_yaw",
            "left_elbow_joint": "left_elbow",
            "left_wrist_roll_joint": "left_wrist_roll",
            "right_shoulder_pitch_joint": "right_shoulder_pitch",
            "right_shoulder_roll_joint": "right_shoulder_roll",
            "right_shoulder_yaw_joint": "right_shoulder_yaw",
            "right_elbow_joint": "right_elbow",
            "right_wrist_roll_joint": "right_wrist_roll",
        }
        return aliases.get(joint_name, joint_name)

    def reset(self):
        self.mujoco.mj_resetData(self.model, self.data)
        initial = self.config["robot"]
        self.data.qpos[:3] = np.asarray(initial["base_position"])
        self.data.qpos[3:7] = np.asarray(initial["base_quaternion_wxyz"])
        self.data.qpos[self.leg_qpos] = self.default_q
        self.data.qpos[self.upper_qpos] = self.upper_home
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.previous_action[:] = 0.0
        self.goal_generator.reset(self.foot_poses())

    def _foot_pose(self, body_id):
        quaternion = self.data.xquat[body_id].copy()
        position = self.data.xpos[body_id] + quat_rotate(
            quaternion, self.foot_offset
        )
        return FootPose(position.copy(), quaternion)

    def foot_poses(self):
        return (
            self._foot_pose(self.left_foot_body),
            self._foot_pose(self.right_foot_body),
        )

    def observation(self):
        base_quaternion = self.data.qpos[3:7].copy()
        projected_gravity = quat_rotate_inverse(
            base_quaternion, np.array([0.0, 0.0, -1.0])
        )
        try:
            base_angular_velocity = self.data.sensor("imu_gyro").data.copy()
        except KeyError:
            base_angular_velocity = quat_rotate_inverse(
                base_quaternion, self.data.qvel[3:6]
            )
        q = self.data.qpos[self.leg_qpos]
        qd = self.data.qvel[self.leg_dofs]
        observation = np.concatenate(
            (
                projected_gravity,
                base_angular_velocity,
                q - self.default_q,
                qd,
                self.previous_action,
                self.goal_generator.phase_sin_cos,
                self.goal_generator.goal,
            )
        ).astype(np.float32)
        if observation.shape != (54,):
            raise RuntimeError(f"Observation shape changed to {observation.shape}")
        return observation

    def apply_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -self.action_clip, self.action_clip)
        if action.shape != (12,):
            raise ValueError(f"Expected 12 actions, got {action.shape}")
        desired_q = self.default_q + self.action_scale * action
        for _ in range(self.decimation):
            q = self.data.qpos[self.leg_qpos]
            qd = self.data.qvel[self.leg_dofs]
            torque = self.kp * (desired_q - q) - self.kd * qd
            self.data.ctrl[self.leg_actuators] = np.clip(
                torque, -self.torque_limits, self.torque_limits
            )
            upper_q = self.data.qpos[self.upper_qpos]
            upper_qd = self.data.qvel[self.upper_dofs]
            upper_torque = self.upper_kp * (self.upper_home - upper_q) - self.upper_kd * upper_qd
            actuator_limits = self.model.actuator_ctrlrange[self.upper_actuators]
            finite_limits = np.where(
                (actuator_limits[:, 1] > actuator_limits[:, 0])[:, None],
                actuator_limits,
                np.array([-25.0, 25.0]),
            )
            self.data.ctrl[self.upper_actuators] = np.clip(
                upper_torque, finite_limits[:, 0], finite_limits[:, 1]
            )
            self.mujoco.mj_step(self.model, self.data)
        self.previous_action[:] = (self.action_scale * action).astype(np.float32)
        self.goal_generator.advance(self.foot_poses())
        return desired_q

    @property
    def has_fallen(self):
        gravity = self.observation()[:3]
        return bool(self.data.qpos[2] < 0.35 or np.linalg.norm(gravity[:2]) > 0.8)
