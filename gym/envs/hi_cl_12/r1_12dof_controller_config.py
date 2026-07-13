"""Unitree R1 configuration with only the 12 leg joints actuated."""

from gym.envs.hi_cl_12.hi_cl12_controller_config import (
    Hicl12ControllerCfg,
    Hicl12ControllerRunnerCfg,
)


class R1_12DofControllerCfg(Hicl12ControllerCfg):
    class env(Hicl12ControllerCfg.env):
        num_actuators = 12

    class init_state(Hicl12ControllerCfg.init_state):
        pos = [0.0, 0.0, 0.715]
        root_pos_range = [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.712, 0.718],
            [-0.15, 0.15],
            [-0.15, 0.15],
            [-0.15, 0.15],
        ]

        default_joint_angles = {
            "left_hip_pitch_joint": -0.35,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.70,
            "left_ankle_pitch_joint": -0.35,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.35,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.70,
            "right_ankle_pitch_joint": -0.35,
            "right_ankle_roll_joint": 0.0,
        }

        dof_pos_range = {
            "left_hip_pitch_joint": [-1.5, 1.0],
            "left_hip_roll_joint": [-0.6, 0.6],
            "left_hip_yaw_joint": [-0.8, 0.8],
            "left_knee_joint": [0.0, 1.8],
            "left_ankle_pitch_joint": [-0.7, 0.5],
            "left_ankle_roll_joint": [-0.2, 0.2],
            "right_hip_pitch_joint": [-1.5, 1.0],
            "right_hip_roll_joint": [-0.6, 0.6],
            "right_hip_yaw_joint": [-0.8, 0.8],
            "right_knee_joint": [0.0, 1.8],
            "right_ankle_pitch_joint": [-0.7, 0.5],
            "right_ankle_roll_joint": [-0.2, 0.2],
        }

        dof_vel_range = {
            "left_hip_pitch_joint": [-0.1, 0.1],
            "left_hip_roll_joint": [-0.1, 0.1],
            "left_hip_yaw_joint": [-0.1, 0.1],
            "left_knee_joint": [-0.1, 0.1],
            "left_ankle_pitch_joint": [-0.1, 0.1],
            "left_ankle_roll_joint": [-0.1, 0.1],
            "right_hip_pitch_joint": [-0.1, 0.1],
            "right_hip_roll_joint": [-0.1, 0.1],
            "right_hip_yaw_joint": [-0.1, 0.1],
            "right_knee_joint": [-0.1, 0.1],
            "right_ankle_pitch_joint": [-0.1, 0.1],
            "right_ankle_roll_joint": [-0.1, 0.1],
        }

    class control(Hicl12ControllerCfg.control):
        actuation_scale = 0.25
        stiffness = {
            "hip_pitch_joint": 45.0,
            "hip_roll_joint": 45.0,
            "hip_yaw_joint": 35.0,
            "knee_joint": 55.0,
            "ankle_pitch_joint": 30.0,
            "ankle_roll_joint": 25.0,
        }
        damping = {
            "hip_pitch_joint": 2.0,
            "hip_roll_joint": 2.0,
            "hip_yaw_joint": 1.5,
            "knee_joint": 2.5,
            "ankle_pitch_joint": 1.2,
            "ankle_roll_joint": 1.0,
        }

    class asset(Hicl12ControllerCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_r1_12dof/R1_12dof.urdf"
        keypoints = ["pelvis_link"]
        foot_name = "ankle_roll_link"
        end_effectors = ["right_ankle_roll_link", "left_ankle_roll_link"]
        right_hip_link = "right_hip_pitch_link"
        left_hip_link = "left_hip_pitch_link"
        right_foot_link = "right_ankle_roll_link"
        left_foot_link = "left_ankle_roll_link"
        right_knee_link = "right_knee_link"
        left_knee_link = "left_knee_link"
        foot_position_offset = [0.04, 0.0, -0.053]
        terminate_after_contacts_on = [
            "pelvis_link",
            "hip_pitch_link",
            "hip_roll_link",
            "hip_yaw_link",
            "knee_link",
        ]
        collapse_fixed_joints = True
        self_collisions = 1
        apply_humanoid_jacobian = False
        rotor_inertia = [0.01] * 12

    class rewards(Hicl12ControllerCfg.rewards):
        base_height_target = 0.68
        min_dist_feet = 0.10
        max_dist_feet = 0.32
        max_contact_force = 600.0

    class scaling(Hicl12ControllerCfg.scaling):
        clip_actions = 4.0


class R1_12DofControllerRunnerCfg(Hicl12ControllerRunnerCfg):
    class runner(Hicl12ControllerRunnerCfg.runner):
        experiment_name = "Unitree_R1_12Dof_Controller"
