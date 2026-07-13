"""Configuration for the Unitree R1 stance-frame foothold tracker."""

from gym.envs.hi_cl_12.r1_12dof_controller_config import (
    R1_12DofControllerCfg,
    R1_12DofControllerRunnerCfg,
)


class R1FootholdTrackerCfg(R1_12DofControllerCfg):
    class env(R1_12DofControllerCfg.env):
        num_envs = 4096
        num_actuators = 12
        episode_length_s = 10
        num_single_obs = 54
        num_observations = 54

    class terrain(R1_12DofControllerCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class commands(R1_12DofControllerCfg.commands):
        curriculum = False
        resampling_time = -1

        class ranges(R1_12DofControllerCfg.commands.ranges):
            sample_period = [30, 31]
            # Flat-ground stage one is intentionally a forward-only curriculum.
            # The fixed value is used by the reward, not exposed to the actor.
            lin_vel_x = [0.4, 0.4]
            lin_vel_y = 0.0
            yaw_vel = 0.0

    class foothold_goal:
        step_length_range = [0.10, 0.22]
        direction_offset_deg = [-5.0, 5.0]
        yaw_offset_deg = [-5.0, 5.0]
        height_range = [0.0, 0.0]
        hold_probability = 0.05
        movement_direction_deg = [0.0, 0.0]
        feet_direction_deg = [0.0, 0.0]
        reach_x = [0.05, 0.30]
        reach_y = [-0.30, 0.30]
        min_lateral_separation = 0.12
        support_platform_generator = None

    class domain_rand(R1_12DofControllerCfg.domain_rand):
        push_robots = False

    class rewards(R1_12DofControllerCfg.rewards):
        foothold_tracking_sigma = 0.04
        touchdown_phase = 0.70
        foot_clearance = 0.07
        swing_knee_height = 0.20

        class weights(R1_12DofControllerCfg.rewards.weights):
            tracking_lin_vel_world = 2.0
            joint_regularization = 0.5
            contact_schedule = 0.0
            swing_foot_xy_tracking = 3.0
            swing_foot_z_tracking = 0.5
            swing_foot_yaw_tracking = 0.5
            touchdown_accuracy = 4.0
            stance_accuracy_latched = 2.0
            stance_contact = 4.0
            flight = 4.0
            # A swing foot that remains planted must cost more than the stance
            # contact reward, otherwise PPO can profit from permanent support.
            early_swing_contact = 8.0
            phase_foot_clearance = 0.5
            swing_knee_height = 0.2
            foot_slip = 0.2
            collision = 1.0
            dof_acc = 1.0e-6


class R1FootholdTrackerRunnerCfg(R1_12DofControllerRunnerCfg):
    class policy(R1_12DofControllerRunnerCfg.policy):
        actor_obs = [
            "projected_gravity",
            "base_ang_vel",
            "dof_pos_error",
            "dof_vel",
            "previous_action",
            "phase_sin_cos",
            "foothold_goal",
        ]
        critic_obs = actor_obs + [
            "base_lin_vel_world",
            "foot_contact_state",
            "world_foot_target_error",
            "foot_contact_force",
        ]
        actions = ["dof_pos_target"]

        class noise(R1_12DofControllerRunnerCfg.policy.noise):
            dof_pos_error = 0.05
            previous_action = 0.0
            phase_sin_cos = 0.0
            foothold_goal = 0.01

    class runner(R1_12DofControllerRunnerCfg.runner):
        experiment_name = "Unitree_R1_Foothold_Tracker"
        run_name = "r1_flat_stance_frame"


# Backward-compatible config names; both aliases configure Unitree R1.
HumanoidFootholdTrackerCfg = R1FootholdTrackerCfg
HumanoidFootholdTrackerRunnerCfg = R1FootholdTrackerRunnerCfg
