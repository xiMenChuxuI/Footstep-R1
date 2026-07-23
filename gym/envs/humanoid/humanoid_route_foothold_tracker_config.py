"""Configuration for Unitree R1 smooth random-route foothold tracking."""

from gym.envs.humanoid.humanoid_random_foothold_tracker_config import (
    R1RandomFootholdTrackerCfg,
    R1RandomFootholdTrackerRunnerCfg,
)


class R1RouteFootholdTrackerCfg(R1RandomFootholdTrackerCfg):
    class foothold_goal(R1RandomFootholdTrackerCfg.foothold_goal):
        yaw_offset_deg = [-3.0, 3.0]

    class random_route:
        speed = 0.4
        rate_smoothing = 0.8
        boundary_reflection = 0.5
        heading_bound_deg = [10.0, 30.0, 45.0]
        heading_noise_deg = [2.0, 4.0, 6.0]
        heading_rate_limit_deg = [4.0, 8.0, 12.0]
        curriculum_step_thresholds = [12000, 24000]
        history_length = 24

    class viewer(R1RandomFootholdTrackerCfg.viewer):
        route_heading_arrow_length = 0.35

    class rewards(R1RandomFootholdTrackerCfg.rewards):
        route_heading_sigma = 0.25

        class weights(R1RandomFootholdTrackerCfg.rewards.weights):
            base_heading = 0.0
            route_heading_tracking = 4.0


class R1RouteFootholdTrackerRunnerCfg(R1RandomFootholdTrackerRunnerCfg):
    class runner(R1RandomFootholdTrackerRunnerCfg.runner):
        experiment_name = "Unitree_R1_Foothold_Tracker"
        run_name = "r1_random_route_curriculum_1500"
