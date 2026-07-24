"""Configuration for the Unitree R1 random foothold tracking task."""

from gym.envs.humanoid.humanoid_foothold_tracker_config import (
    R1FootholdTrackerCfg,
    R1FootholdTrackerRunnerCfg,
)


class R1RandomFootholdTrackerCfg(R1FootholdTrackerCfg):
    class foothold_goal(R1FootholdTrackerCfg.foothold_goal):
        nominal_step_width = 0.20
        jitter_x_range = [-0.05, 0.05]
        jitter_y_range = [-0.06, 0.06]
        play_jitter_x = 0.06
        play_jitter_y = 0.08
        precision_profile = "endpoint_stance"

    class rewards(R1FootholdTrackerCfg.rewards):
        late_swing_xy_tracking_sigma = 0.0064
        touchdown_fine_tracking_sigma = 0.0049

        class weights(R1FootholdTrackerCfg.rewards.weights):
            late_swing_foot_xy_tracking = 2.0
            touchdown_accuracy_fine = 20.0
            stance_accuracy_latched = 4.5


class R1RandomFootholdTrackerRunnerCfg(R1FootholdTrackerRunnerCfg):
    class runner(R1FootholdTrackerRunnerCfg.runner):
        experiment_name = "Unitree_R1_Foothold_Tracker"
        run_name = "r1_random_foothold_stage3_precision"
