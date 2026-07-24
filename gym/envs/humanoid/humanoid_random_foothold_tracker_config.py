"""Configuration for the Unitree R1 random foothold tracking task."""

from gym.envs.humanoid.humanoid_foothold_tracker_config import (
    R1FootholdTrackerCfg,
    R1FootholdTrackerRunnerCfg,
)


class R1RandomFootholdTrackerCfg(R1FootholdTrackerCfg):
    class foothold_goal(R1FootholdTrackerCfg.foothold_goal):
        nominal_step_width = 0.20
        jitter_x_range = [-0.06, 0.06]
        jitter_y_range = [-0.08, 0.08]


class R1RandomFootholdTrackerRunnerCfg(R1FootholdTrackerRunnerCfg):
    class runner(R1FootholdTrackerRunnerCfg.runner):
        experiment_name = "Unitree_R1_Foothold_Tracker"
        run_name = "r1_random_foothold_xy_stage2_expanded"
