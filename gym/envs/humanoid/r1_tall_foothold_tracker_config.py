"""Configuration for fine-tuning the R1 into a taller walking posture."""

from gym.envs.humanoid.humanoid_foothold_tracker_config import (
    R1FootholdTrackerCfg,
    R1FootholdTrackerRunnerCfg,
)


class R1TallFootholdTrackerCfg(R1FootholdTrackerCfg):
    class init_state(R1FootholdTrackerCfg.init_state):
        pos = [0.0, 0.0, 0.725]
        root_pos_range = [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.722, 0.728],
            [-0.15, 0.15],
            [-0.15, 0.15],
            [-0.15, 0.15],
        ]

    class rewards(R1FootholdTrackerCfg.rewards):
        base_height_target = 0.72
        base_height_range = 0.01
        tall_height_sigma = 0.0004
        stance_knee_target = 0.50
        stance_knee_sigma = 0.04

        class weights(R1FootholdTrackerCfg.rewards.weights):
            base_height = 0.0
            base_height_tall = 2.0
            stance_knee_extension = 0.5
            base_z_orientation = 3.0


class R1TallFootholdTrackerRunnerCfg(R1FootholdTrackerRunnerCfg):
    class runner(R1FootholdTrackerRunnerCfg.runner):
        experiment_name = "Unitree_R1_Foothold_Tracker"
        run_name = "r1_tall_posture_stage1"
