"""Configuration for terrain-aware Unitree R1 foothold tracking."""

from gym.envs.humanoid.humanoid_foothold_tracker_config import (
    R1FootholdTrackerCfg,
    R1FootholdTrackerRunnerCfg,
)


class R1RoughFootholdTrackerCfg(R1FootholdTrackerCfg):
    class terrain(R1FootholdTrackerCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = True
        curriculum = True
        selected = False
        terrain_kwargs = None
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 10.0
        terrain_length = 8.0
        terrain_width = 8.0
        platform_size = 3.0
        num_rows = 10
        num_cols = 20
        max_init_terrain_level = 0
        max_difficulty = 0.45
        difficulty = 0.3
        # slope, rough, stairs down, stairs up, obstacles, stones, gaps
        terrain_proportions = [0.25, 0.35, 0.15, 0.20, 0.05, 0.0, 0.0]

    class foothold_goal(R1FootholdTrackerCfg.foothold_goal):
        num_candidates = 8
        support_patch_half_extents = [0.06, 0.035]
        max_support_height_variation = 0.035
        max_step_height = 0.12


class R1RoughFootholdTrackerRunnerCfg(R1FootholdTrackerRunnerCfg):
    class runner(R1FootholdTrackerRunnerCfg.runner):
        experiment_name = "Unitree_R1_Foothold_Tracker"
        run_name = "r1_rough_curriculum_stage1"
