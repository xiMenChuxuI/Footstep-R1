"""Adapter for latching LIPM foothold outputs in the stance-foot frame."""

import torch

from gym.envs.guidance.foothold_goal_sampler import RandomFootholdGoalSampler
from gym.envs.guidance.stance_foot_adapter import (
    base_to_world_footholds,
    world_to_stance_foot_goal,
)


class LIPMFootholdAdapter(RandomFootholdGoalSampler):
    """Use a planner callback when supplied, otherwise retain random fallback."""

    def __init__(self, env, cfg, planner=None, output_frame="world"):
        super().__init__(env, cfg)
        self.planner = planner
        self.output_frame = output_frame

    def planner_targets_to_world(self, position, yaw, env_ids, output_frame=None):
        output_frame = output_frame or self.output_frame
        if output_frame == "world":
            return position, yaw
        if output_frame != "base":
            raise ValueError(f"Unsupported planner output frame: {output_frame}")
        return base_to_world_footholds(
            position,
            yaw,
            self.env.base_pos[env_ids],
            self.env.base_quat[env_ids],
            self.env.base_heading[env_ids, 0],
        )

    def update(self, env_ids, phase_switch_mask):
        if self.planner is None:
            return super().update(env_ids, phase_switch_mask)

        selected = env_ids[phase_switch_mask[env_ids]]
        if selected.numel() == 0:
            return
        output = self.planner(self.env, selected)
        position, yaw = output["position"], output["yaw"]
        output_frame = output.get("frame", self.output_frame)
        position, yaw = self.planner_targets_to_world(
            position, yaw, selected, output_frame
        )

        foot_position, foot_quaternion, foot_yaw = self._canonical_foot_pose(selected)
        batch = torch.arange(selected.numel(), device=self.env.device)
        stance = self.env.stance_foot_index[selected]
        stance_pos = foot_position[batch, stance]
        stance_quat = foot_quaternion[batch, stance]
        stance_yaw = foot_yaw[batch, stance]
        self.world_target_pos[selected] = position
        self.world_target_yaw[selected] = yaw
        self.goal[selected] = world_to_stance_foot_goal(
            position, yaw, stance_pos, stance_quat, stance_yaw
        )
