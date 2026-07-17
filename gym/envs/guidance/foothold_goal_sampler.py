"""Batched foothold goal generation for Isaac Gym humanoid tasks."""

from abc import ABC, abstractmethod

import torch

from gym.envs.guidance.stance_foot_adapter import (
    LEFT_FOOT,
    RIGHT_FOOT,
    world_to_stance_foot_goal,
)
from gym.utils.math import wrap_to_pi


def sample_heightfield(
    height_samples,
    world_xy,
    border_size,
    horizontal_scale,
    vertical_scale,
):
    """Sample terrain heights at arbitrary batched world XY coordinates."""
    original_shape = world_xy.shape[:-1]
    points = ((world_xy.reshape(-1, 2) + border_size) / horizontal_scale).long()
    px = points[:, 0].clamp(0, height_samples.shape[0] - 2)
    py = points[:, 1].clamp(0, height_samples.shape[1] - 2)
    heights = torch.minimum(height_samples[px, py], height_samples[px + 1, py])
    heights = torch.minimum(heights, height_samples[px, py + 1])
    return heights.reshape(original_shape) * vertical_scale


class FootholdGoalGenerator(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def reset(self, env_ids):
        pass

    @abstractmethod
    def update(self, env_ids, phase_switch_mask):
        pass

    @abstractmethod
    def get_goal(self):
        pass


class RandomFootholdGoalSampler(FootholdGoalGenerator):
    """Generate safe flat-ground goals and latch them at phase switches."""

    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        n, device = env.num_envs, env.device
        self.goal = torch.zeros(n, 10, device=device)
        self.world_target_pos = torch.zeros(n, 2, 3, device=device)
        self.world_target_yaw = torch.zeros(n, 2, device=device)
        self.theta_move = torch.zeros(n, device=device)
        self.theta_feet = torch.zeros(n, device=device)
        self.route_heading_override = torch.full((n,), torch.nan, device=device)
        self.feet_heading_override = torch.full((n,), torch.nan, device=device)
        self.step_length_scale = torch.ones(n, device=device)
        self.hold_still = torch.zeros(n, dtype=torch.bool, device=device)
        self.support_platform_generator = None

    def _uniform(self, low, high, count):
        return low + (high - low) * torch.rand(count, device=self.env.device)

    def _canonical_foot_pose(self, env_ids):
        # HumanoidController stores right=0, left=1; this task exposes left=0, right=1.
        states = self.env.foot_states[env_ids][:, [1, 0]]
        yaw = self.env.foot_heading[env_ids][:, [1, 0]]
        return states[:, :, :3], states[:, :, 3:7], yaw

    def reset(self, env_ids):
        count = env_ids.numel()
        direction = self.cfg.movement_direction_deg
        feet = self.cfg.feet_direction_deg
        self.theta_move[env_ids] = torch.deg2rad(
            self._uniform(direction[0], direction[1], count)
        )
        self.theta_feet[env_ids] = torch.deg2rad(
            self._uniform(feet[0], feet[1], count)
        )
        route_override = self.route_heading_override[env_ids]
        feet_override = self.feet_heading_override[env_ids]
        self.theta_move[env_ids] = torch.where(
            torch.isfinite(route_override), route_override, self.theta_move[env_ids]
        )
        self.theta_feet[env_ids] = torch.where(
            torch.isfinite(feet_override), feet_override, self.theta_feet[env_ids]
        )
        mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        mask[env_ids] = True
        self.update(env_ids, mask)

    def set_route_overrides(self, env_ids, heading, step_length_scale):
        """Persist per-environment playback routes across episode resets."""
        self.route_heading_override[env_ids] = heading
        self.feet_heading_override[env_ids] = heading
        self.step_length_scale[env_ids] = step_length_scale
        self.theta_move[env_ids] = heading
        self.theta_feet[env_ids] = heading

    def update(self, env_ids, phase_switch_mask):
        selected = env_ids[phase_switch_mask[env_ids]]
        if selected.numel() == 0:
            return

        count = selected.numel()
        position, quaternion, yaw = self._canonical_foot_pose(selected)
        swing = self.env.swing_foot_index[selected]
        stance = self.env.stance_foot_index[selected]
        batch = torch.arange(count, device=self.env.device)
        stance_pos = position[batch, stance]
        stance_quat = quaternion[batch, stance]
        stance_yaw = yaw[batch, stance]

        length = self._uniform(*self.cfg.step_length_range, count)
        length *= self.step_length_scale[selected]
        width = self._uniform(*self.cfg.step_width_range, count)
        direction_offset = torch.deg2rad(
            self._uniform(*self.cfg.direction_offset_deg, count)
        )
        yaw_offset = torch.deg2rad(self._uniform(*self.cfg.yaw_offset_deg, count))
        height = self._uniform(*self.cfg.height_range, count)
        self.hold_still[selected] = torch.rand(count, device=self.env.device) < self.cfg.hold_probability

        world_direction = wrap_to_pi(self.theta_move[selected] + direction_offset)
        local_direction = wrap_to_pi(world_direction - stance_yaw)
        local_x = length * torch.cos(local_direction)
        direction_y = length * torch.sin(local_direction)
        local_x = local_x.clamp(self.cfg.reach_x[0], self.cfg.reach_x[1])
        local_y = torch.where(
            swing == LEFT_FOOT,
            width + direction_y,
            -width + direction_y,
        ).clamp(self.cfg.reach_y[0], self.cfg.reach_y[1])
        local_y = torch.where(
            swing == LEFT_FOOT,
            local_y.clamp(min=self.cfg.min_lateral_separation),
            local_y.clamp(max=-self.cfg.min_lateral_separation),
        )

        c, s = torch.cos(stance_yaw), torch.sin(stance_yaw)
        world_delta = torch.stack(
            (c * local_x - s * local_y, s * local_x + c * local_y, height), dim=-1
        )
        sampled_pos = stance_pos + world_delta
        sampled_yaw = wrap_to_pi(self.theta_feet[selected] + yaw_offset)

        self.world_target_pos[selected] = position
        self.world_target_yaw[selected] = yaw
        current_swing_pos = position[batch, swing]
        current_swing_yaw = yaw[batch, swing]
        swing_pos = torch.where(self.hold_still[selected, None], current_swing_pos, sampled_pos)
        swing_yaw = torch.where(self.hold_still[selected], current_swing_yaw, sampled_yaw)
        self.world_target_pos[selected, swing] = swing_pos
        self.world_target_yaw[selected, swing] = swing_yaw

        self.goal[selected] = world_to_stance_foot_goal(
            self.world_target_pos[selected],
            self.world_target_yaw[selected],
            stance_pos,
            stance_quat,
            stance_yaw,
        )

    def get_goal(self):
        return self.goal


class TerrainAwareFootholdGoalSampler(RandomFootholdGoalSampler):
    """Sample reachable footholds whose sole patch is supported by terrain."""

    def _terrain_heights(self, world_xy):
        terrain = self.env.terrain.cfg
        return sample_heightfield(
            self.env.height_samples,
            world_xy,
            terrain.border_size,
            terrain.horizontal_scale,
            terrain.vertical_scale,
        )

    def update(self, env_ids, phase_switch_mask):
        selected = env_ids[phase_switch_mask[env_ids]]
        if selected.numel() == 0:
            return

        count = selected.numel()
        candidates = self.cfg.num_candidates
        position, quaternion, yaw = self._canonical_foot_pose(selected)
        swing = self.env.swing_foot_index[selected]
        stance = self.env.stance_foot_index[selected]
        batch = torch.arange(count, device=self.env.device)
        stance_pos = position[batch, stance]
        stance_quat = quaternion[batch, stance]
        stance_yaw = yaw[batch, stance]

        length = self._uniform(
            *self.cfg.step_length_range, count * candidates
        ).reshape(count, candidates)
        length *= self.step_length_scale[selected, None]
        width = self._uniform(
            *self.cfg.step_width_range, count * candidates
        ).reshape(count, candidates)
        direction_offset = torch.deg2rad(
            self._uniform(
                *self.cfg.direction_offset_deg, count * candidates
            ).reshape(count, candidates)
        )
        yaw_offset = torch.deg2rad(
            self._uniform(
                *self.cfg.yaw_offset_deg, count * candidates
            ).reshape(count, candidates)
        )
        height_offset = self._uniform(
            *self.cfg.height_range, count * candidates
        ).reshape(count, candidates)

        world_direction = wrap_to_pi(
            self.theta_move[selected, None] + direction_offset
        )
        local_direction = wrap_to_pi(world_direction - stance_yaw[:, None])
        local_x = (length * torch.cos(local_direction)).clamp(
            self.cfg.reach_x[0], self.cfg.reach_x[1]
        )
        direction_y = length * torch.sin(local_direction)
        local_y = torch.where(
            swing[:, None] == LEFT_FOOT,
            width + direction_y,
            -width + direction_y,
        ).clamp(self.cfg.reach_y[0], self.cfg.reach_y[1])
        local_y = torch.where(
            swing[:, None] == LEFT_FOOT,
            local_y.clamp(min=self.cfg.min_lateral_separation),
            local_y.clamp(max=-self.cfg.min_lateral_separation),
        )

        c, s = torch.cos(stance_yaw[:, None]), torch.sin(stance_yaw[:, None])
        world_x = stance_pos[:, None, 0] + c * local_x - s * local_y
        world_y = stance_pos[:, None, 1] + s * local_x + c * local_y
        candidate_xy = torch.stack((world_x, world_y), dim=-1)
        candidate_yaw = wrap_to_pi(self.theta_feet[selected, None] + yaw_offset)

        half_length, half_width = self.cfg.support_patch_half_extents
        patch_offsets = torch.tensor(
            [
                [0.0, 0.0],
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, half_width],
                [-half_length, -half_width],
            ],
            dtype=candidate_xy.dtype,
            device=self.env.device,
        )
        offset_x = patch_offsets[:, 0]
        offset_y = patch_offsets[:, 1]
        yaw_c = torch.cos(candidate_yaw)[..., None]
        yaw_s = torch.sin(candidate_yaw)[..., None]
        rotated_offsets = torch.stack(
            (
                yaw_c * offset_x - yaw_s * offset_y,
                yaw_s * offset_x + yaw_c * offset_y,
            ),
            dim=-1,
        )
        patch_xy = candidate_xy[:, :, None, :] + rotated_offsets
        patch_heights = self._terrain_heights(patch_xy)
        surface_height = patch_heights[:, :, 0]
        patch_variation = patch_heights.amax(dim=-1) - patch_heights.amin(dim=-1)
        reachable_height = (
            surface_height - stance_pos[:, None, 2]
        ).abs() <= self.cfg.max_step_height
        valid = (patch_variation <= self.cfg.max_support_height_variation) & reachable_height

        has_valid = valid.any(dim=1)
        first_valid = valid.to(torch.int64).argmax(dim=1)
        chosen_xy = candidate_xy[batch, first_valid]
        chosen_yaw = candidate_yaw[batch, first_valid]
        chosen_z = surface_height[batch, first_valid] + height_offset[batch, first_valid]
        chosen_pos = torch.cat((chosen_xy, chosen_z[:, None]), dim=-1)

        random_hold = (
            torch.rand(count, device=self.env.device) < self.cfg.hold_probability
        )
        self.hold_still[selected] = random_hold | ~has_valid
        current_swing_pos = position[batch, swing]
        current_swing_yaw = yaw[batch, swing]
        swing_pos = torch.where(
            self.hold_still[selected, None], current_swing_pos, chosen_pos
        )
        swing_yaw = torch.where(
            self.hold_still[selected], current_swing_yaw, chosen_yaw
        )

        self.world_target_pos[selected] = position
        self.world_target_yaw[selected] = yaw
        self.world_target_pos[selected, swing] = swing_pos
        self.world_target_yaw[selected, swing] = swing_yaw
        self.goal[selected] = world_to_stance_foot_goal(
            self.world_target_pos[selected],
            self.world_target_yaw[selected],
            stance_pos,
            stance_quat,
            stance_yaw,
        )
