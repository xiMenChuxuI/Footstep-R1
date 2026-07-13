"""Batched foothold goal generation for Isaac Gym humanoid tasks."""

from abc import ABC, abstractmethod

import torch

from gym.envs.guidance.stance_foot_adapter import (
    LEFT_FOOT,
    RIGHT_FOOT,
    world_to_stance_foot_goal,
)
from gym.utils.math import wrap_to_pi


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
        mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        mask[env_ids] = True
        self.update(env_ids, mask)

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
        direction_offset = torch.deg2rad(
            self._uniform(*self.cfg.direction_offset_deg, count)
        )
        yaw_offset = torch.deg2rad(self._uniform(*self.cfg.yaw_offset_deg, count))
        height = self._uniform(*self.cfg.height_range, count)
        self.hold_still[selected] = torch.rand(count, device=self.env.device) < self.cfg.hold_probability

        world_direction = wrap_to_pi(self.theta_move[selected] + direction_offset)
        local_direction = wrap_to_pi(world_direction - stance_yaw)
        local_x = length * torch.cos(local_direction)
        local_y = length * torch.sin(local_direction)
        local_x = local_x.clamp(self.cfg.reach_x[0], self.cfg.reach_x[1])
        local_y = local_y.clamp(self.cfg.reach_y[0], self.cfg.reach_y[1])
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

