"""Terrain-aware R1 foothold tracking task."""

import torch

from gym.envs.guidance.foothold_goal_sampler import (
    TerrainAwareFootholdGoalSampler,
)
from gym.envs.humanoid.humanoid_foothold_tracker import R1FootholdTracker


class R1RoughFootholdTracker(R1FootholdTracker):
    """Track phase-latched footholds sampled on traversable terrain patches."""

    def _init_buffers(self):
        super()._init_buffers()
        self.goal_generator = TerrainAwareFootholdGoalSampler(
            self, self.cfg.foothold_goal
        )

    def _reward_base_height(self):
        position, _, _, _ = self._canonical_foot_values()
        batch = torch.arange(self.num_envs, device=self.device)
        stance_height = position[batch, self.stance_foot_index, 2]
        relative_height = self.base_pos[:, 2] - stance_height
        error = (
            torch.abs(self.cfg.rewards.base_height_target - relative_height)
            - self.cfg.rewards.base_height_range
        )
        return self._negsqrd_exp(torch.clamp(error, min=0.0))

    def check_termination(self):
        term_contact = torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        )
        self.terminated = torch.any(term_contact > 1.0, dim=1)
        self.terminated |= torch.norm(self.base_lin_vel, dim=-1) > 15.0
        self.terminated |= torch.norm(self.base_ang_vel, dim=-1) > 15.0
        self.terminated |= torch.abs(self.projected_gravity[:, 0]) > 0.9
        self.terminated |= torch.abs(self.projected_gravity[:, 1]) > 0.9

        position, _, _, _ = self._canonical_foot_values()
        batch = torch.arange(self.num_envs, device=self.device)
        stance_height = position[batch, self.stance_foot_index, 2]
        self.terminated |= (self.base_pos[:, 2] - stance_height) < 0.3
        self.timed_out = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.terminated | self.timed_out

    def _reward_phase_foot_clearance(self):
        position_error, _ = self._swing_trajectory_errors()
        return self._tracking_reward(position_error[:, 2].square())
