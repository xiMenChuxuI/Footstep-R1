"""Velocity-derived foothold goals with phase-latched XY perturbations."""

import torch

from gym.envs.guidance.foothold_goal_sampler import FootholdGoalGenerator
from gym.envs.guidance.stance_foot_adapter import (
    LEFT_FOOT,
    world_to_stance_foot_goal,
)
from gym.utils.math import wrap_to_pi


class PerturbedFootholdGoalSampler(FootholdGoalGenerator):
    """Add bounded stance-frame XY jitter to velocity-derived nominal goals."""

    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        n, device = env.num_envs, env.device
        self.goal = torch.zeros(n, 10, device=device)
        self.world_target_pos = torch.zeros(n, 2, 3, device=device)
        self.world_target_yaw = torch.zeros(n, 2, device=device)
        self.nominal_world_target_pos = torch.zeros(n, 2, 3, device=device)
        self.nominal_world_target_yaw = torch.zeros(n, 2, device=device)
        self.foothold_jitter = torch.zeros(n, 2, device=device)
        self.theta_feet = torch.zeros(n, device=device)
        self.hold_still = torch.zeros(n, dtype=torch.bool, device=device)
        self.support_platform_generator = None

    def _uniform(self, low, high, count):
        return low + (high - low) * torch.rand(count, device=self.env.device)

    def _canonical_foot_pose(self, env_ids):
        # HumanoidController stores right=0, left=1; foothold tasks use left=0.
        states = self.env.foot_states[env_ids][:, [1, 0]]
        yaw = self.env.foot_heading[env_ids][:, [1, 0]]
        return states[:, :, :3], states[:, :, 3:7], yaw

    def reset(self, env_ids):
        count = env_ids.numel()
        feet = self.cfg.feet_direction_deg
        self.theta_feet[env_ids] = torch.deg2rad(
            self._uniform(feet[0], feet[1], count)
        )
        mask = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        mask[env_ids] = True
        self.update(env_ids, mask)

    def _sample_target_yaw(self, selected, count):
        yaw_offset = torch.deg2rad(
            self._uniform(*self.cfg.yaw_offset_deg, count)
        )
        return wrap_to_pi(self.theta_feet[selected] + yaw_offset)

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

        step_time = self.env.step_period[selected, 0] * self.env.dt
        world_dx = self.env.commands[selected, 0] * step_time
        world_dy = self.env.commands[selected, 1] * step_time
        c, s = torch.cos(stance_yaw), torch.sin(stance_yaw)
        nominal_x = c * world_dx + s * world_dy
        command_y = -s * world_dx + c * world_dy
        nominal_y = torch.where(
            swing == LEFT_FOOT,
            self.cfg.nominal_step_width + command_y,
            -self.cfg.nominal_step_width + command_y,
        )

        jitter_x = self._uniform(*self.cfg.jitter_x_range, count)
        jitter_y = self._uniform(*self.cfg.jitter_y_range, count)
        self.foothold_jitter[selected] = torch.stack((jitter_x, jitter_y), dim=-1)

        local_x = (nominal_x + jitter_x).clamp(
            self.cfg.reach_x[0], self.cfg.reach_x[1]
        )
        local_y = (nominal_y + jitter_y).clamp(
            self.cfg.reach_y[0], self.cfg.reach_y[1]
        )
        local_y = torch.where(
            swing == LEFT_FOOT,
            local_y.clamp(min=self.cfg.min_lateral_separation),
            local_y.clamp(max=-self.cfg.min_lateral_separation),
        )

        height = self._uniform(*self.cfg.height_range, count)
        nominal_local = torch.stack(
            (nominal_x, nominal_y, torch.zeros_like(nominal_x)), dim=-1
        )
        sampled_local = torch.stack((local_x, local_y, height), dim=-1)

        def stance_to_world(local):
            return stance_pos + torch.stack(
                (
                    c * local[:, 0] - s * local[:, 1],
                    s * local[:, 0] + c * local[:, 1],
                    local[:, 2],
                ),
                dim=-1,
            )

        nominal_pos = stance_to_world(nominal_local)
        sampled_pos = stance_to_world(sampled_local)
        sampled_yaw = self._sample_target_yaw(selected, count)
        self.hold_still[selected] = (
            torch.rand(count, device=self.env.device) < self.cfg.hold_probability
        )

        self.world_target_pos[selected] = position
        self.world_target_yaw[selected] = yaw
        self.nominal_world_target_pos[selected] = position
        self.nominal_world_target_yaw[selected] = yaw

        current_swing_pos = position[batch, swing]
        current_swing_yaw = yaw[batch, swing]
        active_pos = torch.where(
            self.hold_still[selected, None], current_swing_pos, sampled_pos
        )
        active_yaw = torch.where(
            self.hold_still[selected], current_swing_yaw, sampled_yaw
        )
        self.world_target_pos[selected, swing] = active_pos
        self.world_target_yaw[selected, swing] = active_yaw
        self.nominal_world_target_pos[selected, swing] = nominal_pos
        self.nominal_world_target_yaw[selected, swing] = sampled_yaw

        self.goal[selected] = world_to_stance_foot_goal(
            self.world_target_pos[selected],
            self.world_target_yaw[selected],
            stance_pos,
            stance_quat,
            stance_yaw,
        )

    def get_goal(self):
        return self.goal
