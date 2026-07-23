"""Smooth, bounded random routes for batched humanoid environments."""

import torch

from gym.utils.math import quat_apply, wrap_to_pi


class SmoothRandomRouteGenerator:
    """Maintain a curriculum-driven route heading and world velocity."""

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        n, device = env.num_envs, env.device
        self.heading = torch.zeros(n, device=device)
        self.initial_heading = torch.zeros(n, device=device)
        self.heading_rate = torch.zeros(n, device=device)
        self.cycle_count = torch.zeros(n, dtype=torch.long, device=device)
        self.history = torch.zeros(
            n, cfg.history_length, 3, device=device
        )
        self.history_index = torch.zeros(n, dtype=torch.long, device=device)
        self.history_count = torch.zeros(n, dtype=torch.long, device=device)

    def _uniform(self, low, high, count):
        return low + (high - low) * torch.rand(
            count, device=self.env.device
        )

    def curriculum_stage(self):
        stage = 0
        for threshold in self.cfg.curriculum_step_thresholds:
            stage += int(self.env.common_step_counter >= threshold)
        return min(stage, len(self.cfg.heading_bound_deg) - 1)

    def curriculum_values(self):
        stage = self.curriculum_stage()
        return (
            stage,
            self.cfg.heading_bound_deg[stage],
            self.cfg.heading_noise_deg[stage],
            self.cfg.heading_rate_limit_deg[stage],
        )

    def reset(self, env_ids):
        if env_ids.numel() == 0:
            return
        quaternion = self.env.root_states[env_ids, 3:7]
        forward = quat_apply(
            quaternion,
            torch.tensor(
                [1.0, 0.0, 0.0],
                device=self.env.device,
                dtype=quaternion.dtype,
            ).repeat(env_ids.numel(), 1),
        )
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.initial_heading[env_ids] = heading
        self.heading[env_ids] = heading
        self.heading_rate[env_ids] = 0.0
        self.cycle_count[env_ids] = 0

        position = self.env.root_states[env_ids, :3]
        self.history[env_ids] = position[:, None, :]
        self.history_index[env_ids] = 1 % self.cfg.history_length
        self.history_count[env_ids] = 1
        self.apply_commands(env_ids)

    def update(self, env_ids):
        if env_ids.numel() == 0:
            return

        _, bound_deg, noise_deg, rate_limit_deg = self.curriculum_values()
        count = env_ids.numel()
        noise = torch.deg2rad(self._uniform(-noise_deg, noise_deg, count))
        rate_limit = torch.deg2rad(
            torch.tensor(rate_limit_deg, device=self.env.device)
        )
        rate = (
            self.cfg.rate_smoothing * self.heading_rate[env_ids] + noise
        ).clamp(-rate_limit, rate_limit)

        bound = torch.deg2rad(torch.tensor(bound_deg, device=self.env.device))
        relative = wrap_to_pi(
            self.heading[env_ids] + rate - self.initial_heading[env_ids]
        )
        boundary_hit = relative.abs() > bound
        relative = relative.clamp(-bound, bound)
        rate = torch.where(
            boundary_hit, -self.cfg.boundary_reflection * rate, rate
        )

        self.heading_rate[env_ids] = rate
        self.heading[env_ids] = wrap_to_pi(
            self.initial_heading[env_ids] + relative
        )
        self.cycle_count[env_ids] += 1
        self._record_history(env_ids)
        self.apply_commands(env_ids)

    def _record_history(self, env_ids):
        index = self.history_index[env_ids]
        self.history[env_ids, index] = self.env.root_states[env_ids, :3]
        self.history_index[env_ids] = (
            index + 1
        ) % self.cfg.history_length
        self.history_count[env_ids] = torch.clamp(
            self.history_count[env_ids] + 1,
            max=self.cfg.history_length,
        )

    def apply_commands(self, env_ids):
        heading = self.heading[env_ids]
        self.env.commands[env_ids, 0] = self.cfg.speed * torch.cos(heading)
        self.env.commands[env_ids, 1] = self.cfg.speed * torch.sin(heading)
        self.env.commands[env_ids, 2] = 0.0
