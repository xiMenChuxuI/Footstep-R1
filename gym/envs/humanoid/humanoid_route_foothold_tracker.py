"""Unitree R1 task for foothold tracking along smooth random routes."""

import torch
from isaacgym import gymutil

from gym.envs.guidance.route_foothold_goal_sampler import (
    RouteFootholdGoalSampler,
)
from gym.envs.guidance.smooth_random_route_generator import (
    SmoothRandomRouteGenerator,
)
from gym.envs.humanoid.humanoid_random_foothold_tracker import (
    R1RandomFootholdTracker,
)
from gym.envs.hi_cl_12.hi_cl12_utils import SimpleLineGeometry
from gym.utils.math import wrap_to_pi


class R1RouteFootholdTracker(R1RandomFootholdTracker):
    """Track perturbed footholds while following a smooth random route."""

    def _init_buffers(self):
        super()._init_buffers()
        self.route_generator = SmoothRandomRouteGenerator(
            self, self.cfg.random_route
        )
        self.route_heading = self.route_generator.heading
        self.route_heading_rate = self.route_generator.heading_rate
        self.initial_route_heading = self.route_generator.initial_heading

    def _create_goal_generator(self):
        return RouteFootholdGoalSampler(self, self.cfg.foothold_goal)

    def _prepare_foothold_commands(self, env_ids):
        self.route_generator.reset(env_ids)

    def _before_goal_update(self, switch_ids):
        if switch_ids.numel() == 0:
            return
        # LEFT_FOOT starts a new left/right pair, so both steps share one tangent.
        cycle_ids = switch_ids[self.swing_foot_index[switch_ids] == 0]
        self.route_generator.update(cycle_ids)

    def _reward_route_heading_tracking(self):
        error = wrap_to_pi(
            self.route_heading - self.base_heading.squeeze(1)
        )
        return torch.exp(
            -error.square() / self.cfg.rewards.route_heading_sigma
        )

    def _visualization(self):
        super()._visualization()
        arrow_length = self.cfg.viewer.route_heading_arrow_length
        trail_color = (0.0, 0.8, 0.2)
        arrow_color = (0.0, 1.0, 0.0)

        for env_id in range(self.num_envs):
            start = self.base_pos[env_id].clone()
            start[2] = 0.03
            heading = self.route_heading[env_id]
            end = start + torch.stack(
                (
                    arrow_length * torch.cos(heading),
                    arrow_length * torch.sin(heading),
                    torch.zeros((), device=self.device),
                )
            )
            gymutil.draw_lines(
                SimpleLineGeometry(start, end, color=arrow_color),
                self.gym,
                self.viewer,
                self.envs[env_id],
                pose=None,
            )

            count = int(self.route_generator.history_count[env_id].item())
            next_index = int(
                self.route_generator.history_index[env_id].item()
            )
            first = (
                next_index - count
            ) % self.cfg.random_route.history_length
            for offset in range(max(0, count - 1)):
                a_index = (
                    first + offset
                ) % self.cfg.random_route.history_length
                b_index = (
                    first + offset + 1
                ) % self.cfg.random_route.history_length
                a = self.route_generator.history[env_id, a_index].clone()
                b = self.route_generator.history[env_id, b_index].clone()
                a[2] = b[2] = 0.025
                gymutil.draw_lines(
                    SimpleLineGeometry(a, b, color=trail_color),
                    self.gym,
                    self.viewer,
                    self.envs[env_id],
                    pose=None,
                )
