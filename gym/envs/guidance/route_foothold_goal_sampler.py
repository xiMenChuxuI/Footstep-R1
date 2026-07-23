"""Foothold goals whose position and yaw follow a smooth random route."""

import torch

from gym.envs.guidance.perturbed_foothold_goal_sampler import (
    PerturbedFootholdGoalSampler,
)
from gym.utils.math import wrap_to_pi


class RouteFootholdGoalSampler(PerturbedFootholdGoalSampler):
    """Use the route tangent as the desired foot heading."""

    def _sample_target_yaw(self, selected, count):
        yaw_offset = torch.deg2rad(
            self._uniform(*self.cfg.yaw_offset_deg, count)
        )
        return wrap_to_pi(self.env.route_heading[selected] + yaw_offset)
