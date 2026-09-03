"""Higher-standing R1 foothold tracker without changing actor observations."""

import torch

from gym.envs.guidance.stance_foot_adapter import LEFT_FOOT
from gym.envs.humanoid.humanoid_foothold_tracker import R1FootholdTracker


class R1TallFootholdTracker(R1FootholdTracker):
    """Encourage a taller pelvis and a less crouched stance leg."""

    def _reward_base_height_tall(self):
        error = torch.clamp(
            torch.abs(self.base_height.flatten() - self.cfg.rewards.base_height_target)
            - self.cfg.rewards.base_height_range,
            min=0.0,
        )
        return torch.exp(-error.square() / self.cfg.rewards.tall_height_sigma)

    def _reward_stance_knee_extension(self):
        knee_indices = torch.where(
            self.stance_foot_index == LEFT_FOOT,
            torch.full_like(self.stance_foot_index, 3),
            torch.full_like(self.stance_foot_index, 9),
        )
        batch = torch.arange(self.num_envs, device=self.device)
        error = self.dof_pos[batch, knee_indices] - self.cfg.rewards.stance_knee_target
        return torch.exp(-error.square() / self.cfg.rewards.stance_knee_sigma)
