"""Unitree R1 task for tracking randomly perturbed flat-ground footholds."""

from gym.envs.guidance.perturbed_foothold_goal_sampler import (
    PerturbedFootholdGoalSampler,
)
from gym.envs.humanoid.humanoid_foothold_tracker import R1FootholdTracker


class R1RandomFootholdTracker(R1FootholdTracker):
    """Track velocity-derived footholds with phase-latched XY perturbations."""

    def _create_goal_generator(self):
        return PerturbedFootholdGoalSampler(self, self.cfg.foothold_goal)

    def _prepare_foothold_commands(self, env_ids):
        pass

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        self._prepare_foothold_commands(env_ids)
        # Base reset samples commands after _reset_system(). Relatch the first
        # goal so it uses the newly sampled velocity and per-env step period.
        if hasattr(self, "goal_generator"):
            self.goal_generator.reset(env_ids)
            self.foothold_goal[env_ids] = self.goal_generator.get_goal()[env_ids]
