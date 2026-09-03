"""Deterministic flat-ground goal generator for the first Sim2Sim stage."""

from dataclasses import dataclass

import numpy as np

from .math_utils import world_to_stance_foot_goal, yaw_from_quat


LEFT_FOOT = 0
RIGHT_FOOT = 1


@dataclass
class FootPose:
    position: np.ndarray
    quaternion: np.ndarray

    @property
    def yaw(self):
        return float(yaw_from_quat(self.quaternion))


class FixedFootholdGenerator:
    """Latch midpoint training goals at each left/right phase transition."""

    def __init__(self, half_cycle_steps=30, step_length=0.12, step_width=0.20):
        self.half_cycle_steps = int(half_cycle_steps)
        self.full_cycle_steps = 2 * self.half_cycle_steps
        self.step_length = float(step_length)
        self.step_width = float(step_width)
        self.phase_count = 0
        self.phase = 0.0
        self.swing_foot_index = LEFT_FOOT
        self.stance_foot_index = RIGHT_FOOT
        self.previous_swing_foot_index = LEFT_FOOT
        self.phase_switch = False
        self.world_target_pos = np.zeros((2, 3), dtype=np.float64)
        self.world_target_yaw = np.zeros(2, dtype=np.float64)
        self.goal = np.zeros(10, dtype=np.float32)

    def reset(self, feet):
        self.phase_count = 0
        self.phase = 0.0
        self.swing_foot_index = LEFT_FOOT
        self.stance_foot_index = RIGHT_FOOT
        self.previous_swing_foot_index = LEFT_FOOT
        self.phase_switch = False
        self._latch(feet)
        return self.goal.copy()

    def advance(self, feet):
        self.phase_count = (self.phase_count + 1) % self.full_cycle_steps
        self.phase = self.phase_count / self.full_cycle_steps

        previous = self.swing_foot_index
        self.swing_foot_index = LEFT_FOOT if self.phase < 0.5 else RIGHT_FOOT
        self.stance_foot_index = 1 - self.swing_foot_index
        self.phase_switch = self.swing_foot_index != previous
        self.previous_swing_foot_index = previous
        if self.phase_switch:
            self._latch(feet)
        return self.phase_switch

    def _latch(self, feet):
        positions = np.stack([foot.position for foot in feet])
        yaws = np.array([foot.yaw for foot in feet])
        self.world_target_pos[:] = positions
        self.world_target_yaw[:] = yaws

        stance = feet[self.stance_foot_index]
        swing = self.swing_foot_index
        lateral = self.step_width if swing == LEFT_FOOT else -self.step_width
        stance_yaw = stance.yaw
        cosine, sine = np.cos(stance_yaw), np.sin(stance_yaw)
        local_delta = np.array([self.step_length, lateral, 0.0])
        world_delta = np.array(
            [
                cosine * local_delta[0] - sine * local_delta[1],
                sine * local_delta[0] + cosine * local_delta[1],
                0.0,
            ]
        )
        self.world_target_pos[swing] = stance.position + world_delta
        self.world_target_yaw[swing] = 0.0
        self.goal = world_to_stance_foot_goal(
            self.world_target_pos,
            self.world_target_yaw,
            stance.position,
            stance.quaternion,
            stance_yaw,
        ).astype(np.float32)

    @property
    def phase_sin_cos(self):
        angle = 2.0 * np.pi * self.phase
        return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
