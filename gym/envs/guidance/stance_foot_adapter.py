"""Utilities for latching world-frame footholds in the stance-foot frame."""

import torch

from gym.utils.math import quat_rotate, quat_rotate_inverse, wrap_to_pi


LEFT_FOOT = 0
RIGHT_FOOT = 1


def phase_foot_indices(phase, previous_swing=None):
    """Return swing, stance, and switch mask for normalized gait phase."""
    phase_flat = phase.reshape(-1)
    swing = torch.where(
        phase_flat < 0.5,
        torch.full_like(phase_flat, LEFT_FOOT, dtype=torch.long),
        torch.full_like(phase_flat, RIGHT_FOOT, dtype=torch.long),
    )
    stance = 1 - swing
    if previous_swing is None:
        switched = torch.zeros_like(swing, dtype=torch.bool)
    else:
        switched = swing != previous_swing
    return swing, stance, switched


def world_to_stance_foot_goal(
    target_pos_world,
    target_yaw_world,
    stance_pos_world,
    stance_quat_world,
    stance_yaw_world,
):
    """Return left/right goals as [x, y, z, sin(yaw), cos(yaw)]."""
    count = target_pos_world.shape[0]
    repeated_quat = stance_quat_world[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
    relative_pos = (target_pos_world - stance_pos_world[:, None, :]).reshape(-1, 3)
    position = quat_rotate_inverse(repeated_quat, relative_pos).reshape(count, 2, 3)
    yaw = wrap_to_pi(target_yaw_world - stance_yaw_world[:, None])
    per_foot = torch.cat(
        (position, torch.sin(yaw).unsqueeze(-1), torch.cos(yaw).unsqueeze(-1)),
        dim=-1,
    )
    return per_foot.reshape(count, 10)


def base_to_world_footholds(position_base, yaw_base, base_pos, base_quat, base_yaw):
    """Convert batched left/right base-frame targets to world coordinates."""
    count = position_base.shape[0]
    repeated_quat = base_quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
    position_world = quat_rotate(repeated_quat, position_base.reshape(-1, 3))
    position_world = position_world.reshape(count, 2, 3) + base_pos[:, None, :]
    return position_world, wrap_to_pi(yaw_base + base_yaw[:, None])
