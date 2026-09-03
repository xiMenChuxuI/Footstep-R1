"""Small NumPy geometry helpers using MuJoCo's wxyz quaternion convention."""

import numpy as np


def quat_conjugate(quaternion):
    quaternion = np.asarray(quaternion)
    result = quaternion.copy()
    result[..., 1:] *= -1.0
    return result


def quat_multiply(lhs, rhs):
    lw, lx, ly, lz = np.moveaxis(np.asarray(lhs), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(rhs), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quat_rotate(quaternion, vector):
    quaternion = np.asarray(quaternion)
    vector = np.asarray(vector)
    zeros = np.zeros(vector.shape[:-1] + (1,), dtype=vector.dtype)
    pure = np.concatenate((zeros, vector), axis=-1)
    return quat_multiply(
        quat_multiply(quaternion, pure), quat_conjugate(quaternion)
    )[..., 1:]


def quat_rotate_inverse(quaternion, vector):
    return quat_rotate(quat_conjugate(quaternion), vector)


def yaw_from_quat(quaternion):
    forward = quat_rotate(quaternion, np.array([1.0, 0.0, 0.0]))
    return np.arctan2(forward[..., 1], forward[..., 0])


def yaw_quat(yaw):
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def world_to_stance_foot_goal(
    target_positions, target_yaws, stance_position, stance_quaternion, stance_yaw
):
    """Return left/right [x, y, z, sin(yaw), cos(yaw)] stance-frame goals."""
    relative = np.asarray(target_positions) - np.asarray(stance_position)
    position = np.stack(
        [quat_rotate_inverse(stance_quaternion, relative[index]) for index in range(2)]
    )
    yaw = wrap_to_pi(np.asarray(target_yaws) - stance_yaw)
    return np.concatenate(
        (position, np.sin(yaw)[:, None], np.cos(yaw)[:, None]), axis=1
    ).reshape(10)
