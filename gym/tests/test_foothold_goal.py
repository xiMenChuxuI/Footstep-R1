import math
import unittest
from types import SimpleNamespace

import isaacgym
import torch

from gym.envs.guidance.foothold_goal_sampler import RandomFootholdGoalSampler
from gym.envs.guidance.stance_foot_adapter import (
    LEFT_FOOT,
    RIGHT_FOOT,
    phase_foot_indices,
    world_to_stance_foot_goal,
)
from gym.envs.humanoid.humanoid_foothold_tracker_config import (
    R1FootholdTrackerCfg,
    R1FootholdTrackerRunnerCfg,
)
from gym.envs.humanoid.humanoid_foothold_tracker import R1FootholdTracker
from gym.utils.math import wrap_to_pi


class MockEnv:
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.swing_foot_index = torch.full((num_envs,), LEFT_FOOT, dtype=torch.long)
        self.stance_foot_index = torch.full((num_envs,), RIGHT_FOOT, dtype=torch.long)
        # Environment order is right then left.
        self.foot_states = torch.zeros(num_envs, 2, 7)
        self.foot_states[:, 0, :3] = torch.tensor([0.0, -0.15, 0.0])
        self.foot_states[:, 1, :3] = torch.tensor([0.0, 0.15, 0.0])
        self.foot_states[:, :, 6] = 1.0
        self.foot_heading = torch.zeros(num_envs, 2)


def sampler_cfg():
    return SimpleNamespace(
        step_length_range=[0.15, 0.35],
        direction_offset_deg=[-20.0, 20.0],
        yaw_offset_deg=[-20.0, 20.0],
        height_range=[0.0, 0.0],
        hold_probability=0.10,
        movement_direction_deg=[-20.0, 20.0],
        feet_direction_deg=[-20.0, 20.0],
        reach_x=[-0.10, 0.40],
        reach_y=[-0.30, 0.30],
        min_lateral_separation=0.12,
    )


class FootholdGoalTests(unittest.TestCase):
    def test_world_to_stance_transform(self):
        targets = torch.tensor([[[1.0, 2.0, 0.0], [2.0, 1.0, 0.0]]])
        target_yaw = torch.tensor([[math.pi / 2.0, math.pi]])
        stance_pos = torch.tensor([[1.0, 1.0, 0.0]])
        stance_quat = torch.tensor([[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])
        goal = world_to_stance_foot_goal(
            targets, target_yaw, stance_pos, stance_quat, torch.tensor([math.pi / 2])
        ).reshape(1, 2, 5)
        torch.testing.assert_close(goal[0, 0, :3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5, rtol=0)
        torch.testing.assert_close(goal[0, 1, :3], torch.tensor([0.0, -1.0, 0.0]), atol=1e-5, rtol=0)

    def test_yaw_wrap(self):
        wrapped = wrap_to_pi(torch.tensor([3.0 * math.pi, -3.0 * math.pi, 0.25]))
        torch.testing.assert_close(wrapped, torch.tensor([math.pi, -math.pi, 0.25]), atol=1e-5, rtol=0)

    def test_phase_switch(self):
        phase = torch.tensor([0.49, 0.50, 0.99, 0.0])
        previous = torch.tensor([LEFT_FOOT, LEFT_FOOT, RIGHT_FOOT, RIGHT_FOOT])
        swing, stance, switched = phase_foot_indices(phase, previous)
        self.assertEqual(swing.tolist(), [LEFT_FOOT, RIGHT_FOOT, RIGHT_FOOT, LEFT_FOOT])
        self.assertEqual(stance.tolist(), [RIGHT_FOOT, LEFT_FOOT, LEFT_FOOT, RIGHT_FOOT])
        self.assertEqual(switched.tolist(), [False, True, False, True])

    def test_goal_is_frozen_without_switch(self):
        env = MockEnv()
        sampler = RandomFootholdGoalSampler(env, sampler_cfg())
        ids = torch.arange(env.num_envs)
        sampler.reset(ids)
        before = sampler.get_goal().clone()
        env.foot_states[:, :, 0] += 10.0
        sampler.update(ids, torch.zeros(env.num_envs, dtype=torch.bool))
        torch.testing.assert_close(sampler.get_goal(), before)

    def test_left_right_switch_and_no_crossing(self):
        env = MockEnv()
        sampler = RandomFootholdGoalSampler(env, sampler_cfg())
        ids = torch.arange(env.num_envs)
        sampler.reset(ids)
        left_goal = sampler.get_goal().reshape(env.num_envs, 2, 5)
        self.assertTrue(torch.all(left_goal[:, LEFT_FOOT, 1] >= 0.12))
        env.swing_foot_index[:] = RIGHT_FOOT
        env.stance_foot_index[:] = LEFT_FOOT
        sampler.update(ids, torch.ones(env.num_envs, dtype=torch.bool))
        right_goal = sampler.get_goal().reshape(env.num_envs, 2, 5)
        self.assertTrue(torch.all(right_goal[:, RIGHT_FOOT, 1] <= -0.12))

    def test_goal_tensor_shape(self):
        env = MockEnv(7)
        sampler = RandomFootholdGoalSampler(env, sampler_cfg())
        sampler.reset(torch.arange(env.num_envs))
        self.assertEqual(tuple(sampler.get_goal().shape), (7, 10))

    def test_r1_task_configuration(self):
        env_cfg = R1FootholdTrackerCfg()
        runner_cfg = R1FootholdTrackerRunnerCfg()
        self.assertIn("unitree_r1_12dof", env_cfg.asset.file)
        self.assertEqual(env_cfg.env.num_actuators, 12)
        self.assertAlmostEqual(env_cfg.init_state.pos[2], 0.715)
        self.assertEqual(env_cfg.init_state.root_pos_range[2], [0.712, 0.718])
        self.assertAlmostEqual(env_cfg.control.actuation_scale, 0.25)
        self.assertAlmostEqual(env_cfg.scaling.clip_actions, 4.0)
        self.assertEqual(env_cfg.foothold_goal.movement_direction_deg, [0.0, 0.0])
        self.assertEqual(env_cfg.foothold_goal.feet_direction_deg, [0.0, 0.0])
        self.assertGreaterEqual(env_cfg.foothold_goal.step_length_range[0], 0.0)
        self.assertEqual(env_cfg.commands.ranges.lin_vel_x, [0.4, 0.4])
        self.assertEqual(env_cfg.asset.end_effectors, [
            "right_ankle_roll_link", "left_ankle_roll_link"
        ])
        self.assertEqual(env_cfg.asset.foot_position_offset, [0.04, 0.0, -0.053])
        self.assertGreater(env_cfg.rewards.weights.stance_contact, 0.0)
        self.assertGreater(env_cfg.rewards.weights.flight, 0.0)
        self.assertGreater(
            env_cfg.rewards.weights.early_swing_contact,
            env_cfg.rewards.weights.stance_contact,
        )
        self.assertGreater(env_cfg.rewards.weights.tracking_lin_vel_world, 0.0)
        self.assertAlmostEqual(env_cfg.rewards.swing_xy_tracking_sigma, 0.02)
        self.assertAlmostEqual(env_cfg.rewards.touchdown_tracking_sigma, 0.01)
        self.assertAlmostEqual(env_cfg.rewards.stance_tracking_sigma, 0.01)
        self.assertGreater(
            env_cfg.rewards.weights.touchdown_accuracy,
            env_cfg.rewards.weights.swing_foot_xy_tracking,
        )
        self.assertNotIn("foot_contact_state", runner_cfg.policy.actor_obs)
        self.assertIn("foot_contact_state", runner_cfg.policy.critic_obs)

    def test_swing_z_endpoint_reward_is_late_phase_only(self):
        tracker = R1FootholdTracker.__new__(R1FootholdTracker)
        tracker.phase = torch.tensor([[0.10], [0.45]])
        tracker.cfg = SimpleNamespace(
            rewards=SimpleNamespace(
                touchdown_phase=0.70,
                swing_z_tracking_sigma=0.01,
            )
        )
        error = torch.tensor([[0.0, 0.0, 0.05], [0.0, 0.0, 0.05]])
        tracker._swing_tracking_errors = lambda: (
            error,
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
        )

        reward = tracker._reward_swing_foot_z_tracking()

        self.assertEqual(reward[0].item(), 0.0)
        self.assertGreater(reward[1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
