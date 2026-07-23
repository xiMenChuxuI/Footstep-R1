import math
import unittest
from types import SimpleNamespace

import isaacgym
import torch

from gym.envs.guidance.foothold_goal_sampler import RandomFootholdGoalSampler
from gym.envs.guidance.perturbed_foothold_goal_sampler import (
    PerturbedFootholdGoalSampler,
)
from gym.envs.guidance.route_foothold_goal_sampler import (
    RouteFootholdGoalSampler,
)
from gym.envs.guidance.smooth_random_route_generator import (
    SmoothRandomRouteGenerator,
)
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
from gym.envs.humanoid.humanoid_random_foothold_tracker_config import (
    R1RandomFootholdTrackerCfg,
    R1RandomFootholdTrackerRunnerCfg,
)
from gym.envs.humanoid.humanoid_route_foothold_tracker import (
    R1RouteFootholdTracker,
)
from gym.envs.humanoid.humanoid_route_foothold_tracker_config import (
    R1RouteFootholdTrackerCfg,
    R1RouteFootholdTrackerRunnerCfg,
)
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
        self.commands = torch.zeros(num_envs, 3)
        self.commands[:, 0] = 0.4
        self.step_period = torch.full((num_envs, 1), 30.0)
        self.dt = 0.01


class MockRouteEnv:
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.root_states = torch.zeros(num_envs, 13)
        self.root_states[:, 6] = 1.0
        self.commands = torch.zeros(num_envs, 3)
        self.common_step_counter = 0


def sampler_cfg():
    return SimpleNamespace(
        step_length_range=[0.15, 0.35],
        step_width_range=[0.18, 0.22],
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


def perturbed_sampler_cfg(jitter_x=(-0.04, 0.04), jitter_y=(-0.05, 0.05)):
    return SimpleNamespace(
        nominal_step_width=0.20,
        jitter_x_range=list(jitter_x),
        jitter_y_range=list(jitter_y),
        yaw_offset_deg=[0.0, 0.0],
        height_range=[0.0, 0.0],
        hold_probability=0.0,
        feet_direction_deg=[0.0, 0.0],
        reach_x=[0.05, 0.30],
        reach_y=[-0.30, 0.30],
        min_lateral_separation=0.12,
    )


def route_cfg():
    return SimpleNamespace(
        speed=0.4,
        rate_smoothing=0.8,
        boundary_reflection=0.5,
        heading_bound_deg=[10.0, 30.0, 45.0],
        heading_noise_deg=[2.0, 4.0, 6.0],
        heading_rate_limit_deg=[4.0, 8.0, 12.0],
        curriculum_step_thresholds=[12000, 24000],
        history_length=8,
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

    def test_perturbed_nominal_goal_uses_velocity_and_step_time(self):
        env = MockEnv()
        sampler = PerturbedFootholdGoalSampler(
            env, perturbed_sampler_cfg((0.0, 0.0), (0.0, 0.0))
        )
        sampler.reset(torch.arange(env.num_envs))
        nominal = sampler.nominal_world_target_pos
        # The left foot swings from the right stance foot at y=-0.15.
        torch.testing.assert_close(
            nominal[:, LEFT_FOOT, 0], torch.full((env.num_envs,), 0.12)
        )
        torch.testing.assert_close(
            nominal[:, LEFT_FOOT, 1], torch.full((env.num_envs,), 0.05)
        )
        goal = sampler.get_goal().reshape(env.num_envs, 2, 5)
        torch.testing.assert_close(
            goal[:, LEFT_FOOT, :2],
            torch.tensor([[0.12, 0.20]]).repeat(env.num_envs, 1),
        )

    def test_perturbed_goal_respects_jitter_bounds_and_no_crossing(self):
        env = MockEnv(256)
        sampler = PerturbedFootholdGoalSampler(env, perturbed_sampler_cfg())
        ids = torch.arange(env.num_envs)
        sampler.reset(ids)
        self.assertTrue(torch.all(sampler.foothold_jitter[:, 0].abs() <= 0.04))
        self.assertTrue(torch.all(sampler.foothold_jitter[:, 1].abs() <= 0.05))
        goal = sampler.get_goal().reshape(env.num_envs, 2, 5)
        self.assertTrue(torch.all(goal[:, LEFT_FOOT, 1] >= 0.12))

        env.swing_foot_index[:] = RIGHT_FOOT
        env.stance_foot_index[:] = LEFT_FOOT
        sampler.update(ids, torch.ones(env.num_envs, dtype=torch.bool))
        goal = sampler.get_goal().reshape(env.num_envs, 2, 5)
        self.assertTrue(torch.all(goal[:, RIGHT_FOOT, 1] <= -0.12))

    def test_perturbed_goal_is_frozen_between_phase_switches(self):
        env = MockEnv()
        sampler = PerturbedFootholdGoalSampler(env, perturbed_sampler_cfg())
        ids = torch.arange(env.num_envs)
        sampler.reset(ids)
        goal_before = sampler.get_goal().clone()
        target_before = sampler.world_target_pos.clone()
        jitter_before = sampler.foothold_jitter.clone()
        env.commands[:, 0] = 1.0
        env.foot_states[:, :, 0] += 10.0
        sampler.update(ids, torch.zeros(env.num_envs, dtype=torch.bool))
        torch.testing.assert_close(sampler.get_goal(), goal_before)
        torch.testing.assert_close(sampler.world_target_pos, target_before)
        torch.testing.assert_close(sampler.foothold_jitter, jitter_before)

    def test_random_route_reset_and_curriculum(self):
        env = MockRouteEnv()
        route = SmoothRandomRouteGenerator(env, route_cfg())
        ids = torch.arange(env.num_envs)
        route.reset(ids)
        torch.testing.assert_close(route.heading, torch.zeros(env.num_envs))
        torch.testing.assert_close(
            env.commands[:, 0], torch.full((env.num_envs,), 0.4)
        )
        torch.testing.assert_close(env.commands[:, 1], torch.zeros(env.num_envs))
        self.assertEqual(route.curriculum_values(), (0, 10.0, 2.0, 4.0))
        env.common_step_counter = 12000
        self.assertEqual(route.curriculum_values(), (1, 30.0, 4.0, 8.0))
        env.common_step_counter = 24000
        self.assertEqual(route.curriculum_values(), (2, 45.0, 6.0, 12.0))

    def test_random_route_rate_limit_and_boundary_reflection(self):
        env = MockRouteEnv(2)
        route = SmoothRandomRouteGenerator(env, route_cfg())
        ids = torch.arange(env.num_envs)
        route.reset(ids)
        route.heading[:] = torch.deg2rad(torch.tensor([9.0, -9.0]))
        route.heading_rate[:] = torch.deg2rad(torch.tensor([4.0, -4.0]))
        route._uniform = lambda low, high, count: torch.tensor([2.0, -2.0])
        route.update(ids)
        torch.testing.assert_close(
            torch.rad2deg(route.heading), torch.tensor([10.0, -10.0])
        )
        torch.testing.assert_close(
            torch.rad2deg(route.heading_rate), torch.tensor([-2.0, 2.0])
        )

    def test_route_goal_yaw_follows_route_heading(self):
        env = MockEnv()
        env.route_heading = torch.full((env.num_envs,), 0.3)
        cfg = perturbed_sampler_cfg((0.0, 0.0), (0.0, 0.0))
        sampler = RouteFootholdGoalSampler(env, cfg)
        sampler.reset(torch.arange(env.num_envs))
        goal = sampler.get_goal().reshape(env.num_envs, 2, 5)
        goal_yaw = torch.atan2(
            goal[:, LEFT_FOOT, 3], goal[:, LEFT_FOOT, 4]
        )
        torch.testing.assert_close(
            goal_yaw, torch.full((env.num_envs,), 0.3)
        )

    def test_route_updates_once_per_left_right_pair(self):
        tracker = R1RouteFootholdTracker.__new__(R1RouteFootholdTracker)
        tracker.swing_foot_index = torch.tensor(
            [LEFT_FOOT, RIGHT_FOOT, LEFT_FOOT, RIGHT_FOOT]
        )
        updated = []
        tracker.route_generator = SimpleNamespace(
            update=lambda env_ids: updated.append(env_ids.clone())
        )
        tracker._before_goal_update(torch.arange(4))
        self.assertEqual(updated[0].tolist(), [0, 2])

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
        self.assertEqual(env_cfg.foothold_goal.step_length_range, [0.08, 0.16])
        self.assertEqual(env_cfg.foothold_goal.step_width_range, [0.18, 0.22])
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

    def test_random_task_preserves_policy_shape_and_flat_defaults(self):
        base_cfg = R1FootholdTrackerCfg()
        env_cfg = R1RandomFootholdTrackerCfg()
        runner_cfg = R1RandomFootholdTrackerRunnerCfg()
        self.assertEqual(env_cfg.terrain.mesh_type, "plane")
        self.assertEqual(env_cfg.env.num_observations, 54)
        self.assertEqual(env_cfg.foothold_goal.jitter_x_range, [-0.04, 0.04])
        self.assertEqual(env_cfg.foothold_goal.jitter_y_range, [-0.05, 0.05])
        self.assertEqual(env_cfg.commands.ranges.lin_vel_x, [0.4, 0.4])
        self.assertEqual(runner_cfg.runner.experiment_name,
                         "Unitree_R1_Foothold_Tracker")
        self.assertEqual(len(runner_cfg.policy.actor_obs), 7)
        self.assertEqual(len(runner_cfg.policy.critic_obs), 11)
        self.assertEqual(base_cfg.foothold_goal.step_length_range, [0.08, 0.16])
        self.assertFalse(hasattr(base_cfg.foothold_goal, "jitter_x_range"))

    def test_route_task_configuration(self):
        env_cfg = R1RouteFootholdTrackerCfg()
        runner_cfg = R1RouteFootholdTrackerRunnerCfg()
        self.assertEqual(env_cfg.env.num_observations, 54)
        self.assertEqual(env_cfg.random_route.heading_bound_deg,
                         [10.0, 30.0, 45.0])
        self.assertEqual(env_cfg.rewards.weights.base_heading, 0.0)
        self.assertEqual(env_cfg.rewards.weights.route_heading_tracking, 4.0)
        self.assertEqual(runner_cfg.runner.experiment_name,
                         "Unitree_R1_Foothold_Tracker")
        self.assertEqual(
            R1RandomFootholdTrackerCfg().rewards.weights.base_heading, 4.0
        )

    def test_swing_trajectory_reference(self):
        tracker = R1FootholdTracker.__new__(R1FootholdTracker)
        tracker.num_envs = 2
        tracker.device = torch.device("cpu")
        tracker.phase = torch.tensor([[0.0], [0.25]])
        tracker.swing_foot_index = torch.tensor([LEFT_FOOT, LEFT_FOOT])
        tracker.swing_start_pos = torch.zeros(2, 3)
        tracker.swing_start_yaw = torch.zeros(2)
        tracker.cfg = SimpleNamespace(
            rewards=SimpleNamespace(
                foot_clearance=0.07,
            )
        )
        tracker.goal_generator = SimpleNamespace(
            world_target_pos=torch.tensor(
                [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]] * 2
            ),
            world_target_yaw=torch.zeros(2, 2),
        )
        position = torch.zeros(2, 2, 3)
        position[1, LEFT_FOOT] = torch.tensor([0.5, 0.0, 0.07])
        tracker._canonical_foot_values = lambda: (
            position,
            torch.zeros(2, 2),
            torch.zeros(2, 2, dtype=torch.bool),
            torch.zeros(2, 2, 3),
        )

        position_error, yaw_error = tracker._swing_trajectory_errors()

        torch.testing.assert_close(position_error, torch.zeros_like(position_error))
        torch.testing.assert_close(yaw_error, torch.zeros_like(yaw_error))

    def test_touchdown_event_is_contact_rising_edge(self):
        tracker = R1FootholdTracker.__new__(R1FootholdTracker)
        tracker.num_envs = 2
        tracker.device = torch.device("cpu")
        tracker.phase = torch.tensor([[0.45], [0.45]])
        tracker.swing_foot_index = torch.tensor([LEFT_FOOT, RIGHT_FOOT])
        tracker.touchdown_credited = torch.zeros(2, dtype=torch.bool)
        tracker.previous_foot_contact = torch.zeros(2, 2, dtype=torch.bool)
        tracker.touchdown_event = torch.zeros(2, 2, dtype=torch.bool)
        tracker.last_touchdown_pos = torch.zeros(2, 3)
        tracker.last_touchdown_target = torch.zeros(2, 3)
        tracker.has_touchdown_visual = torch.zeros(2, dtype=torch.bool)
        tracker.goal_generator = SimpleNamespace(
            world_target_pos=torch.ones(2, 2, 3)
        )
        tracker.cfg = SimpleNamespace(
            rewards=SimpleNamespace(touchdown_phase=0.70)
        )
        contact = torch.tensor([[True, False], [False, True]])
        tracker._canonical_foot_values = lambda: (
            torch.zeros(2, 2, 3),
            torch.zeros(2, 2),
            contact,
            torch.zeros(2, 2, 3),
        )

        tracker._update_touchdown_events()
        self.assertEqual(tracker.touchdown_event.tolist(), contact.tolist())
        self.assertTrue(tracker.touchdown_credited.all().item())
        self.assertTrue(tracker.has_touchdown_visual.all().item())
        torch.testing.assert_close(tracker.last_touchdown_target, torch.ones(2, 3))
        tracker.previous_foot_contact[:] = False
        tracker._update_touchdown_events()
        self.assertFalse(tracker.touchdown_event.any().item())


if __name__ == "__main__":
    unittest.main()
