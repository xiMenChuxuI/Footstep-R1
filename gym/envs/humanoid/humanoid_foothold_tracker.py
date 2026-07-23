"""Phase-driven stance-frame foothold tracking task."""

import torch
from isaacgym import gymapi, gymutil

from gym.envs.base.legged_robot import LeggedRobot
from gym.envs.guidance.foothold_goal_sampler import RandomFootholdGoalSampler
from gym.envs.guidance.stance_foot_adapter import (
    LEFT_FOOT,
    RIGHT_FOOT,
    phase_foot_indices,
)
from gym.envs.hi_cl_12.hi_cl12_controller import Hicl12Controller
from gym.envs.hi_cl_12.hi_cl12_utils import FootStepGeometry, SimpleLineGeometry
from gym.utils.math import quat_apply, wrap_to_pi


class R1FootholdTracker(Hicl12Controller):
    """Track phase-latched footholds with the Unitree R1 12-DOF model."""

    def _init_buffers(self):
        super()._init_buffers()
        n, device = self.num_envs, self.device
        self.swing_foot_index = torch.full(
            (n,), LEFT_FOOT, dtype=torch.long, device=device
        )
        self.stance_foot_index = torch.full(
            (n,), RIGHT_FOOT, dtype=torch.long, device=device
        )
        self.previous_swing_foot_index = self.swing_foot_index.clone()
        self.phase_switch_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.phase_sin_cos = torch.zeros(n, 2, device=device)
        self.foothold_goal = torch.zeros(n, 10, device=device)
        self.dof_pos_error = torch.zeros(n, self.num_actuators, device=device)
        self.previous_action = torch.zeros(n, self.num_actuators, device=device)
        self.foot_contact_state = torch.zeros(n, 2, device=device)
        self.foot_contact_force = torch.zeros(n, 6, device=device)
        self.world_foot_target_error = torch.zeros(n, 6, device=device)
        self.stance_accuracy_latched = torch.zeros(n, device=device)
        self.swing_start_pos = torch.zeros(n, 3, device=device)
        self.swing_start_yaw = torch.zeros(n, device=device)
        self.previous_foot_contact = torch.zeros(
            n, 2, dtype=torch.bool, device=device
        )
        self.touchdown_event = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.touchdown_credited = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_touchdown_pos = torch.zeros(n, 3, device=device)
        self.last_touchdown_target = torch.zeros(n, 3, device=device)
        self.has_touchdown_visual = torch.zeros(n, dtype=torch.bool, device=device)
        self.goal_generator = self._create_goal_generator()

    def _create_goal_generator(self):
        return RandomFootholdGoalSampler(self, self.cfg.foothold_goal)

    def _calculate_foot_states(self, foot_states):
        """Move R1 ankle-link poses to the configured sole contact center."""
        offset = torch.tensor(
            self.cfg.asset.foot_position_offset,
            dtype=foot_states.dtype,
            device=self.device,
        ).repeat(self.num_envs, 1)
        foot_states[:, 0, :3] += quat_apply(foot_states[:, 0, 3:7], offset)
        foot_states[:, 1, :3] += quat_apply(foot_states[:, 1, 3:7], offset)
        return foot_states

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        self.swing_foot_index[env_ids] = LEFT_FOOT
        self.stance_foot_index[env_ids] = RIGHT_FOOT
        self.previous_swing_foot_index[env_ids] = LEFT_FOOT
        self.phase_switch_mask[env_ids] = False
        self.stance_accuracy_latched[env_ids] = 0.0
        self.previous_foot_contact[env_ids] = False
        self.touchdown_event[env_ids] = False
        self.touchdown_credited[env_ids] = False
        self.has_touchdown_visual[env_ids] = False
        self.goal_generator.reset(env_ids)
        self.foothold_goal[env_ids] = self.goal_generator.get_goal()[env_ids]
        self._latch_swing_start(env_ids)

    def _post_physics_step_callback(self):
        # Skip the legacy moving-base planner update inherited from Hicl12Controller.
        LeggedRobot._post_physics_step_callback(self)
        self._update_robot_states()
        self.foot_contact[:] = (
            self.contact_forces[:, self.feet_ids, 2]
            > self.cfg.rewards.contact_force_threshold
        )
        self._calculate_CoM()
        self._calculate_ICP()
        self._advance_phase_and_goals()
        self._update_touchdown_events()

    def _advance_phase_and_goals(self):
        wrapped = self.phase[:, 0] >= 1.0
        self.phase[wrapped, 0] -= 1.0
        self.phase_count[wrapped] = 0

        previous = self.swing_foot_index.clone()
        new_swing, new_stance, self.phase_switch_mask = phase_foot_indices(
            self.phase[:, 0], previous
        )
        self.previous_swing_foot_index[:] = previous

        if torch.any(self.phase_switch_mask):
            self._latch_stance_accuracy(self.phase_switch_mask, previous)

        self.swing_foot_index[:] = new_swing
        self.stance_foot_index[:] = new_stance
        switch_ids = self.phase_switch_mask.nonzero(as_tuple=False).flatten()
        self.touchdown_credited[switch_ids] = False
        self._latch_swing_start(switch_ids)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.goal_generator.update(env_ids, self.phase_switch_mask)
        self.foothold_goal[:] = self.goal_generator.get_goal()

    def _latch_swing_start(self, env_ids):
        if env_ids.numel() == 0:
            return
        position, yaw, _, _ = self._canonical_foot_values()
        swing = self.swing_foot_index[env_ids]
        batch = torch.arange(env_ids.numel(), device=self.device)
        self.swing_start_pos[env_ids] = position[env_ids][batch, swing]
        self.swing_start_yaw[env_ids] = yaw[env_ids][batch, swing]

    def _update_touchdown_events(self):
        position, _, contact, _ = self._canonical_foot_values()
        rising = contact & ~self.previous_foot_contact
        batch = torch.arange(self.num_envs, device=self.device)
        valid = (
            rising[batch, self.swing_foot_index]
            & ~self.touchdown_credited
            & (self._local_phase() > self.cfg.rewards.touchdown_phase)
        )
        self.touchdown_event[:] = False
        self.touchdown_event[batch, self.swing_foot_index] = valid
        if torch.any(valid):
            self.last_touchdown_pos[valid] = position[batch, self.swing_foot_index][valid]
            self.last_touchdown_target[valid] = self.goal_generator.world_target_pos[
                batch, self.swing_foot_index
            ][valid]
            self.has_touchdown_visual[valid] = True
        self.touchdown_credited |= valid
        self.previous_foot_contact[:] = contact

    def _visualization(self):
        """Draw active world-frame footholds and their tracking error."""
        if not self.cfg.viewer.draw_foothold_targets:
            return

        self.gym.clear_lines(self.viewer)
        position, yaw, _, _ = self._canonical_foot_values()
        target_pos = self.goal_generator.world_target_pos
        target_yaw = self.goal_generator.world_target_yaw
        nominal_pos = getattr(
            self.goal_generator, "nominal_world_target_pos", None
        )
        nominal_yaw = getattr(
            self.goal_generator, "nominal_world_target_yaw", None
        )
        stance_sphere = gymutil.WireframeSphereGeometry(
            0.018, 6, 6, None, color=(1.0, 1.0, 0.0)
        )
        swing_sphere = gymutil.WireframeSphereGeometry(
            0.018, 6, 6, None, color=(0.0, 1.0, 0.0)
        )
        touchdown_sphere = gymutil.WireframeSphereGeometry(
            0.014, 6, 6, None, color=(1.0, 1.0, 1.0)
        )

        for env_id in range(self.num_envs):
            # Actual feet: left cyan, right magenta.
            for foot, color in ((LEFT_FOOT, (0.0, 1.0, 1.0)), (RIGHT_FOOT, (1.0, 0.0, 1.0))):
                actual = FootStepGeometry(
                    position[env_id, foot, :2], yaw[env_id, foot], color=color
                )
                gymutil.draw_lines(
                    actual, self.gym, self.viewer, self.envs[env_id], pose=None
                )

            # Active targets: left blue, right red. One is the current stance
            # pose and the other is the phase-latched swing target.
            for foot, color in ((LEFT_FOOT, (0.0, 0.0, 1.0)), (RIGHT_FOOT, (1.0, 0.0, 0.0))):
                target = FootStepGeometry(
                    target_pos[env_id, foot, :2], target_yaw[env_id, foot], color=color
                )
                gymutil.draw_lines(
                    target, self.gym, self.viewer, self.envs[env_id], pose=None
                )

            stance = self.stance_foot_index[env_id].item()
            swing = self.swing_foot_index[env_id].item()
            if nominal_pos is not None:
                nominal = FootStepGeometry(
                    nominal_pos[env_id, swing, :2],
                    nominal_yaw[env_id, swing],
                    color=(0.85, 0.85, 0.85),
                )
                gymutil.draw_lines(
                    nominal, self.gym, self.viewer, self.envs[env_id], pose=None
                )
                perturbation = SimpleLineGeometry(
                    nominal_pos[env_id, swing],
                    target_pos[env_id, swing],
                    color=(1.0, 1.0, 0.0),
                )
                gymutil.draw_lines(
                    perturbation,
                    self.gym,
                    self.viewer,
                    self.envs[env_id],
                    pose=None,
                )
            stance_pose = gymapi.Transform(
                gymapi.Vec3(*position[env_id, stance].tolist()), r=None
            )
            swing_pose = gymapi.Transform(
                gymapi.Vec3(*position[env_id, swing].tolist()), r=None
            )
            gymutil.draw_lines(
                stance_sphere, self.gym, self.viewer, self.envs[env_id], stance_pose
            )
            gymutil.draw_lines(
                swing_sphere, self.gym, self.viewer, self.envs[env_id], swing_pose
            )

            active_error = SimpleLineGeometry(
                position[env_id, swing],
                target_pos[env_id, swing],
                color=(1.0, 0.5, 0.0),
            )
            gymutil.draw_lines(
                active_error, self.gym, self.viewer, self.envs[env_id], pose=None
            )

            if self.has_touchdown_visual[env_id]:
                touchdown_pose = gymapi.Transform(
                    gymapi.Vec3(*self.last_touchdown_pos[env_id].tolist()), r=None
                )
                gymutil.draw_lines(
                    touchdown_sphere,
                    self.gym,
                    self.viewer,
                    self.envs[env_id],
                    touchdown_pose,
                )
                touchdown_error = SimpleLineGeometry(
                    self.last_touchdown_pos[env_id],
                    self.last_touchdown_target[env_id],
                    color=(1.0, 1.0, 1.0),
                )
                gymutil.draw_lines(
                    touchdown_error,
                    self.gym,
                    self.viewer,
                    self.envs[env_id],
                    pose=None,
                )

    def _canonical_foot_values(self):
        position = self.foot_states[:, [1, 0], :3]
        yaw = self.foot_heading[:, [1, 0]]
        contact = self.foot_contact[:, [1, 0]]
        force = self.contact_forces[:, self.feet_ids, :][:, [1, 0]]
        return position, yaw, contact, force

    def _latch_stance_accuracy(self, mask, previous_swing):
        position, yaw, _, _ = self._canonical_foot_values()
        ids = mask.nonzero(as_tuple=False).flatten()
        batch = torch.arange(ids.numel(), device=self.device)
        foot = previous_swing[ids]
        target_pos = self.goal_generator.world_target_pos[ids, foot]
        target_yaw = self.goal_generator.world_target_yaw[ids, foot]
        xy_error = torch.norm(position[ids][batch, foot, :2] - target_pos[:, :2], dim=-1)
        yaw_error = torch.abs(wrap_to_pi(yaw[ids][batch, foot] - target_yaw))
        sigma = self.cfg.rewards.stance_tracking_sigma
        self.stance_accuracy_latched[ids] = torch.exp(
            -(xy_error.square() + 0.25 * yaw_error.square()) / sigma
        )

    def _set_obs_variables(self):
        # Do not call the parent: it recomputes legacy goals in the moving base frame.
        self.phase_sin_cos[:, 0] = torch.sin(2.0 * torch.pi * self.phase[:, 0])
        self.phase_sin_cos[:, 1] = torch.cos(2.0 * torch.pi * self.phase[:, 0])
        self.dof_pos_error[:] = self.dof_pos - self.default_dof_pos
        self.previous_action[:] = self.actuation_history[:, : self.num_actuators]
        self.base_lin_vel_world[:] = self.root_states[:, 7:10]

        position, _, contact, force = self._canonical_foot_values()
        self.foot_contact_state[:] = contact.float()
        self.foot_contact_force[:] = force.reshape(self.num_envs, 6)
        self.world_foot_target_error[:] = (
            position - self.goal_generator.world_target_pos
        ).reshape(self.num_envs, 6)

    def _swing_tracking_errors(self):
        position, yaw, contact, _ = self._canonical_foot_values()
        batch = torch.arange(self.num_envs, device=self.device)
        swing = self.swing_foot_index
        target_pos = self.goal_generator.world_target_pos[batch, swing]
        target_yaw = self.goal_generator.world_target_yaw[batch, swing]
        return (
            position[batch, swing] - target_pos,
            wrap_to_pi(yaw[batch, swing] - target_yaw),
            contact[batch, swing],
        )

    def _swing_trajectory_errors(self):
        position, yaw, _, _ = self._canonical_foot_values()
        batch = torch.arange(self.num_envs, device=self.device)
        swing = self.swing_foot_index
        target_pos = self.goal_generator.world_target_pos[batch, swing]
        target_yaw = self.goal_generator.world_target_yaw[batch, swing]

        phase = self._local_phase()
        progress = phase.square() * (3.0 - 2.0 * phase)
        reference_pos = self.swing_start_pos + progress[:, None] * (
            target_pos - self.swing_start_pos
        )
        reference_pos[:, 2] += self.cfg.rewards.foot_clearance * torch.sin(
            torch.pi * phase
        )
        yaw_delta = wrap_to_pi(target_yaw - self.swing_start_yaw)
        reference_yaw = wrap_to_pi(self.swing_start_yaw + progress * yaw_delta)
        return (
            position[batch, swing] - reference_pos,
            wrap_to_pi(yaw[batch, swing] - reference_yaw),
        )

    def _tracking_reward(self, squared_error, sigma=None):
        if sigma is None:
            sigma = self.cfg.rewards.foothold_tracking_sigma
        return torch.exp(-squared_error / sigma)

    def _local_phase(self):
        return torch.where(
            self.phase[:, 0] < 0.5,
            self.phase[:, 0] * 2.0,
            (self.phase[:, 0] - 0.5) * 2.0,
        )

    def _reward_swing_foot_xy_tracking(self):
        error, _ = self._swing_trajectory_errors()
        return self._tracking_reward(
            torch.sum(error[:, :2].square(), dim=-1),
            self.cfg.rewards.swing_xy_tracking_sigma,
        )

    def _reward_swing_foot_z_tracking(self):
        error, _ = self._swing_trajectory_errors()
        return self._tracking_reward(
            error[:, 2].square(), self.cfg.rewards.swing_z_tracking_sigma
        )

    def _reward_swing_foot_yaw_tracking(self):
        _, error = self._swing_trajectory_errors()
        return self._tracking_reward(
            error.square(), self.cfg.rewards.swing_yaw_tracking_sigma
        )

    def _reward_touchdown_accuracy(self):
        error, yaw_error, _ = self._swing_tracking_errors()
        batch = torch.arange(self.num_envs, device=self.device)
        touchdown = self.touchdown_event[batch, self.swing_foot_index]
        accuracy = self._tracking_reward(
            torch.sum(error.square(), dim=-1) + 0.25 * yaw_error.square(),
            self.cfg.rewards.touchdown_tracking_sigma,
        )
        return accuracy * touchdown.float()

    def _reward_stance_accuracy_latched(self):
        return self.stance_accuracy_latched

    def _reward_stance_contact(self):
        _, _, contact, _ = self._canonical_foot_values()
        batch = torch.arange(self.num_envs, device=self.device)
        return contact[batch, self.stance_foot_index].float()

    def _reward_flight(self):
        _, _, contact, _ = self._canonical_foot_values()
        return -torch.logical_not(contact.any(dim=1)).float()

    def _reward_early_swing_contact(self):
        _, _, contact, _ = self._canonical_foot_values()
        batch = torch.arange(self.num_envs, device=self.device)
        local_phase = self._local_phase()
        early_contact = contact[batch, self.swing_foot_index] & (
            local_phase < self.cfg.rewards.touchdown_phase
        )
        return -early_contact.float()

    def _reward_phase_foot_clearance(self):
        position, _, _, _ = self._canonical_foot_values()
        batch = torch.arange(self.num_envs, device=self.device)
        swing = self.swing_foot_index
        stance = self.stance_foot_index
        local_phase = self._local_phase()
        desired = self.cfg.rewards.foot_clearance * torch.sin(torch.pi * local_phase)
        clearance = position[batch, swing, 2] - position[batch, stance, 2]
        return self._tracking_reward((clearance - desired).square())

    def _reward_swing_knee_height(self):
        knee_pos = self.rigid_body_state[
            :,
            [
                self.rigid_body_idx[self.cfg.asset.left_knee_link],
                self.rigid_body_idx[self.cfg.asset.right_knee_link],
            ],
            :3,
        ]
        foot_pos, _, _, _ = self._canonical_foot_values()
        batch = torch.arange(self.num_envs, device=self.device)
        height = knee_pos[batch, self.swing_foot_index, 2] - foot_pos[
            batch, self.stance_foot_index, 2
        ]
        shortfall = torch.clamp(self.cfg.rewards.swing_knee_height - height, min=0.0)
        return self._tracking_reward(shortfall.square())

    def _reward_foot_slip(self):
        _, _, contact, _ = self._canonical_foot_values()
        velocity = self.rigid_body_state[:, self.feet_ids, 7:9][:, [1, 0]]
        return -torch.sum(torch.sum(velocity.square(), dim=-1) * contact.float(), dim=-1)

    def _reward_dof_acc(self):
        n = self.num_actuators
        acceleration = (self.dof_vel_history[:, :n] - self.dof_vel_history[:, n:]) / self.dt
        return -torch.sum(acceleration.square(), dim=-1)


# Backward-compatible import name; this task now always uses Unitree R1.
HumanoidFootholdTracker = R1FootholdTracker
