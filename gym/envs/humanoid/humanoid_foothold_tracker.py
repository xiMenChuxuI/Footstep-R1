"""Phase-driven stance-frame foothold tracking task."""

import torch

from gym.envs.base.legged_robot import LeggedRobot
from gym.envs.guidance.foothold_goal_sampler import RandomFootholdGoalSampler
from gym.envs.guidance.stance_foot_adapter import (
    LEFT_FOOT,
    RIGHT_FOOT,
    phase_foot_indices,
)
from gym.envs.hi_cl_12.hi_cl12_controller import Hicl12Controller
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
        self.goal_generator = RandomFootholdGoalSampler(self, self.cfg.foothold_goal)

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
        self.goal_generator.reset(env_ids)
        self.foothold_goal[env_ids] = self.goal_generator.get_goal()[env_ids]

    def _post_physics_step_callback(self):
        # Skip the legacy moving-base planner update inherited from Hicl12Controller.
        LeggedRobot._post_physics_step_callback(self)
        self._update_robot_states()
        self._calculate_CoM()
        self._calculate_ICP()
        self._advance_phase_and_goals()

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
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.goal_generator.update(env_ids, self.phase_switch_mask)
        self.foothold_goal[:] = self.goal_generator.get_goal()

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
        error, _, _ = self._swing_tracking_errors()
        return self._tracking_reward(
            torch.sum(error[:, :2].square(), dim=-1),
            self.cfg.rewards.swing_xy_tracking_sigma,
        )

    def _reward_swing_foot_z_tracking(self):
        error, _, _ = self._swing_tracking_errors()
        late_swing = self._local_phase() > self.cfg.rewards.touchdown_phase
        return late_swing.float() * self._tracking_reward(
            error[:, 2].square(), self.cfg.rewards.swing_z_tracking_sigma
        )

    def _reward_swing_foot_yaw_tracking(self):
        _, error, _ = self._swing_tracking_errors()
        return self._tracking_reward(
            error.square(), self.cfg.rewards.swing_yaw_tracking_sigma
        )

    def _reward_touchdown_accuracy(self):
        error, yaw_error, contact = self._swing_tracking_errors()
        local_phase = self._local_phase()
        touchdown = contact & (local_phase > self.cfg.rewards.touchdown_phase)
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
