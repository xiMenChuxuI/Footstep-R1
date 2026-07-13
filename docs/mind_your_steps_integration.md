# Mind Your Steps Integration Analysis

## Scope

The new `r1_foothold_tracker` task stays inside the existing Isaac Gym,
PyTorch, PPO, joint-position-target, and PD-control stack. Its only training
robot is the Unitree R1 12-DOF model. It is added beside the original tasks;
no LocoMuJoCo, MuJoCo, MJX, or JAX dependency is used.

The reference design was inspected from LocoMuJoCo's
`loco_mujoco/core/observations/goals_foot_placement.py` and
`loco_mujoco/core/reward/foot_placement.py`. The reusable ideas are phase-based
swing-foot selection, stance-relative goals, safe lateral placement, hold-still
sampling, and touchdown/stance tracking. Simulator-specific code is not copied.

## Existing Control Path

`gym/envs/hi_cl_12/hi_cl12_controller.py` derives from `LeggedRobot`, and the
R1 configuration supplies the 12 actuated leg joints and required low-level path:

1. PPO emits `dof_pos_target`.
2. `_compute_torques()` adds `default_dof_pos`.
3. PD control applies configured stiffness/damping.
4. Torque limits are applied before Isaac Gym PhysX.

The new task therefore inherits `Hicl12Controller` with
`R1_12DofControllerCfg` and replaces only goal, phase-observation, and
foothold-reward behavior.

## Task Registration

`gym/envs/__init__.py` imports task classes/configurations and registers them in
the singleton `task_registry`. The new task must be registered there under
`r1_foothold_tracker`. The requested `humanoid_foothold_tracker` name remains a
compatibility alias to the same R1 class and configuration.

## Observation Pipeline

`BaseTask.get_states()` concatenates tensors named by the runner configuration.
`OnPolicyRunner` independently evaluates `policy.actor_obs` and
`policy.critic_obs`, so asymmetric actor/critic observations are already
supported.

`LeggedRobot` provides `projected_gravity`, `base_ang_vel`, `base_lin_vel`,
`dof_pos`, `dof_vel`, `default_dof_pos`, contact forces, and action histories.
The new task will expose explicit aliases:

- `dof_pos_error = dof_pos - default_dof_pos`
- `previous_action`
- `phase_sin_cos`
- `foothold_goal`, shape `[num_envs, 10]`
- privileged critic tensors for contacts, forces, and world target error

The actor excludes world position, base linear velocity, contacts, terrain
truth, and sampler internals.

The current controller's `_set_obs_variables()` converts world step commands to
the moving base frame every simulation step. That representation must not be
reused by the new actor. The new goal is converted once at reset or a half-cycle
phase switch and then stored unchanged throughout the swing phase.

## Reward Pipeline

`BaseTask._prepare_reward_function()` converts non-zero fields in
`cfg.rewards.weights` into method names `_reward_<field>`. Reward methods return
one value per environment. Existing regularizers in `LeggedRobot` and
`Hicl12Controller` and `LeggedRobot` can be reused by naming their weights in
the new config.

The new task adds swing XY/Z/yaw tracking, touchdown accuracy, latched stance
accuracy, phase clearance, and swing-knee-height methods. Contact may affect
rewards and critic observations but not actor observations.

## Existing Phase Semantics

`Hicl12Controller` increments normalized `phase` every control step. A full
cycle lasts `2 * step_period`; `update_commands_ids` becomes true every
`step_period`, while `update_phase_ids` resets the full cycle. Thus the existing
half-cycle command switch is the correct event for a foothold goal update.

The new task makes this explicit:

- phase `[0.0, 0.5)`: left swing, right stance
- phase `[0.5, 1.0)`: right swing, left stance
- canonical goal foot indices: left `0`, right `1`
- existing environment foot-state indices: right `0`, left `1`

It stores `swing_foot_index`, `stance_foot_index`,
`previous_swing_foot_index`, and `phase_switch_mask`. Contact never drives phase
transitions in the first implementation.

## Goal Representation

Each foot goal is `[x, y, z, sin(yaw), cos(yaw)]`; left then right produces a
10-D actor tensor. World targets are transformed using the stance foot pose:

```text
target_pos_stance = quat_rotate_inverse(
    stance_quat, target_pos_world - stance_pos)
target_yaw_stance = wrap_to_pi(target_yaw_world - stance_yaw)
```

Both feet use the same stance frame for a given swing phase. The transformed
tensor is latched and is not recomputed as the base moves.

## Goal Generator Boundary

`FootholdGoalGenerator` defines `reset(env_ids)`,
`update(env_ids, phase_switch_mask)`, and `get_goal()`. The random implementation
uses batched torch operations and samples episode-level movement/feet headings,
then phase-level step length, direction, yaw, height, and hold mode.

Reachability and crossing prevention are enforced in stance coordinates by
clamping forward/lateral reach and requiring a positive lateral offset for a
left swing and a negative offset for a right swing. Height is configured but
zero for the flat first stage. A support-platform generator hook is reserved.

`LIPMFootholdAdapter` and `IPC3DFootholdAdapter` accept planner outputs through
the same adapter boundary. World-frame outputs are converted directly; a
base-frame output must first be converted to world and then latched in the
stance frame. The initial task selects `RandomFootholdGoalSampler` so existing
LIPM/IPC3D behavior is untouched.

## Implementation Plan and Risks

The task subclasses `Hicl12Controller` with the Unitree R1 12-DOF asset,
overrides phase/goal lifecycle and actor-facing observation construction, and
adds only foothold-specific rewards. The first stage uses R1 on plane terrain.

Main risks are foot-order mismatches, initialization order during environment
reset, inherited base-frame step observations leaking into the actor, and reward
events being counted for more than one control step. Tests therefore target
transforms, yaw wrapping, phase transitions, goal freezing, left/right mapping,
shape contracts, and both old/new task smoke runs.
