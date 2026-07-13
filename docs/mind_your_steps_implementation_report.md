# Mind Your Steps Implementation Report

## Result

The `r1_foothold_tracker` task has been added to the existing Isaac Gym,
PyTorch, PPO, joint-position-target, and PD-control stack and now always loads
the Unitree R1 12-DOF asset. The original task registrations and implementations
were not replaced. No
LocoMuJoCo, MuJoCo, MJX, JAX, or additional dependency was installed.

The flat-ground first stage passes unit tests and GPU smoke training on an
NVIDIA GeForce RTX 4090. The environment used PyTorch `1.10.0+cu113`, NVIDIA
driver `580.126.09`, GPU PhysX, and the GPU tensor pipeline.

## Added Files

- `gym/envs/humanoid/humanoid_foothold_tracker.py`
- `gym/envs/humanoid/humanoid_foothold_tracker_config.py`
- `gym/envs/hi_cl_12/r1_12dof_controller_config.py`
- `gym/envs/guidance/foothold_goal_sampler.py`
- `gym/envs/guidance/stance_foot_adapter.py`
- `gym/envs/guidance/lipm_goal_adapter.py`
- `gym/envs/guidance/ipc3d_goal_adapter.py`
- `gym/tests/test_foothold_goal.py`
- `docs/mind_your_steps_integration.md`
- `docs/mind_your_steps_implementation_report.md`

`gym/envs/__init__.py` was extended to register the new task as
`r1_foothold_tracker`. `humanoid_foothold_tracker` is retained as a compatibility
alias and also loads the same R1 class and configuration.

## Implemented Behavior

- The new task inherits `Hicl12Controller` with `R1_12DofControllerCfg`; it
  reuses the existing PPO action, 12-DOF joint-position target, torque clipping,
  and PD-control path.
- R1 ankle, sole-center, knee, hip, and right/left link mappings are explicit.
- Canonical goal foot order is left then right. Each goal is
  `[x, y, z, sin(yaw), cos(yaw)]`, producing `[num_envs, 10]`.
- World targets are transformed into the current stance-foot frame only at reset
  or a half-cycle phase switch. The actor tensor remains frozen during swing.
- Phase `[0.0, 0.5)` selects left swing/right stance; phase `[0.5, 1.0)` selects
  right swing/left stance. Contact does not control phase transitions.
- `swing_foot_index`, `stance_foot_index`, `previous_swing_foot_index`, and
  `phase_switch_mask` are stored as batched tensors.
- The random generator samples episode headings and phase-level length,
  direction, yaw, height, and hold mode with batched Torch operations. It clamps
  reach and enforces lateral separation to prevent leg crossing.
- A `support_platform_generator` hook and `height_range` are reserved; the active
  configuration uses a plane and zero target-height variation.
- `LIPMFootholdAdapter` and `IPC3DFootholdAdapter` implement the same generator
  contract. Planner callbacks may return world- or base-frame batched targets;
  the adapters convert and latch them in the stance-foot frame.

## Observations

The actor observation is 54 dimensions:

```text
projected_gravity             3
base_angular_velocity         3
dof_position_error           12
dof_velocity                 12
previous_action              12
phase_sin_cos                 2
foothold_goal                10
```

It excludes exact world position, exact base linear velocity, foot contacts,
terrain truth, and sampler parameters.

The critic observation is 71 dimensions. In addition to the actor observation,
it includes world-frame base linear velocity, foot contacts, world-frame foot
target error, and contact forces.

## Rewards

The task registers the requested foothold rewards:

- `swing_foot_xy_tracking`
- `swing_foot_z_tracking`
- `swing_foot_yaw_tracking`
- `touchdown_accuracy`
- `stance_accuracy_latched`
- `phase_foot_clearance`
- `swing_knee_height`

It also reuses inherited orientation, base-height, action-rate, joint-velocity,
joint-acceleration, torque, collision, joint-limit, and foot-slip terms.

After diagnosing the early hopping policy, the task also registers stance-foot
contact, flight, and early-swing-contact terms. Stage-one training uses a fixed
`+0.4 m/s` world-forward curriculum reward while keeping exact base linear
velocity out of the actor observation.

## Verification

Unit and static checks:

```bash
conda run -n footstep python -m unittest gym.tests.test_foothold_goal -v
conda run -n footstep python -m py_compile \
  gym/envs/guidance/stance_foot_adapter.py \
  gym/envs/guidance/foothold_goal_sampler.py \
  gym/envs/guidance/lipm_goal_adapter.py \
  gym/envs/guidance/ipc3d_goal_adapter.py \
  gym/envs/humanoid/humanoid_foothold_tracker.py \
  gym/envs/humanoid/humanoid_foothold_tracker_config.py
git diff --check
```

All seven unit tests passed: stance transform, yaw wrapping, phase switching,
goal freezing, left/right switching with crossing prevention, goal shape, and
the R1 asset/12-DOF observation contract. Compilation and whitespace checks
passed.

Original-task GPU smoke test:

```bash
conda run -n footstep python gym/scripts/train.py \
  --task=humanoid_controller --num_envs=8 --max_iterations=1 \
  --headless --disable_wandb --disable_local_saving
```

Result: exit code 0, one PPO iteration completed, Actor/Critic dimensions remained
52/52, and no NaN, CUDA, or tensor-shape error occurred.

R1 controller baseline smoke test:

```bash
conda run -n footstep python gym/scripts/train.py \
  --task=r1_12dof_controller --num_envs=8 --max_iterations=1 \
  --headless --disable_wandb --disable_local_saving
```

Result: exit code 0, one PPO iteration completed with the R1 12-DOF asset and
12-dimensional action output.

New-task GPU smoke test:

```bash
conda run -n footstep python gym/scripts/train.py \
  --task=r1_foothold_tracker --num_envs=64 --max_iterations=2 \
  --headless --disable_wandb --disable_local_saving
```

Result: exit code 0, two PPO iterations and 3072 simulation steps completed.
Actor/Critic dimensions were 54/71 and the action dimension was 12. No NaN,
CUDA out-of-bounds, or tensor-shape
error occurred.

Hopping diagnostics then changed the R1 spawn height from `0.68` to `0.715`,
applied a `0.25` action scale in the PD target, tightened action clipping, and
added stance/flight timing rewards. A fresh 50-iteration diagnostic run reduced
flight ratio from `90.68%` for the old checkpoint to below `1%`. It did not yet
produce forward alternating walking: double support remained above `95%` and
mean world-forward velocity was negative. These checkpoints are diagnostics,
not converged policies.

The R1 foot collision was subsequently simplified to a sole-aligned box and the
policy foot-point offset was corrected to the mesh sole. A final two-iteration
GPU smoke run completed successfully after this geometry change. The zero-action
robot still falls forward over a longer horizon, so this geometry correction is
not reported as policy convergence.

## Intentionally Deferred

- Complex moving stepping stones and nonzero platform heights are deferred until
  flat-ground policy convergence is evaluated.
- LIPM and IPC3D adapters are implemented but are not selected by the default
  random-goal task configuration; connecting a concrete planner callback is a
  separate experiment and does not alter the existing planners.
- Two fresh 50-iteration diagnostic runs were performed after the smoke test.
  No claim about long-horizon PPO convergence is made, and no long training job
  was started.
