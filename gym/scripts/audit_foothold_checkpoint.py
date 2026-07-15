"""Audit whether an R1 checkpoint responds to stance-frame foothold goals."""

from collections import defaultdict

import isaacgym  # noqa: F401
import torch

from gym.envs import *  # noqa: F401,F403
from gym.utils import get_args, task_registry
from gym.utils.math import quat_rotate_inverse


GROUP_SIZE = 32
GROUPS = ("normal", "goal_zero", "short", "long", "narrow", "wide")


def _group_slice(index):
    start = index * GROUP_SIZE
    return slice(start, start + GROUP_SIZE)


def _set_swing_goal(obs, env, env_ids, goal_offset, x=None, width=None):
    swing = env.swing_foot_index[env_ids]
    rows = env_ids
    base = goal_offset + swing * 5
    if x is not None:
        obs[rows, base] = x
    if width is not None:
        signed_width = torch.where(swing == 0, width, -width)
        obs[rows, base + 1] = signed_width


def audit(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.env.num_envs = GROUP_SIZE * len(GROUPS)
    env_cfg.env.episode_length_s = 100
    # Match play.py: the checkpoint receives no velocity command through the actor.
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = 0.0
    env_cfg.commands.ranges.yaw_vel = 0.0
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_Kp_factor = False
    env_cfg.domain_rand.randomize_Kd_factor = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    runner.alg.actor_critic.eval()

    actor_obs = runner.policy_cfg["actor_obs"]
    goal_index = actor_obs.index("foothold_goal")
    goal_offset = sum(runner.get_obs_size([name]) for name in actor_obs[:goal_index])
    goal_size = runner.get_obs_size(["foothold_goal"])
    assert goal_size == 10

    group_ids = {
        name: torch.arange(
            index * GROUP_SIZE,
            (index + 1) * GROUP_SIZE,
            device=env.device,
        )
        for index, name in enumerate(GROUPS)
    }
    metrics = defaultdict(lambda: defaultdict(float))
    warmup_steps, sample_steps = 30, 300
    previous_goal = env.foothold_goal.clone()
    frozen_max_error = 0.0
    switch_change_sum = 0.0
    switch_change_samples = 0
    action_delta_sum = 0.0
    action_delta_samples = 0

    with torch.inference_mode():
        for step in range(warmup_steps + sample_steps):
            obs = runner.get_obs(actor_obs)
            normal_actions = runner.alg.actor_critic.act_inference(obs)

            zero_obs = obs.clone()
            zero_obs[:, goal_offset : goal_offset + goal_size] = 0.0
            zero_actions = runner.alg.actor_critic.act_inference(zero_obs)
            action_delta_sum += (
                (normal_actions - zero_actions).square().mean(dim=1).sqrt().sum().item()
            )
            action_delta_samples += env.num_envs

            controlled_obs = obs.clone()
            zero_ids = group_ids["goal_zero"]
            controlled_obs[zero_ids, goal_offset : goal_offset + goal_size] = 0.0
            _set_swing_goal(
                controlled_obs, env, group_ids["short"], goal_offset, x=0.08
            )
            _set_swing_goal(
                controlled_obs, env, group_ids["long"], goal_offset, x=0.16
            )
            _set_swing_goal(
                controlled_obs, env, group_ids["narrow"], goal_offset, width=0.18
            )
            _set_swing_goal(
                controlled_obs, env, group_ids["wide"], goal_offset, width=0.22
            )
            actions = runner.alg.actor_critic.act_inference(controlled_obs)
            runner.set_actions(actions)
            env.step()

            goal_delta = (env.foothold_goal - previous_goal).abs().amax(dim=1)
            stable = ~env.phase_switch_mask & ~env.reset_buf
            if stable.any():
                frozen_max_error = max(
                    frozen_max_error, goal_delta[stable].max().item()
                )
            switched = env.phase_switch_mask & ~env.reset_buf
            if switched.any():
                switch_change_sum += goal_delta[switched].sum().item()
                switch_change_samples += switched.sum().item()
            previous_goal.copy_(env.foothold_goal)

            if step >= warmup_steps:
                position, _, _, _ = env._canonical_foot_values()
                canonical_quat = env.foot_states[:, [1, 0], 3:7]
                batch = torch.arange(env.num_envs, device=env.device)
                swing = env.swing_foot_index
                stance = env.stance_foot_index
                relative = position[batch, swing] - position[batch, stance]
                actual_local = quat_rotate_inverse(
                    canonical_quat[batch, stance], relative
                )
                touchdown = env.touchdown_event[batch, swing]

                for index, name in enumerate(GROUPS):
                    ids = group_ids[name]
                    event = touchdown[ids]
                    metrics[name]["speed"] += env.root_states[ids, 7].sum().item()
                    metrics[name]["samples"] += ids.numel()
                    metrics[name]["resets"] += env.reset_buf[ids].sum().item()
                    if event.any():
                        local = actual_local[ids][event]
                        metrics[name]["touchdowns"] += event.sum().item()
                        metrics[name]["local_x"] += local[:, 0].sum().item()
                        metrics[name]["abs_local_y"] += local[:, 1].abs().sum().item()

            runner.reset_envs()
            previous_goal.copy_(env.foothold_goal)

    print(f"actor_obs_dim: {runner.num_actor_obs}")
    print(f"critic_obs_dim: {runner.num_critic_obs}")
    print(f"goal_obs_slice: [{goal_offset}:{goal_offset + goal_size}]")
    print(f"commands_abs_max: {env.commands.abs().max().item():.6f}")
    print(f"frozen_goal_max_change: {frozen_max_error:.8f}")
    print(
        "mean_goal_change_on_switch: "
        f"{switch_change_sum / max(switch_change_samples, 1):.6f}"
    )
    print(
        "same_state_goal_zero_action_rms_delta: "
        f"{action_delta_sum / action_delta_samples:.6f} rad"
    )
    for name in GROUPS:
        values = metrics[name]
        touchdown_count = max(values["touchdowns"], 1.0)
        print(
            f"{name}: speed={values['speed'] / values['samples']:.4f} m/s, "
            f"touchdown_local_x={values['local_x'] / touchdown_count:.4f} m, "
            f"touchdown_abs_local_y={values['abs_local_y'] / touchdown_count:.4f} m, "
            f"touchdowns={int(values['touchdowns'])}, resets={int(values['resets'])}"
        )


if __name__ == "__main__":
    audit(get_args())
