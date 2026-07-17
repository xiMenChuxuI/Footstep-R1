"""Evaluate an R1 foothold checkpoint on one selected terrain type."""

import isaacgym  # noqa: F401
import torch

from gym.envs import *  # noqa: F401,F403
from gym.utils import get_args, task_registry


def diagnose(args):
    if args.terrain_type is None:
        raise ValueError("--terrain_type is required for fixed-terrain diagnosis")

    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.env.num_envs = 64
    env_cfg.env.episode_length_s = 100
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = True
    env_cfg.terrain.terrain_kwargs = {"type": args.terrain_type}
    env_cfg.terrain.difficulty = args.terrain_difficulty
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.measure_heights = True
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

    # At the nominal 0.4 m/s command, 600 policy steps cover about 4.8 m.
    # This is long enough to leave the terrain generator's 3 m center platform.
    warmup_steps, sample_steps = 20, 600
    totals = {
        "flight": 0.0,
        "double_support": 0.0,
        "stance_contact": 0.0,
        "speed": 0.0,
        "vertical_speed": 0.0,
        "slip": 0.0,
        "slip_samples": 0.0,
        "touchdown_error": 0.0,
        "touchdowns": 0.0,
        "resets": 0.0,
        "hold": 0.0,
        "target_surface_error": 0.0,
        "target_surface_samples": 0.0,
    }

    with torch.inference_mode():
        for step in range(warmup_steps + sample_steps):
            actions = runner.get_inference_actions()
            runner.set_actions(actions)
            env.step()
            if step >= warmup_steps:
                position, _, contact, _ = env._canonical_foot_values()
                batch = torch.arange(env.num_envs, device=env.device)
                swing = env.swing_foot_index
                totals["flight"] += (~contact.any(dim=1)).sum().item()
                totals["double_support"] += contact.all(dim=1).sum().item()
                totals["stance_contact"] += contact[
                    batch, env.stance_foot_index
                ].sum().item()
                totals["speed"] += env.root_states[:, 7].sum().item()
                totals["vertical_speed"] += env.root_states[:, 9].abs().sum().item()
                totals["resets"] += env.reset_buf.sum().item()
                totals["hold"] += env.goal_generator.hold_still.sum().item()

                target = env.goal_generator.world_target_pos[batch, swing]
                surface = env.goal_generator._terrain_heights(target[:, :2])
                moving = ~env.goal_generator.hold_still
                totals["target_surface_error"] += (
                    target[moving, 2] - surface[moving]
                ).abs().sum().item()
                totals["target_surface_samples"] += moving.sum().item()

                tracking_error, _, _ = env._swing_tracking_errors()
                touchdown = env.touchdown_event[batch, swing]
                totals["touchdowns"] += touchdown.sum().item()
                totals["touchdown_error"] += tracking_error[
                    touchdown, :2
                ].norm(dim=-1).sum().item()

                foot_velocity = env.rigid_body_state[
                    :, env.feet_ids, 7:9
                ][:, [1, 0]].norm(dim=-1)
                totals["slip"] += foot_velocity[contact].sum().item()
                totals["slip_samples"] += contact.sum().item()
            runner.reset_envs()

    samples = env.num_envs * sample_steps
    touchdown_samples = max(totals["touchdowns"], 1.0)
    slip_samples = max(totals["slip_samples"], 1.0)
    print(f"terrain_type: {args.terrain_type}")
    print(f"terrain_difficulty: {args.terrain_difficulty:.3f}")
    print(f"samples: {samples}")
    print(f"mean_world_forward_speed: {totals['speed'] / samples:.4f} m/s")
    print(f"mean_abs_vertical_speed: {totals['vertical_speed'] / samples:.4f} m/s")
    print(f"stance_contact_ratio: {totals['stance_contact'] / samples:.4f}")
    print(f"flight_ratio: {totals['flight'] / samples:.4f}")
    print(f"double_support_ratio: {totals['double_support'] / samples:.4f}")
    print(f"mean_touchdown_xy_error: {totals['touchdown_error'] / touchdown_samples:.4f} m")
    print(f"touchdown_samples: {int(totals['touchdowns'])}")
    print(f"mean_contact_foot_slip_speed: {totals['slip'] / slip_samples:.4f} m/s")
    print(f"hold_ratio: {totals['hold'] / samples:.4f}")
    target_samples = max(totals["target_surface_samples"], 1.0)
    print(f"mean_target_surface_error: {totals['target_surface_error'] / target_samples:.6f} m")
    print(f"resets: {int(totals['resets'])}")


if __name__ == "__main__":
    diagnose(get_args())
