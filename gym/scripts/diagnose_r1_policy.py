"""Measure contact and vertical-motion statistics for an R1 checkpoint."""

import isaacgym
import torch

from gym.envs import *
from gym.utils import get_args, task_registry


def diagnose(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.env.num_envs = 64
    env_cfg.env.episode_length_s = 100
    env_cfg.init_state.pos[2] = 0.715
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

    warmup_steps = 20
    sample_steps = 200
    flight_count = 0
    double_support_count = 0
    stance_contact_count = 0
    reset_count = 0
    vertical_speed = []
    forward_speed = []
    action_rms = []
    swing_xy_error = []
    touchdown_xy_error_sum = 0.0
    touchdown_x_error_sum = 0.0
    touchdown_y_error_sum = 0.0
    touchdown_yaw_error_sum = 0.0
    touchdown_samples = 0
    contact_slip_speed_sum = 0.0
    contact_slip_samples = 0

    with torch.inference_mode():
        for step in range(warmup_steps + sample_steps):
            actions = runner.get_inference_actions()
            runner.set_actions(actions)
            env.step()
            if step >= warmup_steps:
                _, _, contact, _ = env._canonical_foot_values()
                batch = torch.arange(env.num_envs, device=env.device)
                flight_count += (~contact.any(dim=1)).sum().item()
                double_support_count += contact.all(dim=1).sum().item()
                stance_contact_count += contact[
                    batch, env.stance_foot_index
                ].sum().item()
                reset_count += env.reset_buf.sum().item()
                vertical_speed.append(env.root_states[:, 9].abs().mean().item())
                forward_speed.append(env.root_states[:, 7].mean().item())
                action_rms.append(actions.square().mean().sqrt().item())
                tracking_error, yaw_error, swing_contact = env._swing_tracking_errors()
                swing_xy_error.append(
                    tracking_error[:, :2].norm(dim=-1).mean().item()
                )
                local_phase = torch.where(
                    env.phase[:, 0] < 0.5,
                    env.phase[:, 0] * 2.0,
                    (env.phase[:, 0] - 0.5) * 2.0,
                )
                touchdown = swing_contact & (
                    local_phase > env.cfg.rewards.touchdown_phase
                )
                touchdown_samples += touchdown.sum().item()
                touchdown_xy_error_sum += (
                    tracking_error[touchdown, :2].norm(dim=-1).sum().item()
                )
                touchdown_x_error_sum += tracking_error[touchdown, 0].abs().sum().item()
                touchdown_y_error_sum += tracking_error[touchdown, 1].abs().sum().item()
                touchdown_yaw_error_sum += yaw_error[touchdown].abs().sum().item()
                foot_xy_velocity = env.rigid_body_state[
                    :, env.feet_ids, 7:9
                ][:, [1, 0]]
                contact_slip_speed_sum += (
                    foot_xy_velocity.norm(dim=-1)[contact].sum().item()
                )
                contact_slip_samples += contact.sum().item()
            runner.reset_envs()

    samples = env.num_envs * sample_steps
    print(f"samples: {samples}")
    print(f"flight_ratio: {flight_count / samples:.4f}")
    print(f"double_support_ratio: {double_support_count / samples:.4f}")
    print(f"stance_contact_ratio: {stance_contact_count / samples:.4f}")
    print(f"mean_abs_vertical_speed: {sum(vertical_speed) / len(vertical_speed):.4f} m/s")
    print(f"mean_world_forward_speed: {sum(forward_speed) / len(forward_speed):.4f} m/s")
    print(f"mean_swing_xy_error: {sum(swing_xy_error) / len(swing_xy_error):.4f} m")
    if touchdown_samples:
        print(f"mean_touchdown_xy_error: {touchdown_xy_error_sum / touchdown_samples:.4f} m")
        print(f"mean_touchdown_x_error: {touchdown_x_error_sum / touchdown_samples:.4f} m")
        print(f"mean_touchdown_y_error: {touchdown_y_error_sum / touchdown_samples:.4f} m")
        print(f"mean_touchdown_yaw_error: {touchdown_yaw_error_sum / touchdown_samples:.4f} rad")
    print(f"touchdown_samples: {touchdown_samples}")
    if contact_slip_samples:
        print(
            "mean_contact_foot_slip_speed: "
            f"{contact_slip_speed_sum / contact_slip_samples:.4f} m/s"
        )
    print(f"mean_action_rms: {sum(action_rms) / len(action_rms):.4f} rad")
    print(f"resets: {reset_count}")


if __name__ == "__main__":
    diagnose(get_args())
