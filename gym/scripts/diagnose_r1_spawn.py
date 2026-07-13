"""Compare R1 spawn heights under zero policy action."""

import isaacgym
import torch
from isaacgym import gymtorch

from gym.envs import *
from gym.utils import get_args, task_registry


def diagnose(args):
    cfg, _ = task_registry.get_cfgs(args)
    heights = [0.68, 0.69, 0.70, 0.71, 0.715, 0.72, 0.73, 0.75]
    cfg.env.num_envs = len(heights)
    cfg.env.episode_length_s = 100
    cfg.init_state.reset_mode = "reset_to_basic"
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_com_displacement = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.randomize_Kd_factor = False
    cfg.domain_rand.push_robots = False
    cfg.asset.terminate_after_contacts_on = []

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_envs(env_ids)
    requested_heights = torch.tensor(heights, device=env.device)
    env.root_states[:, 2] = requested_heights
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states)
    )
    env.dof_pos_target.zero_()

    print("requested_heights", heights, flush=True)
    report_steps = (0, 1, 2, 3, 5, 8, 11, 24, 49, 99, 199)
    for step in range(200):
        env.step()
        if step not in report_steps:
            continue
        foot_z = env.foot_states[:, :, 2]
        force_z = env.contact_forces[:, env.feet_ids, 2].abs().max(dim=1).values
        body_force = torch.norm(env.contact_forces, dim=-1)
        max_body_force, max_body_index = body_force.max(dim=1)
        body_names = {index: name for name, index in env.rigid_body_idx.items()}
        print(
            step,
            "base_z",
            [round(value, 4) for value in env.root_states[:, 2].cpu().tolist()],
            "vz",
            [round(value, 4) for value in env.root_states[:, 9].cpu().tolist()],
            "base_x",
            [round(value, 4) for value in env.root_states[:, 0].cpu().tolist()],
            "vx",
            [round(value, 4) for value in env.root_states[:, 7].cpu().tolist()],
            "foot_z_min",
            [round(value, 4) for value in foot_z.min(dim=1).values.cpu().tolist()],
            "foot_Fz_max",
            [round(value, 1) for value in force_z.cpu().tolist()],
            "body_F_max",
            [round(value, 1) for value in max_body_force.cpu().tolist()],
            "body",
            [body_names[index] for index in max_body_index.cpu().tolist()],
            flush=True,
        )


if __name__ == "__main__":
    diagnose(get_args())
