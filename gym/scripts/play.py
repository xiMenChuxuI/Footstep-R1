# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gym import LEGGED_GYM_ROOT_DIR
import os
import time
from colorama import Fore

import isaacgym
from isaacgym import gymtorch
from gym.envs import *
from gym.envs.guidance.foothold_goal_sampler import sample_heightfield
from gym.utils import get_args, task_registry, Logger, VisualizationRecorder, ScreenShotter, AnalysisRecorder, CSVLogger, DictLogger, SuccessRater, GridVideoRecorder
from gym.scripts.plotting import LivePlotter

import numpy as np
import torch


def _sample_separated_spawn_xy(origins, jitter, min_spacing):
    """Sample small per-cell offsets while enforcing global XY separation."""
    result = origins[:, :2].clone()
    for env_id in range(origins.shape[0]):
        candidates = origins[env_id, :2] + (
            2.0 * torch.rand(64, 2, device=origins.device) - 1.0
        ) * jitter
        if env_id == 0:
            result[env_id] = candidates[0]
            continue
        clearance = torch.cdist(candidates, result[:env_id]).amin(dim=1)
        valid = (clearance >= min_spacing).nonzero(as_tuple=False).flatten()
        chosen = valid[0] if valid.numel() else clearance.argmax()
        result[env_id] = candidates[chosen]
    if origins.shape[0] > 1:
        distances = torch.cdist(result, result)
        distances.fill_diagonal_(torch.inf)
        actual_spacing = distances.amin().item()
        if actual_spacing < min_spacing:
            raise ValueError(
                f"Cannot satisfy --multi_min_spacing={min_spacing:.3f}; "
                f"best sampled spacing was {actual_spacing:.3f} m"
            )
    return result


def _configure_multi_robot_demo(env, env_cfg, args):
    if env.num_envs <= 1 or args.disable_multi_randomization:
        return None
    if not hasattr(env, "goal_generator") or not hasattr(
        env.goal_generator, "set_route_overrides"
    ):
        print("Multi-robot route randomization is unavailable for this task.")
        return None

    device = env.device
    count = env.num_envs
    env_ids = torch.arange(count, device=device)
    spawn_xy = _sample_separated_spawn_xy(
        env.env_origins,
        max(args.multi_spawn_jitter, 0.0),
        max(args.multi_min_spacing, 0.0),
    )
    heading_range = abs(args.multi_heading_range)
    heading_bin = 2.0 * heading_range / count
    headings_deg = -heading_range + (
        torch.arange(count, device=device) + torch.rand(count, device=device)
    ) * heading_bin
    headings = torch.deg2rad(headings_deg[torch.randperm(count, device=device)])
    speed_low = min(args.multi_speed_min, args.multi_speed_max)
    speed_high = max(args.multi_speed_min, args.multi_speed_max)
    speed_bin = (speed_high - speed_low) / count
    speeds = speed_low + (
        torch.arange(count, device=device) + torch.rand(count, device=device)
    ) * speed_bin
    speeds = speeds[torch.randperm(count, device=device)]

    env.root_states[:, :2] = spawn_xy
    if env.cfg.terrain.mesh_type in ("heightfield", "trimesh"):
        terrain_cfg = env.terrain.cfg
        surface_height = sample_heightfield(
            env.height_samples,
            spawn_xy,
            terrain_cfg.border_size,
            terrain_cfg.horizontal_scale,
            terrain_cfg.vertical_scale,
        )
        env.root_states[:, 2] = surface_height + env_cfg.init_state.pos[2]
    yaw_quat = torch.zeros(count, 4, device=device)
    yaw_quat[:, 2] = torch.sin(0.5 * headings)
    yaw_quat[:, 3] = torch.cos(0.5 * headings)
    env.root_states[:, 3:7] = yaw_quat
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states)
    )
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    phase = env.phase.clone()
    phase_count = env.phase_count.clone()
    update_count = env.update_count.clone()
    env._update_robot_states()
    env.phase[:] = phase
    env.phase_count[:] = phase_count
    env.update_count[:] = update_count

    nominal_speed = 0.4
    step_scale = speeds / nominal_speed
    env.goal_generator.set_route_overrides(env_ids, headings, step_scale)
    env.goal_generator.reset(env_ids)
    env.foothold_goal[:] = env.goal_generator.get_goal()
    velocity_commands = torch.stack(
        (speeds * torch.cos(headings), speeds * torch.sin(headings)), dim=-1
    )
    env.commands[:, :2] = velocity_commands

    distances = torch.cdist(spawn_xy, spawn_xy)
    distances.fill_diagonal_(torch.inf)
    print("Multi-robot playback randomization")
    print(f"  robots: {count}")
    print(f"  minimum spawn spacing: {distances.amin().item():.3f} m")
    print(
        f"  speed range: {speeds.amin().item():.3f}.."
        f"{speeds.amax().item():.3f} m/s"
    )
    print(
        f"  heading range: {torch.rad2deg(headings).amin().item():.1f}.."
        f"{torch.rad2deg(headings).amax().item():.1f} deg"
    )
    return velocity_commands

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    requested_envs = args.num_envs if args.num_envs is not None else 1
    if args.video_2x2:
        requested_envs = max(requested_envs, 4)
    env_cfg.env.num_envs = requested_envs
    env_cfg.env.env_spacing = 2.
    env_cfg.env.episode_length_s = int(1e7) # 5, int(1e7)
    env_cfg.seed = 1
    if args.terrain_mode == "plane":
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.mesh_type = 'plane'
        env_cfg.terrain.measure_heights = False
    else:
        if args.terrain_level is not None:
            env_cfg.terrain.max_init_terrain_level = args.terrain_level
        if args.terrain_type is not None:
            env_cfg.terrain.curriculum = False
            env_cfg.terrain.selected = True
            env_cfg.terrain.terrain_kwargs = {"type": args.terrain_type}
            env_cfg.terrain.difficulty = args.terrain_difficulty
            grid_rows = int(np.ceil(np.sqrt(env_cfg.env.num_envs)))
            grid_cols = int(np.ceil(env_cfg.env.num_envs / grid_rows))
            env_cfg.terrain.num_rows = grid_rows
            env_cfg.terrain.num_cols = grid_cols
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False # True
    env_cfg.init_state.reset_mode = 'reset_to_basic' # 'reset_to_basic', 'reset_to_range'
    env_cfg.commands.resampling_time = -1 # -1, 3, 5, 15
    env_cfg.commands.curriculum = False # True, False
    # env_cfg.viewer.pos = [0, -1.5, 1] # [0, -3.5, 3]
    env_cfg.viewer.pos = [0, -2.5, .5] # [0, -3.5, 3]
    # env_cfg.viewer.lookat = [0, 0, 0] # [1, 1.5, 0]
    env_cfg.viewer.lookat = [0, 0.5, 0] # [1, 1.5, 0]
    env_cfg.commands.ranges.lin_vel_x = [0., 0.] # [0., 0.], [-2., 2.]
    env_cfg.commands.ranges.lin_vel_y = 0. # 0., 1.
    env_cfg.commands.ranges.yaw_vel = 0. # 0., 1.

    if env_cfg.__class__.__name__ == 'HumanoidControllerCfg':
        env_cfg.commands.adjust_step_command = False # True, False
        env_cfg.commands.adjust_prob = 0.05
        env_cfg.commands.sample_angle_offset = 20
        env_cfg.commands.sample_radius_offset = 0.05 # 0.05

        env_cfg.commands.ranges.sample_period = [35, 36] # [20, 21], [35, 36]
        env_cfg.commands.ranges.dstep_width = [0.3, 0.3]

    # * prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # * load policy
    train_cfg.runner.resume = True
    policy_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy_runner.alg.actor_critic.eval()
    multi_robot_commands = _configure_multi_robot_demo(env, env_cfg, args)

    # * export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported')
        policy_runner.export(path)
        print('Exported policy model to: ', path)
    
    start = time.time()
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    default_camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)

    # log_states = [ "root_states", "step_length", "step_width" ] # "base_lin_vel", "base_pos", "dof_pos", 
    # log_commands = [ "commands", "dstep_length", "dstep_width"] # "step_commands" 
    log_states = [ "contact_forces" ] # "base_lin_vel", "base_pos", "dof_pos", 
    log_commands = [] # "step_commands" 
    log_rewards = [] #[ "wrong_contact", "step_location", ]
    max_it = args.video_length if args.video_2x2 else int(1e6)
    screenshotter = ScreenShotter(env, train_cfg.runner.experiment_name, policy_runner.log_dir)
    grid_recorder = None
    if args.video_2x2:
        robot_ids = [int(value.strip()) for value in args.video_robot_ids.split(",")]
        grid_recorder = GridVideoRecorder(
            env,
            train_cfg.runner.experiment_name,
            robot_ids,
            width=args.video_width,
            height=args.video_height,
            fps=args.video_fps,
            run_name=args.run_name or train_cfg.runner.run_name,
        )

    if LIVE_PLOT:
        live_plotter = LivePlotter(log_states, log_commands, log_rewards, robot_index, joint_index, env.dt)
    if RECORD_FRAMES:
        recorder = AnalysisRecorder(env, train_cfg.runner.experiment_name, policy_runner.log_dir)
    if SAVE_CSV:
        csv_logger = CSVLogger(env, train_cfg.runner.experiment_name, policy_runner.log_dir, max_it)
    if SAVE_DICT:
        dict_logger = DictLogger(env, train_cfg.runner.experiment_name, policy_runner.log_dir, max_it)
    if CHECK_SUCCESS_RATE:
        success_rater = SuccessRater(env, train_cfg.runner.experiment_name, policy_runner.log_dir)
        test_episodes = 10000

    for i in range(max_it):
        actions = policy_runner.get_inference_actions()
        policy_runner.set_actions(actions)
        env.step()
        policy_runner.reset_envs()
        if multi_robot_commands is not None:
            env.commands[:, :2] = multi_robot_commands

        if grid_recorder is not None:
            grid_recorder.record_frame()
        
        if env.screenshot:
            image = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handle, isaacgym.gymapi.IMAGE_COLOR)
            image = image.reshape(image.shape[0], -1, 4)[..., :3]
            screenshotter.screenshot(image)
            env.screenshot = False

        if CUSTOM_COMMANDS:
            # * Scenario 1 (For flat terrain)
            if (i+1) == 500:
                env.commands[:, 0] = 0.5
                env.commands[:, 1] = 0.
            elif (i+1) == 1000: 
                env.commands[:, 0] = 1.0
                env.commands[:, 1] = 0.

        if MOVE_CAMERA and not env.headless:
            if env.num_envs > 1:
                positions = env.base_pos[:, :3].cpu().numpy()
                group_center = positions.mean(axis=0)
                group_span = max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1]), 3.0)
                camera_offset = np.array([0.0, -1.4 * group_span, 0.8 * group_span + 2.0])
                env.set_camera(group_center + camera_offset, group_center)
            else:
                camera_position += camera_vel * env.dt
                camera_position = default_camera_position + env.base_pos[robot_index].cpu().numpy()
                env.set_camera(camera_position, camera_position + camera_direction)

        if LIVE_PLOT:
            live_plotter.log(
                env.reset_buf[robot_index].item(),
                env.manual_reset_flag,
                states_dict = {
                    state: eval(f"env.{state}[{robot_index}]", {"env": env}) for state in log_states
                },
                commands_dict = {
                    command: eval(f"env.{command}[{robot_index}]", {"env": env}) for command in log_commands
                },
                rewards_dict = {
                    reward: eval(f"env.episode_sums['{reward}'][{robot_index}]", {"env": env}) for reward in log_rewards
                },
                extras_dict = {
                    'update_count': env.update_count[robot_index].item(),
                    'update_commands_ids': env.update_commands_ids[robot_index].item(),
                }
            )
            live_plotter.plot()
        
        if RECORD_FRAMES:
            image = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handle, isaacgym.gymapi.IMAGE_COLOR)
            image = image.reshape(image.shape[0], -1, 4)[..., :3]
            recorder.log(
                image,
                states_dict = {
                    state: eval(f"env.{state}[{robot_index}]", {"env": env}) for state in log_states
                },
                commands_dict = {
                    command: eval(f"env.{command}[{robot_index}]", {"env": env}) for command in log_commands
                }
            )
            if env.record_done:
                recorder.save_and_exit()

        if SAVE_CSV:
            csv_logger.log()
            if env.record_done:
                csv_logger.save_and_exit()
        
        if SAVE_DICT:
            dict_logger.log()
            if env.record_done:
                dict_logger.save_and_exit()
        
        if CHECK_SUCCESS_RATE:
            if env.record_done:
                stop = time.time()
                print("collection_time:", stop - start)
                success_rater.save_and_exit()
            success_rater.log(i, test_episodes, env.reset_buf.sum().item(), env.timed_out.sum().item())

    if grid_recorder is not None:
        grid_recorder.close()



if __name__ == '__main__':
    EXPORT_POLICY = True # True, False
    CUSTOM_COMMANDS = False # True, False
    MOVE_CAMERA = True # True, False
    LIVE_PLOT = False # True, False
    RECORD_FRAMES = False # True, False
    SAVE_CSV = False # True, False
    SAVE_DICT = False # True, False
    CHECK_SUCCESS_RATE = False # True, False
    args = get_args()

    # # * custom loading
    # args.load_files = True # True, False
    # args.load_run = 'Feb06_00-27-24_sf' # load run name
    # args.checkpoint = '1000'

    with torch.inference_mode():
        play(args)
