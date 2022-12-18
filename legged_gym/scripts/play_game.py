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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from isaacgym import gymapi

def goal_reach_lin_yaw_cmd(curr_pos, goal_pos, ranges):
    dxy = goal_pos - curr_pos
    yaw = torch.atan2(dxy[:, 1], dxy[:, 0])

    command_lin_vel_x = torch.clamp(dxy[0,0], min=ranges.lin_vel_x[0], max=ranges.lin_vel_x[1])
    command_lin_vel_y = torch.clamp(dxy[0,1], min=ranges.lin_vel_y[0], max=ranges.lin_vel_y[1])
    command_yaw = torch.clamp(yaw, min=ranges.ang_vel_yaw[0], max=ranges.ang_vel_yaw[1])

    return command_lin_vel_x, command_lin_vel_y, command_yaw

def play_game(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    max_num_envs = 5
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, max_num_envs)
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 4    # number of terrain rows (levels)
    env_cfg.terrain.num_cols = 4    # number of terrain cols (types)
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # if flag is true, then uses a pre-trained RL policy to avoid predator
    #            false, then simulates the high-level commands to a fixed goal
    run_rl_policy = True

    if run_rl_policy:
        # prepare environment
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

        obs = env.get_observations()
        # load policy
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = 'hl_vanilla_ll_a1_vanilla'
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)

        # export policy as a jit module (used to run it from C++)
        if EXPORT_POLICY:
            path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print('Exported policy as jit script to: ', path)

        camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([1., 1., 0.])
        camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        img_idx = 0

        for i in range(10 * int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                            'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)
    else:

        # prepare environment
        print("     Preparing high-level environment...")
        hl_env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        print("     Finished initializing high level and low level environment")

        # get the first observation
        obs = hl_env.compute_observations() # this is zero at first!

        # set the high-level goal position
        goal_xy_pos = torch.tensor([[32, 8]], device=hl_env.device)

        # simulation loop
        for i in range(10 * int(hl_env.max_episode_length)):

            # get the high-level command and set it as part of the env
            prey_xy_pos = hl_env.ll_env.root_states[hl_env.ll_env.prey_indices, :2]

            print("Agent positions:")
            print("     prey (robot): ", prey_xy_pos)
            print("     predator (red dot): ", hl_env.predator_pos[:2])

            command_lin_vel_x, command_lin_vel_y, command_yaw = goal_reach_lin_yaw_cmd(prey_xy_pos, goal_xy_pos,
                                                                                       env_cfg.commands.ranges)
            print("PREY     -- dist to goal pos (env 1):", torch.norm(prey_xy_pos[0, :] - goal_xy_pos[:]))
            print("PREDATOR -- dist to prey (env 1):", torch.norm(prey_xy_pos[0, :] - hl_env.predator_pos[0, :2]))
            # print("dist to goal pos (env 2):", torch.norm(curr_pos[1, :] - goal_pos[:]))

            # set the high-level command
            hl_command = torch.zeros(hl_env.ll_env.num_envs, hl_env.num_actions, device=hl_env.device, requires_grad=False)  # of shape [num_envs, 4]
            hl_command[:, 0] = command_lin_vel_x  # * env.commands_scale[0] # lin vel x
            hl_command[:, 1] = command_lin_vel_y  # * env.commands_scale[1] # lin vel y
            hl_command[:, 2] = command_yaw  # * env.commands_scale[2] # ang vel yaw
            hl_command[:, 3] = 0

            # forward simulate the low-level actions, updates observation buffer too
            hl_obs, _, hl_rews, hl_dones, hl_infos = hl_env.step(hl_command)


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play_game(args)
