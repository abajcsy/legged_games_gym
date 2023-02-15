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


def play_dec_game(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    max_num_envs = 4
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, max_num_envs)
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 4    # number of terrain rows (levels)
    env_cfg.terrain.num_cols = 4    # number of terrain cols (types)
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # # prepare environment
    print("[play_dec_game] making environment...")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("[play_dec_game] getting observations for both agents..")
    obs_agent = env.get_observations_agent()
    obs_robot = env.get_observations_robot()

    # load policies of agent and robot
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = 'Feb15_12-14-29_'
    train_cfg.runner.checkpoint = 1400 # TODO: WITHOUT THIS IT GRABS WRONG CHECKPOINT
    dec_ppo_runner, train_cfg = task_registry.make_dec_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy_agent = dec_ppo_runner.get_inference_policy(agent_id=0, device=env.device)
    policy_robot = dec_ppo_runner.get_inference_policy(agent_id=1, device=env.device)

    # camer info.
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10 * int(env.max_episode_length)):
        actions_agent = policy_agent(obs_agent.detach())
        actions_robot  = policy_robot(obs_robot .detach())
        obs_agent, obs_robot , _, _, rews_agent, rews_robot, dones, infos = env.step(actions_agent.detach(), actions_robot.detach())

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                        'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

if __name__ == '__main__':
    args = get_args()
    play_dec_game(args)
