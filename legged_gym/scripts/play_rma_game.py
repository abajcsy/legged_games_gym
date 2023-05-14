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
from legged_gym.utils import get_dec_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from isaacgym import gymapi


def play_rma_game(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    max_num_envs = 6
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, max_num_envs)

    env_cfg.env.debug_viz = True
    env_cfg.commands.use_joypad = False

    # # prepare environment
    print("[play_rma_game] making environment...")
    env, _ = task_registry.make_dec_env(name=args.task, args=args, env_cfg=env_cfg)
    print("[play_rma_game] getting observations for both agents..")
    obs_agent = env.get_observations_agent()
    privileged_obs_robot = env.get_privileged_observations_robot()
    obs_robot = env.get_observations_robot()

    # load policies of agent and robot
    evol_checkpoint = 0
    learn_checkpoint = 1600
    train_cfg.runner.resume_robot = True # only load robot
    train_cfg.runner.resume_agent = False

    train_cfg.runner.load_run = 'phase_2_policy_v2'

    train_cfg.runner.learn_checkpoint_robot = learn_checkpoint # TODO: WITHOUT THIS IT GRABS WRONG CHECKPOINT
    train_cfg.runner.evol_checkpoint_robot = evol_checkpoint  # TODO: WITHOUT THIS IT GRABS WRONG CHECKPOINT

    dagger_runner, train_cfg = task_registry.make_dagger_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy_robot = dagger_runner.get_estimator_inference_policy(device=env.device)

    # camer info.
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10 * int(env.max_episode_length)):

        # Get the estimator information
        h = torch.zeros((dagger_runner.alg.actor_critic.estimator.num_layers, env.num_envs,
                         dagger_runner.alg.actor_critic.estimator.hidden_size), device=env.device, requires_grad=True)
        c = torch.zeros((dagger_runner.alg.actor_critic.estimator.num_layers, env.num_envs,
                         dagger_runner.alg.actor_critic.estimator.hidden_size), device=env.device, requires_grad=True)
        hidden_state = (h, c)

        # get the estimator latent
        estimator_obs = obs_robot.clone().unsqueeze(0)
        zhat, hidden_state = dagger_runner.alg.actor_critic.estimate_latent(estimator_obs, hidden_state)

        # actions_robot = dagger_runner.alg.actor_critic.estimate_actor(obs_robot.detach(), zhat[0].detach())
        actions_robot = policy_robot(obs_robot.detach(), zhat[0].detach())

        # zexpert = dagger_runner.alg.actor_critic.acquire_latent(privileged_obs_robot)
        # actions_robot_expert = dagger_runner.alg.actor_critic.RMA_actor(privileged_obs_robot.detach())
        #
        # env_id = 0
        # print("(env 1) latent MSE:", 1/8*torch.sum((zexpert[env_id, :] - zhat[0, env_id, :]))**2)
        # print("(env 1) action dist:", torch.norm(actions_robot[0, :] - actions_robot_expert[0, :]))

        # spoof agent actions, since they are overridden anyway.
        actions_agent = torch.zeros(env.num_envs, env.num_actions_agent, device=env.device, requires_grad=False)
        obs_agent, obs_robot , _, privileged_obs_robot, rews_agent, rews_robot, dones, infos = env.step(actions_agent, actions_robot.detach())

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
    args = get_dec_args()
    play_rma_game(args)
