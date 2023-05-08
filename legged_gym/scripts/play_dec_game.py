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


def play_dec_game(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    max_num_envs = 3
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, max_num_envs)
    env_cfg.env.debug_viz = True

    # # prepare environment
    print("[play_dec_game] making environment...")
    env, _ = task_registry.make_dec_env(name=args.task, args=args, env_cfg=env_cfg)
    print("[play_dec_game] getting observations for both agents..")
    obs_agent = env.get_observations_agent()
    obs_robot = env.get_observations_robot()

    # load policies of agent and robot
    evol_checkpoint = 0
    learn_checkpoint = 0
    #learn_checkpoint = 1400
    train_cfg.runner.resume_robot = True
    train_cfg.runner.resume_agent = True

    # train_cfg.runner.load_run = 'Apr03_15-16-53_' # Policy without obstacles, in 'dec_high_level_game'
    # train_cfg.runner.load_run = 'Apr21_19-16-43_' # true rel yaw local, FULL FOV
    # train_cfg.runner.load_run = 'Apr22_16-22-37_' # KF rel yaw local, FULL FOV
    # train_cfg.runner.load_run = 'Apr22_13-12-56_' # KF, ZED FOV NO curriculum, w/ covariance, cmd clip
    # train_cfg.runner.load_run = 'Apr22_13-54-32_' # KF, ZED FOV w/ curriculum, w/ covariance, cmd clip
    # train_cfg.runner.load_run = 'Apr22_15-35-56_' # local rel pos, ZED FOV
    # train_cfg.runner.load_run = 'Apr21_17-39-32_' # local rel pos, FULL FOV

    #train_cfg.runner.load_run = 'Apr23_09-40-25_' # WORKING POLICY with BC
    #train_cfg.runner.load_run = 'Apr26_18-08-49_' # policy with -||u||^2

    #train_cfg.runner.load_run = 'Apr28_09-17-01_' # policy with (xrel, upred^t:t:10)

    #train_cfg.runner.load_run = 'Apr30_18-18-17_' # perfect-state policy,5 hz HL; fixed weaving agent path
    # train_cfg.runner.load_run = 'May01_20-23-22_' # perfect-state policy, 5 hz HL; more stochastic agent path
    #train_cfg.runner.load_run = 'May02_06-30-16_' # limited FOV policy, 5 hz HL; fixed weaving agent path

    #train_cfg.runner.load_run = 'May03_19-55-57_' # 0.02 dt, 3-step history, only pursuit reward
    #train_cfg.runner.load_run = 'May04_03-57-17_' # 0.02 dt, 3-step history, exp_pursuit + 1000*term
    # train_cfg.runner.load_run = 'May04_03-59-49_' # 0.2 dt, 3-step history, exp_pursuit + 1000*term
    # train_cfg.runner.load_run ='May04_04-48-07_' # 0.2 dt, 8-step history, exp_pursuit + 1000*term

    # train_cfg.runner.load_run = 'May04_10-32-17_' # 0.02 dt, 3-step history, pursuit; world frame position, angle in body frame
    #train_cfg.runner.load_run = 'May04_12-15-56_' #'May04_13-19-17_' # 0.02 dt, 3-step history, pursuit; world frame position, angle and robot ctrls in body frame

    #train_cfg.runner.load_run = 'May04_13-56-21_' # 0.02 dt, 3-step history, pursuit rew; body frame observations
    #train_cfg.runner.load_run = 'May04_14-54-46_' # 0.02 dt, 3-step history, pursuit + terminal rew; body frame observations

    # train_cfg.runner.load_run = 'May04_18-26-33_' # 0.02 dt, 3-step FUTURE, pursuit + terminal rew; body frame observations
    #train_cfg.runner.load_run = 'May05_04-36-41_' #'May05_11-00-12_'  # 0.2 dt, 3-step FUTURE, pursuit + terminal rew; body frame observations
    # train_cfg.runner.load_run = 'May05_04-47-21_'   # 0.2 dt, 8-step FUTURE, pursuit + terminal rew; body frame observations

    # train_cfg.runner.load_run = 'May06_20-18-47_' #'May06_09-41-34_' #'May05_23-57-10_'
    #train_cfg.runner.load_run = 'May06_23-05-56_' # pi(x, dx) with 0.5 * foveation + pursuit + 100 * terminal
    # train_cfg.runner.load_run = 'May07_00-00-04_' # pi(x, dx) with 5.0 * foveation + pursuit + 100 * terminal

    #train_cfg.runner.load_run = 'May07_19-19-40_'

    train_cfg.runner.load_run = 'May08_14-00-00_'

    # train_cfg.runner.load_run = 'May08_19-17-39_' # pi(x, dx, elapsed_t); rew = pursuit -0.1 * elapsed_t + 100 * terminal

    train_cfg.runner.learn_checkpoint_robot = learn_checkpoint # TODO: WITHOUT THIS IT GRABS WRONG CHECKPOINT
    train_cfg.runner.learn_checkpoint_agent = learn_checkpoint
    train_cfg.runner.evol_checkpoint_robot = evol_checkpoint  # TODO: WITHOUT THIS IT GRABS WRONG CHECKPOINT
    train_cfg.runner.evol_checkpoint_agent = evol_checkpoint

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
        # print("[play_dec_game] current obs_robot: ", obs_robot.detach())
        actions_agent = policy_agent(obs_agent.detach())
        actions_robot = policy_robot(obs_robot.detach())
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
    args = get_dec_args()
    play_dec_game(args)
