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

import pickle
from datetime import datetime


def collect_prey_data(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    max_num_envs = 1000
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, max_num_envs)
    env_cfg.env.debug_viz = False
    env_cfg.env.capture_dist = 0 #THIS IS NEVER TRUE SO THEY WILL NEVER BE CAPTURED
    env_cfg.env.agent_turn_freq = [50,150]
    env_cfg.env.agent_straight_freq = [100,200]

    # prepare environment
    print("[collect_prey_data] making environment...")
    env, _ = task_registry.make_dec_env(name=args.task, args=args, env_cfg=env_cfg)
    print("[collect_prey_data] getting observations for both agents..")
    obs_agent = env.get_observations_agent()
    obs_robot = env.get_observations_robot()

    # load policies of agent and robot
    evol_checkpoint = 0
    learn_checkpoint = 1600
    train_cfg.runner.resume_robot = True
    train_cfg.runner.resume_agent = True

    train_cfg.runner.load_run = 'May08_22-48-07_'  # 5Hz, 8-step future, pi(x, x_future)

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


    # data saving info.
    rel_state_data = None
    num_sim_steps = int(env_cfg.env.episode_length_s / env_cfg.env.robot_hl_dt)

    for i in range(num_sim_steps):
        print("Collecting data at tstep: ", i, " / ", num_sim_steps, "...")

        # save the current agent state.
        agent_state = env.ll_env.root_states[env.ll_env.agent_indices, :3]
        robot_state = env.ll_env.root_states[env.ll_env.robot_indices, :3]
        rel_state = agent_state - robot_state
        if rel_state_data is None:
            rel_state_data = np.array(rel_state.unsqueeze(1).cpu().numpy())
        else:
            rel_state_data = np.append(rel_state_data, rel_state.unsqueeze(1).cpu().numpy(), axis=1)

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

    print("DONE! Saving data...")
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
    filename = LEGGED_GYM_ROOT_DIR + "/legged_gym/predictors/data/rel_state_data_" + str(max_num_envs) + "agents.pickle"
    data_dict = {"rel_state_data": rel_state_data, 
                 "dt": env_cfg.env.robot_hl_dt,
                 "traj_length_s": env_cfg.env.episode_length_s}

    with open(filename, 'wb') as handle:
        pickle.dump(data_dict, handle)

if __name__ == '__main__':
    args = get_dec_args()
    collect_prey_data(args)
