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
import pdb

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from collections import deque
import statistics
import numpy as np
import torch
import matplotlib.pyplot as plt

class Expert:
    def __init__(self, policy, device='cpu', baseDim=42):
        self.policy = policy
        self.policy.to(device)
        for params in self.policy.parameters():
            params.requires_grad = False
        self.device = device
        self.baseDim = baseDim 
        # 19 is size of priv info for the cvpr policy
        self.scan_dim = 51
        self.end_idx = baseDim + 23 + self.scan_dim #-self.geomDim*(self.n_futures+1) - 1
        
    def reorder(self, inp):
        reorder_indices = torch.tensor([3,4,5,0,1,2,9,10,11,6,7,8],device=self.device).long()
        return torch.index_select(inp,1,reorder_indices)
        # return torch.index_select(inp.cpu(),1,torch.LongTensor(reroder_indices)).to(self.device)

    def __call__(self, obs):
        with torch.no_grad():
            resized_obs = obs#[:, :self.end_idx]
            prop_latent = self.policy.prop_encoder(resized_obs[:,self.baseDim:-self.scan_dim])
            geom_latent = self.policy.geom_encoder(resized_obs[:,-self.scan_dim:])
            input_t = torch.cat((resized_obs[:,:self.baseDim], prop_latent, geom_latent), dim=1)
            action = self.policy.action_mlp(input_t)
            #make the policy output in sim order
            return self.reorder(action)
        
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    min_num_envs = 1
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, min_num_envs)
    # env_cfg.terrain.num_rows = 1
    # env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    # env_cfg.rewards.scales.episode_length = 1
    # env_cfg.rewards.scales.x_distance = 1
    # env_cfg.rewards.scales.y_distance = 1
    # env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # env.commands[:, 0] = 0.5  # lin_vel_x, lin_vel_y, ang_vel_yaw, heading
    # env.commands[:, 1] = 0
    # env.commands[:, 2] = 0
    # env.commands[:, 3] = 0

    obs = env.get_observations()
    env.reset() 
    # load policy
    train_cfg.runner.resume = True
    path = "/home/abajcsy/legged_workspace/legged_games_gym/logs/a1_wxv_rough/vision_policy_8000/policy_8000.pt"
    graph = torch.jit.load(path,map_location=env.device)
    policy = Expert(graph,env.device)
    
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    died_envs = torch.zeros(env_cfg.env.num_envs, device=env.device, dtype=torch.long)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    lenbuffer = deque(maxlen=100)
    actions = torch.zeros((env_cfg.env.num_envs,env_cfg.env.num_actions), device=env.device)
    for i in range(10*int(env.max_episode_length)):
        # actions = policy(obs.detach())
        # print("obs_buf.detach()",obs.detach())
        actions = policy(env.prior_obs_buf.detach())
        # actions[:,:12] = leg_actions

        obs, _, rews, dones, infos = env.step(actions.detach())
        
        cur_episode_length += 1
        new_ids = (dones > 0).nonzero(as_tuple=False)
        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        cur_episode_length[new_ids] = 0
        
        died_envs += (dones * ~env.time_out_buf) #record any env that has died once
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
       
if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
