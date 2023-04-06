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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os


from isaacgym.torch_utils import *

from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor, device
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class LeggedRobotVision(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.iter_num = 0.
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        
        self._init_buffers()
        self._prepare_reward_function()
        
        self.init_done = True
        self.delay_curriculum = getattr(self.cfg.env, 'delay_curriulum', False)
        self.cur_delay = 0
        folder_path = "/home/abajcsy/legged_workspace/legged_games_gym/logs/a1_wxv_rough/vision_policy_8000/"
        mean_path = folder_path + "mean8000.csv"
        var_path = folder_path + "var8000.csv"
        mean = np.loadtxt(mean_path,dtype=np.float32)[0][-117:-1]
        self.obs_mean = torch.from_numpy(mean).to(self.device)[None,...].repeat(self.num_envs,1)
        
        var = np.loadtxt(var_path,dtype=np.float32)[0][-117:-1]
        self.obs_var = torch.from_numpy(np.sqrt(var)).to(self.device)[None,...].repeat(self.num_envs,1)
        
        if getattr( self.cfg.env, "num_observations_prior", 0) > 0:
            self.prior_obs_buf = torch.zeros(self.num_envs, self.cfg.env.num_observations_prior, device=self.device, dtype=torch.float)
        if getattr( self.cfg.env, "num_observations_reach", 0) > 0:
            self.reach_obs = torch.zeros(self.num_envs, self.cfg.env.num_observations_reach, device=self.device, dtype=torch.float)
        self.prior = getattr(self.cfg.env,'prior',True)
        # self.prior = False
        self.mass = 1.
        self.mass_noise = 1.
        self.friction_coeffs = torch.ones((self.num_envs,1,1),device=self.device)*self.cfg.terrain.static_friction
        self.vis = getattr(self.cfg.env,"vis",False)
        self.update_force = False
        self.forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.max_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        # self.force_signs =torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float)
        self.apply_force = torch.zeros((self.num_envs,1), device=self.device, dtype=torch.float)
        self.push_duration = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        self.push_interval = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        
        self.ee_indice = self.gym.find_actor_rigid_body_handle(self.envs[0],self.actor_handles[0],'wx200/ee_gripper_link')
        
        self.train_dog_only = getattr(self.cfg.env,"train_dog",False)
        self.teacher_obs_buf = torch.zeros_like(self.obs_buf,device=self.device)
        
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # print("raw actions",actions[0])
        # print("arm vel",self.dof_vel[0,12:],self.dof_names)
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        #set arm actions to be 0
        # self.actions *= self.cfg.control.action_filter.to(self.device)
        # print("cmd",self.commands[:10,0])
        # print("lin_vel",self.base_lin_vel[:10,0])
        
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # print("torque",self.torques)

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        # return torch.zeros_like(self.obs_buf), self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.roll,self.pitch,self.yaw = get_euler_xyz(self.base_quat)
        self.roll[torch.where(self.roll>torch.pi)] -= 2*torch.pi
        self.pitch[torch.where(self.pitch>torch.pi)] -= 2*torch.pi
        self.yaw[torch.where(self.yaw>torch.pi)] -= 2*torch.pi

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.compute_prior_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        

        self.last_dof_acc[:] = (self.last_dof_vel - self.dof_vel) / self.dt
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_pos[:] = self.root_states[:,:3]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        
        # collision_buf = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=(1,2)) > 50
        # print("col", torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=(1,2)) )

        # self.reset_buf |= collision_buf
        
        if 'roll' in self.cfg.asset.terminate_condition.keys():
            self.roll_term_buf = torch.abs(self.roll) > self.cfg.asset.terminate_condition['roll']
            self.reset_buf |= self.roll_term_buf
        if 'pitch' in self.cfg.asset.terminate_condition.keys():
            self.pitch_term_buf = torch.abs(self.pitch) > self.cfg.asset.terminate_condition['pitch']
            self.reset_buf |= self.pitch_term_buf
        if 'base_height' in self.cfg.asset.terminate_condition.keys():
            self.base_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) < self.cfg.asset.terminate_condition['base_height']
            self.reset_buf |= self.base_height_buf
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids, eval=False)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_dof_acc[env_ids] = 0.
        self.last_root_pos[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            if name in self.cfg.rewards.exponential_decay:
                reward_scales =  self.reward_scales[name]*((self.iter_num/self.cfg.rewards.max_iter)**3 )
            else:
                reward_scales =  self.reward_scales[name]
            # rew = self.reward_functions[i]() * self.reward_scales[name]
            rew = self.reward_functions[i]() * reward_scales
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
        
    def reorder(self, inp):
        reroder_indices = [3,4,5,0,1,2,9,10,11,6,7,8]
        return torch.index_select(inp.cpu(),1,torch.LongTensor(reroder_indices)).to(self.device)

    def compute_prior_observations(self):

        contact =  (torch.norm(self.contact_forces[:, self.feet_indices, :3],dim=-1)>0.001).type(torch.float32)
        contact_ind = torch.tensor([1,0,3,2],device=self.device).long()
        cmd = torch.stack([self.commands[:, 0]-0.5,self.commands[:,2],self.commands[:, 0]-0.5,self.commands[:,2]],dim=-1)

        self.prior_obs_buf = torch.cat((  
                            self.roll[...,None],
                            self.pitch[...,None],
                            self.reorder(self.dof_pos[:,:12]),
                            self.reorder(self.dof_vel[:,:12]),
                            self.reorder(self.actions[:,:12]),#policy out in real
                            cmd,
                            #priviledge obs 
                            #contact
                            torch.index_select(contact,1,contact_ind),
                            #mass
                            torch.tensor([1.,0,0],device=self.device)[None,...].repeat(self.num_envs,1)*self.mass_noise,
                            #motor strength
                            torch.ones_like(self.dof_pos[:,:12],device=self.device),
                            #friction
                            self.friction_coeffs.squeeze(-1).to(self.device),
                            self.base_lin_vel[:,:2],
                            0*self.base_ang_vel[:,-1].unsqueeze(1),
                            self.measured_heights
                            ),dim=-1)
        print("[pre normalization] prior_obs_buf measured heights: %0.2f", self.prior_obs_buf[:, -51:])
        self.prior_obs_buf = (self.prior_obs_buf - self.obs_mean)/self.obs_var
        print("[post normalization] prior_obs_buf measured heights: %0.2f", self.prior_obs_buf[:, -51:])
        # import pdb; pdb.set_trace()
        self.prior_obs_buf = torch.clip(self.prior_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)


    def compute_observations(self):
        """ Computes observations
        """
        contact =  (torch.norm(self.contact_forces[:, self.feet_indices, :3],dim=-1)>0.001).type(torch.float32)
        contact_ind = torch.tensor([1,0,3,2],device=self.device).long()
        cmd = torch.stack([self.commands[:, 0]-0.5,self.commands[:,2],self.commands[:, 0]-0.5,self.commands[:,2]],dim=-1)
        
        self.obs_buf = torch.cat((  
                            self.roll[...,None],
                            self.pitch[...,None],
                            self.reorder(self.dof_pos[:,:12]),
                            self.reorder(self.dof_vel[:,:12]),
                            self.reorder(self.actions[:,:12]),#policy out in real
                            cmd,
                            #priviledge obs 
                            #contact
                            torch.index_select(contact,1,contact_ind),
                            #mass
                            torch.tensor([1.,0,0],device=self.device)[None,...].repeat(self.num_envs,1)*self.mass_noise,
                            #motor strength
                            torch.ones_like(self.dof_pos[:,:12],device=self.device),
                            #friction
                            self.friction_coeffs.squeeze(-1).to(self.device),
                            self.base_lin_vel[:,:2], 0*self.base_ang_vel[:,-1].unsqueeze(1),
                            self.measured_heights
                            ),dim=-1)
        self.obs_buf = (self.obs_buf - self.obs_mean)/self.obs_var
        
        
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                if 'friction' in self.cfg.domain_rand.curriculum:
                    friction_range = self.cfg.terrain.static_friction + torch.tensor(self.cfg.domain_rand.friction_range)*((self.iter_num/self.cfg.rewards.max_iter)**3 )
                else:
                    friction_range = self.cfg.terrain.static_friction + torch.tensor(self.cfg.domain_rand.friction_range)
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
                
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            if 'mass' in self.cfg.domain_rand.curriculum:
                rng = torch.tensor(self.cfg.domain_rand.added_mass_range)*((self.iter_num/self.cfg.rewards.max_iter)**3 )
            else:
                rng = self.cfg.domain_rand.added_mass_range
            self.mass_noise = np.random.uniform(rng[0], rng[1])
            props[0].mass += self.mass_noise
            self.mass = props[0].mass
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            if self.commands[0,0]==0:
                if len(torch.where(self.commands[:,0]<0.01)[0])>0:
                    self.commands[torch.where(self.commands[:,0]<0.01)[0], 2] = 0

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            
            
        if self.cfg.domain_rand.push_robots:
            if self.cfg.domain_rand.impulse:
                if (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
                    self._push_robot_impulse()
                else:
                    self.forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            else:
                self._push_robots()
            if self.vis:
                self._draw_push_vis()
                if self.common_step_counter%2==0:
                    self.gym.clear_lines(self.viewer)
        else:
            self.apply_force[:,0]=0
            self.forces[...] = 0
            self.gym.clear_lines(self.viewer)
        
        
        # if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.push_interval <=self.push_duration):
        #     self.apply_force[:,0]=1
        #     if self.common_step_counter % self.cfg.domain_rand.push_interval ==0:
        #         self.update_force = True
        #         self.push_duration = (self.cfg.domain_rand.push_duration* torch_rand_float(self.cfg.domain_rand.push_duration_range[0],self.cfg.domain_rand.push_duration_range[1],(1,1),self.device)).item()
                
        #         self.push_interval = (self.cfg.domain_rand.push_interval* torch_rand_float(self.cfg.domain_rand.push_interval_range[0],self.cfg.domain_rand.push_interval_range[1],(1,1),self.device)).item()
        #         self.push_interval = np.ceil(self.push_interval)
        #     else:
        #         self.update_force = False
        #     self._push_robots()
        #     if self.vis:
        #         self._draw_push_vis()
        # else:
        #     self.apply_force[:,0]=0
        #     self.forces[...] = 0
        #     self.gym.clear_lines(self.viewer)
        

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if "lin_vel_x" in self.cfg.commands.discrete:
            cmd_idxs = torch.randint(10, (len(env_ids),), device=self.device)
            cmd_idxs[torch.where(cmd_idxs<self.cfg.commands.ranges.ratio)]=0
            cmd_idxs[torch.where(cmd_idxs>=self.cfg.commands.ranges.ratio)]=1
            self.commands[env_ids,0] = torch.gather(torch.tensor(self.command_ranges["lin_vel_x"],device=self.device),0,cmd_idxs)
        else:    
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            # self.commands[torch.where(self.commands[:,0]<self.cfg.commands.ranges.ratio*self.command_ranges["lin_vel_x"][1])[0], 0] = 0

        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        
        if self.cfg.commands.num_commands ==5:
            #for ee orientation
            self.commands[env_ids,4] =  (torch.randint(0,2,(len(env_ids),1),device=self.device)-0.5)[:,0]*2

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale.to(self.device)
        # print("scaled action", actions_scaled[0])
        # print("biased action", actions_scaled[0] + self.default_dof_pos) 
        # print("dof pos",self.dof_pos)
        # actions_scaled = torch.zeros_like(actions_scaled,device=self.device)
        
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        # print("torques",torch.clip(torques, -self.torque_limits, self.torque_limits))
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
        # return torques

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def _reset_root_states(self, env_ids, eval=False):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            if eval:
                self.root_states[env_ids, 1] += self.env_origins[env_ids,1]
            else:
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            if eval:
                self.root_states[env_ids, 1] += self.env_origins[env_ids,1]
            else:
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _draw_push_vis(self):
        # line_color = gymapi.Vec3(1.0, 0.0, 1.0)

        # enumerate over all robots and draw a line indicating the direction of the push
        for i in range(self.num_envs):
            # robot position
            robot_pos = (self.rb_positions[i,0, :3]).cpu().numpy()
            for j in [-0.002,0.002]:
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(robot_pos[0]+j, robot_pos[1]+j, robot_pos[2])
                arrow_geom = gymutil.WireframeTriangleGeometry(self.forces[i,0,0],self.forces[i,0,1])
                gymutil.draw_lines(arrow_geom, self.gym, self.viewer, self.envs[i], pose)

    def sinforce(self):
        tstep = torch.pi*(self.common_step_counter % self.cfg.domain_rand.push_duration)/self.cfg.domain_rand.push_duration #from 0-pi
        self.forces =self.max_forces* torch.sin(torch.tensor(tstep) )
        
        # print("previous",self.update_force,self.max_forces[:,0,:2])
        # print(tstep,torch.sin(torch.tensor(tstep) ),self.forces[:,0,:2])
    
    def _push_robot_impulse(self):
        new_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        force_signs = (torch.randint(0,2,(self.num_envs,2),device=self.device)-0.5)*2
        new_forces[:,0,:2] =torch.einsum('ij,ij->ij',force_signs,torch_rand_float(self.cfg.domain_rand.min_push_vel_xy, self.cfg.domain_rand.max_push_vel_xy, (self.num_envs,2), device=self.device)) #17 is for shoulder link
        # new_forces[:,0,0] /= 10.
        
        Prob1 = torch.logical_and( 
                (torch_rand_float(0, 1, (self.num_envs,1), device=self.device) < 0.4)[:,0],  
                (torch.norm(self.forces[:,0],dim=-1) < 0.1)
                ) 
        Prob1 = Prob1.to(torch.float32)
        
        self.forces = torch.einsum("ijk,i->ijk",new_forces,Prob1)

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces),gymtorch.unwrap_tensor(self.forces))
        
        # self.forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        # force_signs = (torch.randint(0,2,(self.num_envs,),device=self.device)-0.5)*2
        # self.forces[:,0,:2] =torch.einsum('i,ij->ij',force_signs,torch_rand_float(self.cfg.domain_rand.min_push_vel_xy,  self.cfg.domain_rand.max_push_vel_xy, (self.num_envs,2), device=self.device))
        # self.forces[:,0,0] /= 10.
        
        # # print("self.forces",self.forces[:,0,:2])
        # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces),gymtorch.unwrap_tensor(self.forces))
    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """        
        self.push_duration[torch.norm(self.forces[:,0],dim=-1) > 1] += 1
        # self.push_interval[:] += 1
        
        new_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        force_signs = (torch.randint(0,2,(self.num_envs,2),device=self.device)-0.5)*2
        new_forces[:,0,0] = force_signs[:,0] * torch_rand_float(self.cfg.domain_rand.min_push_vel_xy[0], self.cfg.domain_rand.max_push_vel_xy[0], (self.num_envs,1), device=self.device)[:,0]
        new_forces[:,0,1] = force_signs[:,1]*torch_rand_float(self.cfg.domain_rand.min_push_vel_xy[1], self.cfg.domain_rand.max_push_vel_xy[1], (self.num_envs,1), device=self.device)[:,0]

        new_force_offset = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        new_force_offset[:,0] = torch_rand_float(self.cfg.domain_rand.offset[0][0], self.cfg.domain_rand.offset[0][1], (self.num_envs,1), device=self.device)[:,0]
        new_force_offset[:,1] = torch_rand_float(self.cfg.domain_rand.offset[1][0], self.cfg.domain_rand.offset[1][1], (self.num_envs,1), device=self.device)[:,0]
        new_force_offset[:,2] = torch_rand_float(self.cfg.domain_rand.offset[2][0], self.cfg.domain_rand.offset[2][1], (self.num_envs,1), device=self.device)[:,0]
        
        if getattr(self.cfg.domain_rand,'robot_frame',False):
            new_forces_world = new_forces.clone()
            new_forces_world[:,0,0] = new_forces[:,0,0]*torch.cos(self.yaw)
            new_forces_world[:,0,1] = new_forces[:,0,0]*torch.sin(self.yaw)
        else:        
            new_forces_world = new_forces.clone()
            perm = torch.randperm(self.num_envs)
            new_forces_world[perm[:self.num_envs//2],0,0] /= 10.
            new_forces_world[perm[self.num_envs//2:],0,1] /= 10.
        
        
        Prob1 = torch.logical_and( 
                (torch_rand_float(0, 1, (self.num_envs,1), device=self.device) < 0.3),  
                (torch.norm(self.forces[:,0],dim=-1) < 0.1)
                ) 
        Prob1 = Prob1.to(torch.float32)
        
        self.forces = torch.einsum("ijk,i->ijk",new_forces_world,Prob1[:,0]) + torch.einsum("ijk,i->ijk",self.forces,1- Prob1[:,0])
        
        self.force_offset = torch.einsum("ij,i->ij",new_force_offset,Prob1[:,0]) + torch.einsum("ij,i->ij",self.force_offset,1- Prob1[:,0])
        
        
        Prob2 = torch_rand_float(0, 1, (self.num_envs,1), device=self.device) < 0.1
        Prob2 = torch.logical_or(
            Prob2, torch.logical_and(
                self.push_duration < 50,
                torch.norm(self.forces[:,0],dim=-1) > 1
            )
        )
        Prob2[self.push_duration>=150,0]=0
        self.push_duration[~Prob2[:,0]]=0
        

        Prob2 = Prob2.to(torch.float32)
        self.forces =torch.einsum("ijk,i->ijk",self.forces,Prob2[:,0])         
        self.force_offset = torch.einsum("ij,i->ij",self.force_offset,Prob2[:,0])

        self.rb_positions = self.rigid_body_states[:,:,:3].clone()
        self.rb_positions[:,0,:3] += self.force_offset 

        if getattr(self.cfg.domain_rand,"force_only",False):
            self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(self.forces),gymtorch.unwrap_tensor(self.rb_positions),gymapi.ENV_SPACE)
        else:
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces),gymtorch.unwrap_tensor(self.forces))
            
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:2] = 0.01* noise_level
        noise_vec[5:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:43] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[49:53] = 5* noise_level # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_states).view(self.num_envs,-1,13)
        self.rb_positions = self.rigid_body_states[:,:,:3].clone()
        self.force_offset = torch.zeros(self.num_envs,3,device=self.device)
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_acc = torch.zeros_like(self.dof_vel)
        self.last_root_pos = torch.zeros_like(self.root_states[:,:3])
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.roll,self.pitch,self.yaw = get_euler_xyz(self.base_quat)
        self.roll[torch.where(self.roll>torch.pi)] -= 2*torch.pi
        self.pitch[torch.where(self.pitch>torch.pi)] -= 2*torch.pi
        self.yaw[torch.where(self.yaw>torch.pi)] -= 2*torch.pi
        
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        print("friction",self.cfg.terrain.static_friction)
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # load A1 robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Get the number of robot dofs and rigid bodies
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        print("num robot bodies: ", self.num_bodies)
        print("num robot dofs: ", self.num_dof)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        print("body_names:", body_names)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print("dof names:", self.dof_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # hip_names = [s for s in body_names if self.cfg.asset.hip_name in s]
        print("feet_names:", feet_names)
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # Define start pose for robot
        base_init_state_list = self.cfg.init_state.pos + \
                               self.cfg.init_state.rot + \
                               self.cfg.init_state.lin_vel + \
                               self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        # self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        # start_pose = gymapi.Transform()
        # start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # Potentially randomize start pose of the A1 robot
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # Create the A1 robot
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose,
                                                 self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)

            # Store/change/randomize the DOF properties of the environment
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.base_indice = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], 'base')
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        
        # self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(hip_names)):
        #     self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hip_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.push_interval = self.cfg.domain_rand.push_interval
        
        self.cfg.domain_rand.push_duration = np.ceil(self.cfg.domain_rand.push_duration_s / self.dt)
        self.push_duration = self.cfg.domain_rand.push_duration

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.cfg.terrain.measure_heights: # self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        feet_idx = torch.tensor([self.feet_indices[_idx] for _idx in [1,0,3,2]],device=self.device).long()
        feet_poses = torch.index_select(self.rigid_body_states,1,feet_idx)
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            feet_pos = feet_poses[i,:,:3].cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i,:,:3]).cpu().numpy()
            for k in range(4):
                for j in range(self.num_per_feet):
                    x = height_points[self.num_per_feet*k+j, 0] + feet_pos[k,0]
                    y = height_points[self.num_per_feet*k+j, 1] + feet_pos[k,1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
                    
            for j in range(4*self.num_per_feet,self.height_points.shape[1]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 


    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        self.num_height_points = self.cfg.env.num_vis_obs
        points = torch.zeros(self.num_envs, self.num_height_points, 4, device=self.device, requires_grad=False) #x,y,z
        
        #first 9 point around each feet RF, LF, RR, LR but we have LF, RF, LR, RR so the feet_idx should be [1,0,3,2]
        self.num_per_feet = 9
        scan_radius = 0.1
        angle = torch.arange(self.num_per_feet-1)*(2*torch.pi/(self.num_per_feet-1))
        # needs to be offset by feet position
        feet_x = torch.zeros(self.num_per_feet,device=self.device)
        feet_x[1:] = scan_radius * torch.sin(angle) 
        feet_y = torch.zeros(self.num_per_feet,device=self.device)
        feet_y[1:] = scan_radius * torch.cos(angle)
        points[:,:4*self.num_per_feet,0] = feet_x.repeat(4)
        points[:,:4*self.num_per_feet,1] = feet_y.repeat(4)

        # then 9 points infront of the robot
        grid_x, grid_y = torch.meshgrid(torch.arange(-1,2,1)*0.15, torch.arange(-1,2,1)*0.11)
        # needs to be offset by root state after yaw
        front_x = grid_x.flatten() + 0.4
        front_y = grid_y.flatten()
        points[:,4*self.num_per_feet:4*self.num_per_feet+9,0] = front_x
        points[:,4*self.num_per_feet:4*self.num_per_feet+9,1] = front_y
        
        # finally 6 points under the robot belly
        grid_x, grid_y = torch.meshgrid(torch.arange(-1,2,1)*0.12, torch.arange(-0.5,1,1)*0.08)
        # needs to be offset by root state after yaw
        belly_x = grid_x.flatten() -0.05
        belly_y = grid_y.flatten()
        points[:,-6:,0] = belly_x
        points[:,-6:,1] = belly_y
        
        # y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        # x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        # grid_x, grid_y = torch.meshgrid(x, y)

        # self.num_height_points = grid_x.numel()
        # points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        # points[:, :, 0] = grid_x.flatten()
        # points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # RF, LF, RR, LR but we have LF, RF, LR, RR so the feet_idx should be [1,0,3,2]
        feet_idx = torch.tensor([self.feet_indices[_idx] for _idx in [1,0,3,2]],device=self.device).long()
        feet_pos = torch.index_select(self.rigid_body_states,1,feet_idx)
        feet_points = 4*self.num_per_feet
        other_points = self.height_points.shape[1]-feet_points
        points = self.height_points.clone()
        if env_ids:
            # points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids,:,:3]) 
            points[env_ids,feet_points:,:3] = quat_apply_yaw(self.base_quat[env_ids].repeat(1, other_points), self.height_points[env_ids,feet_points:,:3]) 
            points[env_ids,:feet_points,:3] += feet_pos[env_ids,:,:3].repeat(1,1,self.num_per_feet).view(len(env_ids),feet_points,3)
            points[env_ids,feet_points:,:3] += self.root_states[env_ids,None,:3].repeat(1,other_points,1)
        else:
            # points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points[...,:3]) 
            points[:,feet_points:,:3] = quat_apply_yaw(self.base_quat.repeat(1, other_points), self.height_points[:,feet_points:,:3]) 
            points[:,:feet_points,:3] += feet_pos[...,:3].repeat(1,1,self.num_per_feet).view(self.num_envs,feet_points,3)
            points[:,feet_points:,:3] += self.root_states[:,None,:3].repeat(1,other_points,1)
            
        points[:,:,:2] += self.terrain.cfg.border_size
        points[:,:,:2] = (points[:,:,:2]/self.terrain.cfg.horizontal_scale)
        px = points[:, :, 0].view(-1).long()
        py = points[:, :, 1].view(-1).long()
        px = torch.clip(px, 0, self.height_samples.shape[0]-1)
        py = torch.clip(py, 0, self.height_samples.shape[1]-1)

        
        heights = self.height_samples[px, py]
        
        # heights1 = self.height_samples[px, py]
        # heights2 = self.height_samples[px+1, py]
        # heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        
        heights = heights.view(self.num_envs,-1) * self.terrain.cfg.vertical_scale
        points[:,:feet_points,-1] = points[:,:feet_points,2] - heights[:,:feet_points]
        points[:,feet_points:,-1] = heights[:,feet_points:] - points[:,feet_points:,2]
        
        points[:,feet_points:,-1] += 0.02 # hardcoded to deal with the difference btwn raisim and isaacgym
        # points[:,:feet_points,-1] = heights[:,:feet_points]
        # points[:,feet_points:,-1] = heights[:,feet_points:]
        
        points[:,:,-1] = torch.clip(points[:,:,-1],-1,1)
        
        # return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        return points[:,:,-1]

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return 1 - torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.abs(self.last_actions[:,12:] - self.actions[:,12:]), dim=1)
        # return torch.sum(torch.abs(self.last_actions[:,:12] - self.actions[:,:12]), dim=1) + torch.sum(torch.abs(self.last_actions[:,12:] - self.actions[:,12:]), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        lin_vel_rew = tensor_clamp(self.base_lin_vel[:,0],torch.tensor(-0.1),self.commands[:,0]) + self.cfg.commands.active*(self.commands[:,0]==0)
        # lin_vel_rew = torch.sum(tensor_clamp(self.base_lin_vel[:,0],-0.05,0.35))
        return lin_vel_rew
    
    def _reward_lin_vel_error(self):
        return torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
    
    def _reward_ang_vel_error(self):
        return torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    
    def _reward_high_vel(self):
        dif =tensor_clamp( self.base_lin_vel[:,0] - self.commands[:,0], torch.tensor(0), self.base_lin_vel[:,0])
        return dif
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        weight = torch.ones(self.num_envs,device=self.device)
        weight[self.commands[:, 0] ==0] = 2
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
        # ang_vel_rew = tensor_clamp(self.base_ang_vel[:,2],-0.05,self.commands[:,2]) + self.cfg.commands.active*torch.sum(self.commands[:,2]==0)
        return ang_vel_error*weight

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        # rew = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        rew = torch.sum(torch.square(self.actions), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        # print("stand_still,",rew)
        return rew
    
    def _reward_dog_action_norm(self):
        weight = torch.ones(self.num_envs,device=self.device)
        weight[self.commands[:, 0] ==0] = 2
        dog_mask = torch.tensor(self.cfg.rewards.dog_mask,device=self.device) 
        rew = torch.sum(torch.square(self.actions[:,:12]*dog_mask), dim=1)
        return rew*weight


    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_dog_work(self):
        weight = torch.ones(self.num_envs,device=self.device)
        weight[self.commands[:, 0] ==0] = 2
        # return torch.abs(torch.sum(self.torques[:,:12] * self.dof_vel[:,:12], dim = 1))*weight
        return torch.sum(torch.abs(self.torques[:,:12] * self.dof_vel[:,:12]), dim = 1)*weight

        
    def _reward_slip(self):
        # Penalize if feet is in contact with ground and has nonzero velocity      
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        feet_xy_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:9]).sum(dim=-1)
        dragging_vel = contact * feet_xy_vel
        return dragging_vel.sum(dim=-1)  
    
    def _reward_hip_norm(self):
        #0 3 6 9 L R L R
        rew = torch.sum(torch.square(self.actions[:,0:3:9]), dim=1)
        return rew
    
    def _reward_inward(self):
        #0 3 6 9 L R L R
        lhip = self.actions[:,[0,6]]
        lhip *= (lhip<0) #penalize negative left hip 
        rhip = self.actions[:,[3,9]]
        rhip *= (rhip>0) #penalize positive right hip
        rew = torch.sum(torch.square(lhip), dim=1) + torch.sum(torch.square(rhip), dim=1)
        return rew
    
    def _reward_episode_length(self):
        return self.episode_length_buf

    def _reward_x_distance(self):
        return self.root_states[:,0]-self.last_root_pos[:,0]
    
    def _reward_y_distance(self):
        return self.root_states[:,1]-self.last_root_pos[:,1]
    
    def _reward_base_vel(self):
        #penalize base vel at the y direction
        return torch.abs(self.base_lin_vel[:,1])
    
    def _reward_roll_pitch(self):
        sumrp = torch.abs(self.roll) + torch.abs(self.pitch)
        return torch.exp(-sumrp/0.5)

    
    def _reward_ee_rp(self):
    
        ee_quat = self.rigid_body_states[:,self.ee_indice,3:7]#x,y,z,w
        roll,pitch,yaw = get_euler_xyz(ee_quat)
        
        roll[torch.where(roll>torch.pi)] -= 2*torch.pi
        pitch[torch.where(pitch>torch.pi)] -= 2*torch.pi

        # dist = torch.abs(roll%torch.pi) + torch.abs(pitch-(torch.pi+self.commands[:,4]*torch.pi/2))
        #limit to point upward
        # dist = torch.abs(roll) + torch.abs(pitch-(-torch.pi+torch.pi/2))
        if self.cfg.commands.num_commands ==5:
            dist = torch.abs(roll) + torch.abs(pitch-(self.commands[:,4]*torch.pi/2))
        else:
            dist = torch.abs(roll) + torch.abs(pitch-(torch.pi/2))
        # print("roll",roll,pitch,dist,torch.exp(-dist))
        
        return torch.exp(-dist)
    
    
    def _reward_folded_leg(self):
        # "FL (1,2), RR (10, 11), thigh 0.8, calf -2.1"
        FL_dog = self.dof_pos[:,1:3] #num_envc,2
        RR_dog = self.dof_pos[:,10:12]
        target = torch.zeros_like(FL_dog,device=self.device)
        target[:,0] = 0.8
        target[:,1] = -2.25
        FL_dif = torch.sum(torch.abs(FL_dog-target),dim=-1)
        RR_dif = torch.sum(torch.abs(RR_dog-target),dim=-1)
        return FL_dif + RR_dif
    
    def _reward_folded_contact(self):
        # penalize high contact forces
        feet_indices = [self.feet_indices[0],self.feet_indices[3]]
        return torch.sum(torch.norm(self.contact_forces[:, feet_indices, :], dim=-1), dim=1)
