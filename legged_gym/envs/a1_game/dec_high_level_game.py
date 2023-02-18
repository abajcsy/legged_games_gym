
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict, get_load_path
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils import task_registry
import sys

from rsl_rl.runners.low_level_policy_runner import LLPolicyRunner

class DecHighLevelGame():
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
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
        print("[DecHighLevelGame] initializing ...")
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # setup the capture distance between two agents
        self.capture_dist = self.cfg.env.capture_dist
        self.MAX_REL_POS = 100.

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        # setup low-level policy loader
        # ll_env_cfg, ll_train_cfg = task_registry.get_cfgs(name="low_level_game")
        ll_env_cfg, ll_train_cfg = task_registry.get_cfgs(name="a1")

        # need to make sure that the low-level and high level have the same representation
        ll_env_cfg.env.num_envs = self.cfg.env.num_envs
        ll_env_cfg.terrain.num_rows = self.cfg.terrain.num_rows
        ll_env_cfg.terrain.num_cols = self.cfg.terrain.num_cols
        ll_env_cfg.terrain.curriculum = self.cfg.terrain.curriculum
        ll_env_cfg.noise.add_noise = self.cfg.noise.add_noise
        ll_env_cfg.domain_rand.randomize_friction = self.cfg.domain_rand.randomize_friction
        ll_env_cfg.domain_rand.push_robots = self.cfg.domain_rand.push_robots
        ll_env_cfg.terrain.mesh_type = self.cfg.terrain.mesh_type

        # TODO: HACK! Bump up the torque penalty to simulate "instantaneous control effort cost"
        # before: -0.00001;
        ll_env_cfg.rewards.scales.torques = -5.

        # TODO: HACK! Align init states between the two configs
        ll_env_cfg.init_state.pos = self.cfg.init_state.robot_pos
        ll_env_cfg.init_state.rot = self.cfg.init_state.robot_rot
        ll_env_cfg.init_state.lin_vel = self.cfg.init_state.robot_lin_vel
        ll_env_cfg.init_state.ang_vel = self.cfg.init_state.robot_ang_vel
        ll_env_cfg.init_state.agent_pos = self.cfg.init_state.agent_pos
        ll_env_cfg.init_state.agent_rot = self.cfg.init_state.agent_rot
        ll_env_cfg.init_state.agent_lin_vel = self.cfg.init_state.agent_lin_vel
        ll_env_cfg.init_state.agent_ang_vel = self.cfg.init_state.agent_ang_vel
        ll_env_cfg.commands.heading_command = self.cfg.commands.heading_command

        # create the policy loader
        ll_train_cfg_dict = class_to_dict(ll_train_cfg)
        ll_policy_runner = LLPolicyRunner(ll_env_cfg, ll_train_cfg_dict, self.device)

        # make the low-level environment
        print("[DecHighLevelGame] preparing low level environment...")
        self.ll_env, _ = task_registry.make_env(name="low_level_game", args=None, env_cfg=ll_env_cfg)

        # load low-level policy
        print("[DecHighLevelGame] loading low level policy... for: ", ll_train_cfg.runner.experiment_name)
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', ll_train_cfg.runner.experiment_name)
        # load_run = 'vanilla_policy'
        load_run = 'sideways_walking_policy'
        # load_run = ll_train_cfg.runner.load_run
        path = get_load_path(log_root, load_run=load_run,
                             checkpoint=ll_train_cfg.runner.checkpoint)
        self.ll_policy = ll_policy_runner.load_policy(path)

        # parse the high-level config (note: it depends on the low-level config, so has to be done after)
        self._parse_cfg(self.cfg)

        self.gym = gymapi.acquire_gym()

        self.num_envs = self.cfg.env.num_envs

        # robot policy info
        self.num_obs_robot = self.cfg.env.num_observations_robot
        self.num_obs_encoded_robot = self.cfg.env.num_obs_encoded_robot
        self.embedding_sz_robot = self.cfg.env.embedding_sz_robot
        self.num_privileged_obs_robot = self.cfg.env.num_privileged_obs_robot
        self.num_actions_robot = self.cfg.env.num_actions_robot

        # agent policy info
        self.num_obs_agent = self.cfg.env.num_observations_agent
        self.num_obs_encoded_agent = self.cfg.env.num_obs_encoded_agent
        self.embedding_sz_agent = self.cfg.env.embedding_sz_agent
        self.num_privileged_obs_agent = self.cfg.env.num_privileged_obs_agent
        self.num_actions_agent = self.cfg.env.num_actions_agent


        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # PARTIAL obs space: setup the indicies of relevant observation quantities
        # self.pos_idxs = list(range(0,12)) # all relative position history
        # self.ang_global_idxs = list(range(12,20)) # all relative (global) angle
        # self.ang_local_idxs = list(range(20,28))  # all relative (local) angle
        # self.detect_idxs = list(range(28, 32))  # all detected bools
        # self.visible_idxs = list(range(32, 36))  # all visible bools


        # FULL obs space: but with extra info
        self.pos_idxs = list(range(0,3)) # all relative position history
        self.ang_global_idxs = list(range(3,5)) # all relative (global) angle
        self.ang_global_idxs = list(range(5,7))  # all relative (global) angle
        self.detect_idxs = list(range(7,8))  # all detected bools
        self.visible_idxs = list(range(8,9))  # all visible bools

        # allocate robot buffers
        # self.obs_buf_robot = self.MAX_REL_POS * torch.ones(self.num_envs, self.num_obs_robot, device=self.device, dtype=torch.float)
        # self.obs_buf_robot[:, -2:] = 0. # reset the robot's sense booleans to zero
        self.obs_buf_robot = torch.zeros(self.num_envs, self.num_obs_robot, device=self.device, dtype=torch.float)
        self.obs_buf_robot[:, self.pos_idxs] = self.MAX_REL_POS
        self.rew_buf_robot = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)


        # allocate agentator buffers
        self.obs_buf_agent = -self.MAX_REL_POS * torch.ones(self.num_envs, self.num_obs_agent, device=self.device, dtype=torch.float)
        self.obs_buf_agent[:, -2:] = 0.
        # self.obs_buf_agent = torch.zeros(self.num_envs, self.num_obs_agent, device=self.device, dtype=torch.float)
        # self.obs_buf_agent[:, self.pos_idxs] = -self.MAX_REL_POS
        self.rew_buf_agent = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.capture_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.curr_episode_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.detected_buf_agent = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool) # stores in which environment the agent has sensed the robot
        self.detected_buf_robot = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)  # stores in which environment the robot has sensed the agent

        # used for transforms.
        self.x_unit_tensor = to_torch([1., 0., 0.], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0., 0., 1.], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        if self.num_privileged_obs_agent is not None:
            self.privileged_obs_buf_agent = torch.zeros(self.num_envs, self.num_privileged_obs_agent, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf_agent = None
            # self.num_privileged_obs = self.num_obs

        if self.num_privileged_obs_robot is not None:
            self.privileged_obs_buf_robot = torch.zeros(self.num_envs, self.num_privileged_obs_robot, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf_robot = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # TODO: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        print("[DecHighLevelGame] initializing buffers...")
        self._init_buffers()
        print("[DecHighLevelGame] preparing AGENT reward functions...")
        self._prepare_reward_function_agent()
        print("[DecHighLevelGame] preparing ROBOT reward functions...")
        self._prepare_reward_function_robot()
        print("[DecHighLevelGame] done with initializing!")
        self.init_done = True

        print("[DecHighLevelGame: init] agent pos: ", self.ll_env.root_states[self.ll_env.agent_indices, :3])
        print("[DecHighLevelGame: init] robot pos: ", self.ll_env.root_states[self.ll_env.robot_indices, :3])
        print("[DecHighLevelGame: init] dist: ", torch.norm(self.ll_env.root_states[self.ll_env.agent_indices, :3] - self.ll_env.root_states[self.ll_env.robot_indices, :3], dim=-1))

    def step(self, command_agent, command_robot):
        """Applies a high-level command to a particular agent, simulates, and calls self.post_physics_step()

        Args:
            command_agent (torch.Tensor): Tensor of shape (num_envs, num_actions_agent) for the high-level command
            command_robot (torch.Tensor): Tensor of shape (num_envs, num_actions_robot) for the high-level command

        Returns:
            obs_buf_agent, obs_buf_robot, priviledged_obs_buf_agent, priviledged_obs_buf_robot, ...
            rew_buf_agent, rew_buf_robot, reset_buf, extras
        """
        # TODO: HACK!! This is for debugging.
        command_agent *= 0
        # rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        # _, _, robot_yaw = get_euler_xyz(self.ll_env.base_quat)
        # robot_yaw = wrap_to_pi(robot_yaw)
        #
        # rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        # rel_yaw_global = wrap_to_pi(rel_yaw_global)
        #
        # for env_idx in range(self.num_envs):
        #     if torch.abs(rel_yaw_global[env_idx]) < 0.15:
        #         command_robot[env_idx, 0] = 1
        #         command_robot[env_idx, 1] = 0
        #         command_robot[env_idx, 2] = 0
        #     else:
        #         command_robot[env_idx, 0] = 0
        #         command_robot[env_idx, 1] = 0
        #         command_robot[env_idx, 2] = 1
        # TODO: HACK!! This is for debugging.

        # clip the robot and agent's commands
        command_robot = self.clip_command_robot(command_robot)
        command_agent = self.clip_command_agent(command_agent)

        # print("command_robot: ", command_robot)

        # update the low-level simulator command since it deals with the robot
        command_robot = torch.cat((command_robot, # lin_vel_x, lin_vel_y, ang_vel_yaw
                                  torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)), # heading
                                  dim=-1)
        # NOTE: low-level policy requires 4D control
        # command_robot = torch.cat((command_robot[:, 0].unsqueeze(-1), # lin_vel_x
        #                           torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False), # lin_vel_y
        #                           command_robot[:, 1].unsqueeze(-1), # ang_vel_yaw
        #                           torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)), # heading
        #                           dim=-1)
        self.ll_env.commands = command_robot

        # get the low-level actions as a function of low-level obs
        ll_robot_obs = self.ll_env.get_observations()
        ll_robot_actions = self.ll_policy(ll_robot_obs.detach())

        # forward simulate the low-level actions
        ll_obs, _, ll_rews, ll_dones, ll_infos = self.ll_env.step(ll_robot_actions.detach())

        # forward simulate the agent action too
        # self.step_agent_single_integrator(command=command_agent)
        self.step_agent_omnidirectional(command=command_agent)

        # take care of terminations, compute observations, rewards, and dones
        self.post_physics_step(ll_rews, ll_dones)

        return self.obs_buf_agent, self.obs_buf_robot, self.privileged_obs_buf_agent, self.privileged_obs_buf_robot, self.rew_buf_agent, self.rew_buf_robot, self.reset_buf, self.extras

    def clip_command_robot(self, command_robot):
        """Clips the robot's commands"""
        # clip the robot's commands
        command_robot[:, 0] = torch.clip(command_robot[:, 0], min=self.command_ranges["lin_vel_x"][0], max=self.command_ranges["lin_vel_x"][1])
        command_robot[:, 1] = torch.clip(command_robot[:, 1], min=self.command_ranges["ang_vel_yaw"][0], max=self.command_ranges["ang_vel_yaw"][1])

        # set small commands to zero; this was originally used by lowlevel code to enable robot to learn to stand still
        command_robot[:, :1] *= (torch.norm(command_robot[:, :1], dim=1) > 0.2).unsqueeze(1)

        return command_robot

    def clip_command_agent(self, command_agent):
        """Clips the agent's commands"""
        command_agent[:, 0] = torch.clip(command_agent[:, 0], min=self.command_ranges["agent_lin_vel_x"][0], max=self.command_ranges["agent_lin_vel_x"][1])
        command_agent[:, 1] = torch.clip(command_agent[:, 1], min=self.command_ranges["agent_lin_vel_y"][0], max=self.command_ranges["agent_lin_vel_y"][1])
        command_agent[:, 2] = torch.clip(command_agent[:, 2], min=self.command_ranges["agent_ang_vel_yaw"][0], max=self.command_ranges["agent_ang_vel_yaw"][1])
        return command_agent

    def step_agent_omnidirectional(self, command):
        """
        Steps agent modeled as an omnidirectional agent
            x' = x + dt * u1 * cos(theta)
            y' = y + dt * u2 * sin(theta)
            z' = z
            theta' = theta + dt * u3
        where the control is 3-dimensional:
            u = [u1 u2 u3] = [vx, vy, omega]
        """
        lin_vel_x = command[:, 0]
        lin_vel_y = command[:, 1]
        ang_vel = command[:, 2]

        # TODO: agent gets simulated at the same Hz as the low-level controller!
        for _ in range(self.ll_env.cfg.control.decimation):
            self.agent_pos[:, 0] += self.ll_env.cfg.sim.dt * lin_vel_x #* torch.cos(self.agent_heading)
            self.agent_pos[:, 1] += self.ll_env.cfg.sim.dt * lin_vel_y #* torch.sin(self.agent_heading)
            self.agent_heading += self.ll_env.cfg.sim.dt * ang_vel
            self.agent_heading = wrap_to_pi(self.agent_heading) # make sure heading is between -pi and pi

        # convert from heading angle to quaternion
        heading_quat = quat_from_angle_axis(self.agent_heading, self.z_unit_tensor)

        # update the simulator state for the agent!
        self.ll_env.root_states[self.ll_env.agent_indices, :3] = self.agent_pos
        self.ll_env.root_states[self.ll_env.agent_indices, 3:7] = heading_quat

        self.ll_env.root_states[self.ll_env.agent_indices, 7] = lin_vel_x  # lin vel x
        self.ll_env.root_states[self.ll_env.agent_indices, 8] = lin_vel_y  # lin vel y
        self.ll_env.root_states[self.ll_env.agent_indices, 9] = 0.  # lin vel z
        self.ll_env.root_states[self.ll_env.agent_indices, 10] = 0. # ang vel x
        self.ll_env.root_states[self.ll_env.agent_indices, 11] = 0. # ang_vel y
        self.ll_env.root_states[self.ll_env.agent_indices, 12] = ang_vel # ang_vel z

        # self.ll_env.gym.set_actor_root_state_tensor(self.ll_env.sim, gymtorch.unwrap_tensor(self.ll_env.root_states))
        agent_env_ids_int32 = self.ll_env.agent_indices.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.ll_env.sim,
                                                     gymtorch.unwrap_tensor(self.ll_env.root_states),
                                                     gymtorch.unwrap_tensor(agent_env_ids_int32),
                                                     len(agent_env_ids_int32))

    def step_agent_single_integrator(self, command):
        """
        Steps agent modeled as a single integrator:
            x' = x + dt * u1/4
            y' = y + dt * u2
            z' = z          (assume z-dim is constant)
        """

        # agent that has full observability of the robot state
        command_lin_vel_x = command[:, 0]
        command_lin_vel_y = command[:, 1]

        # TODO: agent gets simulated at the same Hz as the low-level controller!
        for _ in range(self.ll_env.cfg.control.decimation):
            self.agent_pos[:, 0] += self.ll_env.cfg.sim.dt * command_lin_vel_x
            self.agent_pos[:, 1] += self.ll_env.cfg.sim.dt * command_lin_vel_y

        heading_quat = quat_from_angle_axis(self.agent_heading, self.z_unit_tensor)
        # print("[DecHighLevelGame | step_agent_single_integrator] heading_quat (z-unit): ", heading_quat)
        # print("[DecHighLevelGame] curr agent quaternion: ", self.ll_env.root_states[self.ll_env.agent_indices, 3:7])

        # update the simulator state for the agent!
        self.ll_env.root_states[self.ll_env.agent_indices, :3] = self.agent_pos
        self.ll_env.root_states[self.ll_env.agent_indices, 3:7] = heading_quat

        self.ll_env.root_states[self.ll_env.agent_indices, 7] = command_lin_vel_x # lin vel x
        self.ll_env.root_states[self.ll_env.agent_indices, 8] = command_lin_vel_y # lin vel y
        self.ll_env.root_states[self.ll_env.agent_indices, 9] = 0.  # lin vel z
        self.ll_env.root_states[self.ll_env.agent_indices, 10:13] = 0. # ang vel xyz

        # self.ll_env.gym.set_actor_root_state_tensor(self.ll_env.sim, gymtorch.unwrap_tensor(self.ll_env.root_states))
        agent_env_ids_int32 = self.ll_env.agent_indices.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.ll_env.sim,
                                                     gymtorch.unwrap_tensor(self.ll_env.root_states),
                                                     gymtorch.unwrap_tensor(agent_env_ids_int32),
                                                     len(agent_env_ids_int32))

    def post_physics_step(self, ll_rews, ll_dones):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
        """
        self.gym.refresh_actor_root_state_tensor(self.ll_env.sim)

        # updates the local copies of agent state info
        self._update_agent_states()

        self.episode_length_buf += 1
        self.curr_episode_step += 1 # this is used to model the progress of "time" in the episode

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward_robot(ll_rews) # robot (robot) combines the high-level and low-level rewards
        self.compute_reward_agent()
        # print("agent_dist: ", torch.norm(self.robot_states[:, :3] - self.agent_pos, dim=-1))
        # print("rew_buf_agent: ", self.rew_buf_agent)

        # NOTE: ll_env.step() already reset based on the low-level reset criteria
        self.reset_buf |= self.ll_env.reset_buf # combine the low-level dones and the high-level dones
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.reset_idx(env_ids)

        # compute the high-level observation for each agent
        self.compute_observations_agent()
        self.compute_observations_robot()

        # print("[in post_physics_step] obs_buf_agent: ", self.obs_buf_agent)

    def check_termination(self):
        """ Check if environments need to be reset under various conditions.
        """
        # reset agent-robot if they are within the capture distance
        self.capture_buf = torch.norm(self.robot_states[:, :2] - self.agent_pos[:, :2], dim=-1) < self.capture_dist
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf = self.capture_buf.clone()
        self.reset_buf |= self.time_out_buf


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # resets the agent and robot states in the low-level simulator environment
        self.ll_env._reset_dofs(env_ids)
        self.ll_env._reset_root_states(env_ids)

        # update the state variables
        self._update_agent_states()
        self.agent_heading[env_ids] = self.ll_env.init_agent_heading[env_ids].clone()

        # reset buffers
        # reset_obs = torch.zeros(len(env_ids), self.num_obs_agent, device=self.device, requires_grad=False)
        # reset_obs[:, self.pos_idxs] = -self.MAX_REL_POS # only the initial relative position is reset different
        # self.obs_buf_agent[env_ids, :] = reset_obs

        self.obs_buf_agent[env_ids, :] = -self.MAX_REL_POS
        self.obs_buf_agent[env_ids, -2:] = 0.
        # print("reset obs buf agent: ", self.obs_buf_agent[env_ids, :])

        # reset buffers
        reset_robot_obs = torch.zeros(len(env_ids), self.num_obs_robot, device=self.device, requires_grad=False)
        reset_robot_obs[:, self.pos_idxs] = self.MAX_REL_POS # only the initial relative position is reset different
        self.obs_buf_robot[env_ids, :] = reset_robot_obs

        # self.obs_buf_robot[env_ids, :] = self.MAX_REL_POS      # reset the robot's rel_agent_pos to max relative position
        # self.obs_buf_robot[env_ids, -2:] = 0                    # reset the robot's sense booleans to zero
        
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.curr_episode_step[env_ids] = 0
        self.capture_buf[env_ids] = 0
        self.detected_buf_agent[env_ids, :] = 0
        self.detected_buf_robot[env_ids, :] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums_agent.keys():
            self.extras["episode"]['rew_agent_' + key] = torch.mean(self.episode_sums_agent[key][env_ids]) / self.max_episode_length_s
            self.episode_sums_agent[key][env_ids] = 0.
        for key in self.episode_sums_robot.keys():
            self.extras["episode"]['rew_robot_' + key] = torch.mean(self.episode_sums_robot[key][env_ids]) / self.max_episode_length_s
            self.episode_sums_robot[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        actions_agent = torch.zeros(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)
        actions_robot = torch.zeros(self.num_envs, self.num_actions_robot, device=self.device, requires_grad=False)
        obs_agent, obs_robot, privileged_obs_agent, privileged_obs_robot, _, _, _, _ = self.step(actions_agent, actions_robot)
        return obs_agent, obs_robot, privileged_obs_agent, privileged_obs_robot

    def compute_reward_robot(self, ll_rews):
        """ Compute rewards for the robot
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward

            Args: ll_rews (torch.Tensor) of size num_envs containing low-level reward per environment
        """
        self.rew_buf_robot[:] = 0.
        for i in range(len(self.reward_functions_robot)):
            name = self.reward_names_robot[i]
            rew = self.reward_functions_robot[i]() * self.reward_scales_robot[name]
            self.rew_buf_robot += rew
            self.episode_sums_robot[name] += rew
            # print("[", name, "] rew :", rew)
        # sum together the low-level reward and the high-level reward
        # ll_rew_weight = 2.0
        # self.rew_buf_robot += ll_rew_weight * ll_rews
        if self.cfg.rewards_robot.only_positive_rewards:
            self.rew_buf_robot[:] = torch.clip(self.rew_buf_robot[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales_robot:
            # rew = self._reward_termination() * self.reward_scales_robot["termination"]
            rew_capture = self.capture_buf * self.reward_scales_robot["termination"] # TODO: THIS IS A HACK!!!
            rew_other = torch.logical_xor(self.reset_buf, self.capture_buf) * -1 * self.reward_scales_robot["termination"]
            rew = rew_capture + rew_other
            # rew = self.capture_buf * self.reward_scales_robot["termination"] # TODO: THIS IS A HACK!!!
            self.rew_buf_robot += rew
            self.episode_sums_robot["termination"] += rew

        # print("episode_sum_robot[pursuit]: ", self.episode_sums_robot["pursuit"])

    def compute_reward_agent(self):
        """ Compute rewards for the AGENT
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf_agent[:] = 0.
        for i in range(len(self.reward_functions_agent)):
            name = self.reward_names_agent[i]
            rew = self.reward_functions_agent[i]() * self.reward_scales_agent[name]
            self.rew_buf_agent += rew
            self.episode_sums_agent[name] += rew
        if self.cfg.rewards_agent.only_positive_rewards:
            self.rew_buf_agent[:] = torch.clip(self.rew_buf_agent[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales_agent:
            # rew = self._reward_termination() * self.reward_scales_agent["termination"]
            rew = self.capture_buf * self.reward_scales_agent["termination"]
            self.rew_buf_agent += rew
            self.episode_sums_agent["termination"] += rew
        # print("high level rewards + low-level rewards: ", self.rew_buf)

    def compute_observations_robot(self):
        """ Computes observations of the robot robot
        """
        # self.compute_observations_pos_robot()
        # self.compute_observations_pos_angle_robot()
        self.compute_observations_full_obs()
        # self.compute_observations_limited_FOV_robot()

        # print("[DecHighLevelGame] self.obs_buf_robot: ", self.obs_buf_robot)

    def compute_observations_pos_robot(self):
        """ Computes observations of the agent with full FOV
        """
        # from agent's POV, get the relative position to the robot
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        self.obs_buf_robot = rel_agent_pos_xyz

    def compute_observations_pos_angle_robot(self):
        """ Computes observations of the agent with full FOV
        """
        # from agent's POV, get the relative position to the robot
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        _, _, robot_yaw = get_euler_xyz(self.ll_env.base_quat)
        robot_yaw = wrap_to_pi(robot_yaw)

        rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        rel_yaw_global = rel_yaw_global.unsqueeze(-1)
        rel_yaw_global = wrap_to_pi(rel_yaw_global)

        # print("num positive angs: ", len((rel_yaw_global > 0).nonzero()))
        # print("num neg angs: ", len((rel_yaw_global <= 0).nonzero()))

        self.obs_buf_robot = torch.cat((rel_agent_pos_xyz * 0.1,
                                      torch.cos(rel_yaw_global),
                                      torch.sin(rel_yaw_global)
                                      ), dim=-1)

    def compute_observations_full_obs(self):
        """ Computes observations of the robot with partial observability.
                    obs_buf is laid out as:

                   [s1, sin(h1_g), cos(h1_g), sin(h1_l), cos(h1_l), d1, v1]

                   where s1 = (px1, py1, pz1) is the relative position (of size 3)
                         h1_g = is the *global* relative yaw angle between robot yaw and COM of agent (of size 1)
                         h1_l = is the *local* relative yaw angle between robot yaw and agent yaw (of size 1)
                         d1 = is "detected" boolean which tracks if the agent was detected once (of size 1)
                         v1 = is "visible" boolean which tracks if the agent was visible in FOV (of size 1)
                """
        sense_rel_pos, sense_rel_yaw_global, sense_rel_yaw_local, detected_bool, visible_bool = self.robot_sense_agent(partial_obs=False)

        self.obs_buf_robot = torch.cat((sense_rel_pos * 0.1,
                                        sense_rel_yaw_global,
                                        # sense_rel_yaw_local,
                                        detected_bool.long(),
                                        visible_bool.long(),
                                        ), dim=-1)

    def compute_observations_limited_FOV_robot(self):
        """ Computes observations of the robot with partial observability.
            obs_buf is laid out as:

           [s1, s2, s3, s4,
            sin(h1_g), cos(h1_g), sin(h2_g), cos(h2_g), sin(h3_g), cos(h3_g), sin(h4_g), cos(h4_g),
            sin(h1_l), cos(h1_l), sin(h2_l), cos(h2_l), sin(h3_l), cos(h3_l), sin(h4_l), cos(h4_l),
            d1, d2, d3, d4,
            v1, v2, v3, v4]

           where s1 = (px1, py1, pz1) is the relative position (of size 3)
                 h1_g = is the *global* relative yaw angle between robot yaw and COM of agent (of size 1)
                 h1_l = is the *local* relative yaw angle between robot yaw and agent yaw (of size 1)
                 d1 = is "detected" boolean which tracks if the agent was detected once (of size 1)
                 v1 = is "visible" boolean which tracks if the agent was visible in FOV (of size 1)
        """
        # from agent's POV, get its sensing
        sense_rel_pos, sense_rel_yaw_global, sense_rel_yaw_local, detected_bool, visible_bool = self.robot_sense_agent(partial_obs=True)

        old_sense_rel_pos = self.obs_buf_robot[:, self.pos_idxs[3:]].clone()  # remove the oldest relative position observation
        old_sense_rel_yaw_global = self.obs_buf_robot[:, self.ang_global_idxs[2:]].clone()  # removes the oldest global relative heading observation
        old_sense_rel_yaw_local = self.obs_buf_robot[:, self.ang_local_idxs[2:]].clone()  # removes the oldest local relative heading observation
        old_detect_bool = self.obs_buf_robot[:, self.detect_idxs[1:]].clone()  # remove the corresponding oldest detected bool
        old_visible_bool = self.obs_buf_robot[:, self.visible_idxs[1:]].clone()  # remove the corresponding oldest visible bool

        # if robot is visible, then we need to scale; otherwise its an old measurement so no need
        obs_pos_scale = 0.1
        obs_yaw_local_scale = 1.0
        obs_yaw_global_scale = 1.0

        # TODO: There is something messed up with the scaling here! Tends towards zeros
        visible_env_ids = (visible_bool == 1).nonzero(as_tuple=False).flatten()
        sense_rel_pos[visible_env_ids, :] *= obs_pos_scale
        sense_rel_yaw_local[visible_env_ids, :] *= obs_yaw_local_scale
        sense_rel_yaw_global[visible_env_ids, :] *= obs_yaw_global_scale

        # a new observation has the form: (rel_opponent_pos, rel_opponent_heading, detected_bool, visible_bool)
        self.obs_buf_robot = torch.cat((old_sense_rel_pos,
                                        sense_rel_pos,
                                        old_sense_rel_yaw_global,
                                        sense_rel_yaw_global,
                                        old_sense_rel_yaw_local,
                                        sense_rel_yaw_local,
                                        old_detect_bool,
                                        detected_bool.long(),
                                        old_visible_bool,
                                        visible_bool.long(),
                                        ), dim=-1)

        # print("[DecHighLevelGame] in compute_observations_robot obs_buf_robot: ", self.obs_buf_robot)

    def compute_observations_agent(self):
        """ Computes observations of the agent
        """
        # self.compute_observations_pos_agent()
        self.compute_observations_pos_angle_agent()
        # self.compute_observations_limited_FOV_agent()

    def compute_observations_pos_agent(self):
        """ Computes observations of the agent with only position
        """
        rel_robot_pos_xyz = self.robot_states[:, :3] - self.agent_pos[:, :3]
        self.obs_buf_agent = rel_robot_pos_xyz

    def compute_observations_pos_angle_agent(self):
        """ Computes observations of the agent with full FOV
        """
        # from agent's POV, get the relative position to the robot
        rel_robot_pos_xyz = self.robot_states[:, :3] - self.agent_pos[:, :3]

        angle_btwn_agents_global = torch.atan2(rel_robot_pos_xyz[:, 1], rel_robot_pos_xyz[:, 0]) - self.agent_heading
        angle_btwn_agents_global = angle_btwn_agents_global.unsqueeze(-1)
        angle_btwn_agents_global = wrap_to_pi(angle_btwn_agents_global)

        cos_angle_global = torch.cos(angle_btwn_agents_global)
        sin_angle_global = torch.sin(angle_btwn_agents_global)

        self.obs_buf_agent = torch.cat((rel_robot_pos_xyz * 0.1,
                                      cos_angle_global,
                                      sin_angle_global
                                      ), dim=-1)

    def compute_observations_limited_FOV_agent(self):
        """ Computes observations of the agent with partial observability.
            obs_buf is laid out as:

           [s1, s2, s3, s4,
            sin(h1_g), cos(h1_g), sin(h2_g), cos(h2_g), sin(h3_g), cos(h3_g), sin(h4_g), cos(h4_g),
            sin(h1_l), cos(h1_l), sin(h2_l), cos(h2_l), sin(h3_l), cos(h3_l), sin(h4_l), cos(h4_l),
            d1, d2, d3, d4,
            v1, v2, v3, v4]

           where s1 = (px1, py1, pz1) is the relative position (of size 3)
                 h1_g = is the *global* relative yaw angle between agent heading and COM of robot (of size 1)
                 h1_l = is the *local* relative yaw angle between agent heading and robot heading (of size 1)
                 d1 = is "detected" boolean which tracks if the robot was detected once (of size 1)
                 v1 = is "visible" boolean which tracks if the robot was visiblein FOV (of size 1)

        """

        # from agent's POV, get its sensing
        sense_rel_pos, sense_rel_yaw_global, sense_rel_yaw_local, detected_bool, visible_bool = self.agent_sense_robot()

        old_sense_rel_pos = self.obs_buf_agent[:, self.pos_idxs[3:]].clone()         # remove the oldest relative position observation
        old_sense_rel_yaw_global = self.obs_buf_agent[:, self.ang_global_idxs[2:]].clone()    # removes the oldest global relative heading observation
        old_sense_rel_yaw_local = self.obs_buf_agent[:, self.ang_local_idxs[2:]].clone()  # removes the oldest local relative heading observation
        old_detect_bool = self.obs_buf_agent[:, self.detect_idxs[1:]].clone()               # remove the corresponding oldest detected bool
        old_visible_bool = self.obs_buf_agent[:, self.visible_idxs[1:]].clone()                # remove the corresponding oldest visible bool

        # if robot is visible, then we need to scale; otherwise its an old measurement so no need
        obs_pos_scale = 1.0
        obs_yaw_local_scale = 1.0
        obs_yaw_global_scale = 1.0

        # TODO: There is something messed up with the scaling here! Tends towards zeros
        visible_env_ids = (visible_bool == 1).nonzero(as_tuple=False).flatten()
        sense_rel_pos[visible_env_ids, :] *= obs_pos_scale
        sense_rel_yaw_local[visible_env_ids, :] *= obs_yaw_local_scale
        sense_rel_yaw_global[visible_env_ids, :] *= obs_yaw_global_scale

        # print("OLD self.obs_buf_agent: ", self.obs_buf_agent)
        # print("old_sense_rel_pos: ", old_sense_rel_pos)
        # print("sense_rel_pos: ", sense_rel_pos)
        # print("old_sense_rel_yaw_global: ", old_sense_rel_yaw_global)
        # print("sense_rel_yaw_global: ", sense_rel_yaw_global)
        # print("old_sense_rel_yaw_local: ", old_sense_rel_yaw_local)
        # print("sense_rel_yaw_local: ", sense_rel_yaw_local)
        # print("old_detect_bool: ", old_detect_bool)
        # print("detected_bool: ", detected_bool)
        # print("old_visible_bool: ", old_visible_bool)
        # print("visible_bool: ", visible_bool)

        # a new observation has the form: (rel_opponent_pos, rel_opponent_heading, detected_bool, visible_bool)
        self.obs_buf_agent = torch.cat((old_sense_rel_pos,
                                  sense_rel_pos,
                                  old_sense_rel_yaw_global,
                                  sense_rel_yaw_global,
                                  old_sense_rel_yaw_local,
                                  sense_rel_yaw_local,
                                  old_detect_bool,
                                  detected_bool.long(),
                                  old_visible_bool,
                                  visible_bool.long(),
                                  ), dim=-1)

        # print("NEW self.obs_buf_agent: ", self.obs_buf_agent)
        # print("[DecHighLevelGame] in compute_observations_agent: ", self.obs_buf_agent[0, :])

    def get_observations_agent(self):
        self.compute_observations_agent()
        return self.obs_buf_agent

    def get_observations_robot(self):
        self.compute_observations_robot()
        return self.obs_buf_robot

    def get_privileged_observations_agent(self):
        return self.privileged_obs_buf_agent

    def get_privileged_observations_robot(self):
        return self.privileged_obs_buf_robot

    def robot_sense_agent(self, partial_obs):
        """
        Args:
            partial_obs (bool): true if partial observability, false if otherwise

        Returns: sensing information the POV of the robot. Returns five values:

            sense_rel_pos (torch.Tensor): [num_envs * 3] if is visible, contains the relative agent position;
                                                if not visible, copies the last relative agent position
            sense_rel_yaw_global (torch.Tensor): [num_envs * 2] if visible, contains cos(global_yaw) and sin(global_yaw);
                                                if not visible, copies the last cos / sin of last global yaw
            sense_rel_yaw_local (torch.Tensor): [num_envs * 2] if visible, contains cos(local_yaw) and sin(local_yaw);
                                                if not visible, copies the last cos / sin of last local yaw
            detected_buf_agent (torch.Tensor): [num_envs * 1] boolean if the agent has been detected before
            visible_bool (torch.Tensor): [num_envs * 1] boolean if the agent has is currently visible

        TODO: I treat the FOV the same for horiz, vert, etc. For realsense camera:
                (HorizFOV)  = 64 degrees (~1.20428 rad)
                (DiagFOV)   = 72 degrees
                (VertFOV)   = 41 degrees
        """
        half_fov = 1.20428 / 2.

        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        _, _, robot_yaw = get_euler_xyz(self.ll_env.base_quat)
        robot_yaw = wrap_to_pi(robot_yaw)

        # relative yaw between the agent's heading and the robot's heading (global)
        rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        rel_yaw_global = rel_yaw_global.unsqueeze(-1)
        rel_yaw_global = wrap_to_pi(rel_yaw_global)

        # relative yaw between the agent's heading and the robot's heading (local)
        # robot_forward = quat_apply_yaw(self.ll_env.base_quat, self.ll_env.forward_vec)
        # robot_heading = torch.atan2(robot_forward[:, 1], robot_forward[:, 0])
        rel_yaw_local = wrap_to_pi(robot_yaw - self.agent_heading) # make sure between -pi and pi
        rel_yaw_local = rel_yaw_local.unsqueeze(-1)

        # pack the final sensing measurements
        sense_rel_pos = rel_agent_pos_xyz.clone()
        sense_rel_yaw_global = torch.cat((torch.cos(rel_yaw_global), torch.sin(rel_yaw_global)), dim=-1)
        sense_rel_yaw_local = torch.cat((torch.cos(rel_yaw_local), torch.sin(rel_yaw_local)), dim=-1)

        # find environments where robot is visible
        fov_bool = torch.any(torch.abs(rel_yaw_global) <= half_fov, dim=1)
        visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()
        hidden_env_ids = (fov_bool == 0).nonzero(as_tuple=False).flatten()

        # mark envs where robot has been detected at least once
        self.detected_buf_robot[visible_env_ids, :] = 1

        # mark envs where we robot was visible
        visible_bool = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device, requires_grad=False)
        visible_bool[visible_env_ids, :] = 1

        if partial_obs:
            # if we didn't sense the robot now, copy over the agent's previous sensed position as our new "measurement"
            sense_rel_pos[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.pos_idxs[9:]].clone()
            sense_rel_yaw_global[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.ang_global_idxs[6:]].clone()
            sense_rel_yaw_local[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.ang_local_idxs[6:]].clone()

        # print("sense_rel_pos: ", sense_rel_pos)
        # print("rel_yaw_global: ", rel_yaw_global)
        # print("rel_yaw_local: ", rel_yaw_local)
        # print("detected_buf_robot: ", self.detected_buf_robot)
        # print("visible_bool: ", visible_bool)

        return sense_rel_pos, sense_rel_yaw_global, sense_rel_yaw_local, self.detected_buf_robot, visible_bool

    def agent_sense_robot(self):
        """
        Returns: sensing information the POV of the agent. Returns five values:

            sense_rel_pos (torch.Tensor): [num_envs * 3] if is visible, contains the relative robot position;
                                                if not visible, copies the last relative robot position
            sense_rel_yaw_global (torch.Tensor): [num_envs * 2] if visible, contains cos(global_yaw) and sin(global_yaw);
                                                if not visible, copies the last cos / sin of last global yaw
            sense_rel_yaw_local (torch.Tensor): [num_envs * 2] if visible, contains cos(local_yaw) and sin(local_yaw);
                                                if not visible, copies the last cos / sin of last local yaw
            detected_buf_agent (torch.Tensor): [num_envs * 1] boolean if the robot has been detected before
            visible_bool (torch.Tensor): [num_envs * 1] boolean if the robot has is currently visible
        """
        half_fov = 1.20428 / 2. # full FOV is ~64 degrees; same FOV as robot

        # relative position of robot w.r.t. agent
        rel_robot_pos = self.robot_states[:, :3] - self.agent_pos

        # angle between the agent's heading and the robot's COM (global)
        angle_btwn_agents_global = torch.atan2(rel_robot_pos[:, 1], rel_robot_pos[:, 0]) - self.agent_heading
        angle_btwn_agents_global = wrap_to_pi(angle_btwn_agents_global.unsqueeze(-1)) # make sure between -pi and pi

        # relative heading between the agent's heading and the robot's heading (local)
        robot_forward = quat_apply_yaw(self.ll_env.base_quat, self.ll_env.forward_vec)
        robot_heading = torch.atan2(robot_forward[:, 1], robot_forward[:, 0])
        angle_btwn_heading_local = wrap_to_pi(robot_heading - self.agent_heading) # make sure between -pi and pi
        angle_btwn_heading_local = angle_btwn_heading_local.unsqueeze(-1)

        # pack the final sensing measurements
        sense_rel_pos = rel_robot_pos.clone()
        sense_rel_yaw_global = torch.cat((torch.cos(angle_btwn_agents_global),
                                              torch.sin(angle_btwn_agents_global)), dim=-1)
        sense_rel_yaw_local = torch.cat((torch.cos(angle_btwn_heading_local),
                                              torch.sin(angle_btwn_heading_local)), dim=-1)

        # print("[agent_sense_robot] rel_robot_pos: ", rel_robot_pos)
        # print("[agent_sense_robot] torch.atan2(rel_robot_pos[:, 1], rel_robot_pos[:, 0]): ", torch.atan2(rel_robot_pos[:, 1], rel_robot_pos[:, 0]))
        # print("[agent_sense_robot] self.agent_heading: ", self.agent_heading)
        # print("[agent_sense_robot] robot_heading: ", robot_heading)
        # print("[agent_sense_robot] angle_btwn_heading_local: ", angle_btwn_heading_local)
        # print("[agent_sense_robot] angle_btwn_agents_global: ", angle_btwn_agents_global)
        # print("[agent_sense_robot] half_fov: ", half_fov)

        # find environments where robot is visible
        fov_bool = torch.any(torch.abs(angle_btwn_agents_global) <= half_fov, dim=1)
        visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()
        hidden_env_ids = (fov_bool == 0).nonzero(as_tuple=False).flatten()

        # mark envs where robot has been detected at least once
        self.detected_buf_agent[visible_env_ids, :] = 1

        # mark envs where we robot was visible
        visible_bool = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device, requires_grad=False)
        visible_bool[visible_env_ids, :] = 1

        # if we didn't sense the robot now, copy over the agent's previous sensed position as our new "measurement"
        sense_rel_pos[hidden_env_ids, :] = self.obs_buf_agent[hidden_env_ids][:, self.pos_idxs[9:]].clone()
        sense_rel_yaw_global[hidden_env_ids, :] = self.obs_buf_agent[hidden_env_ids][:, self.ang_global_idxs[6:]].clone()
        sense_rel_yaw_local[hidden_env_ids, :] = self.obs_buf_agent[hidden_env_ids][:, self.ang_local_idxs[6:]].clone()

        return sense_rel_pos, sense_rel_yaw_global, sense_rel_yaw_local, self.detected_buf_agent, visible_bool

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    # ------------- Callbacks --------------#

    def _update_agent_states(self):
        """Goes to the low-level environment and grabs the most recent simulator states for the agent and robot."""
        self.robot_states = self.ll_env.root_states[self.ll_env.robot_indices, :]
        self.base_quat = self.robot_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot_states[:, 10:13])
        self.agent_pos = self.ll_env.root_states[self.ll_env.agent_indices, :3]
        self.agent_states = self.ll_env.root_states[self.ll_env.agent_indices, :]

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.robot_states = self.ll_env.root_states[self.ll_env.robot_indices, :]
        self.base_quat = self.robot_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot_states[:, 10:13])

        self.agent_pos = self.ll_env.root_states[self.ll_env.agent_indices, :3]
        self.agent_states = self.ll_env.root_states[self.ll_env.agent_indices, :]
        self.agent_heading = self.ll_env.init_agent_heading.clone()

    def _prepare_reward_function_agent(self):
        """ Prepares a list of reward functions for the agent, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales_agent.keys()):
            scale = self.reward_scales_agent[key]
            if scale == 0:
                self.reward_scales_agent.pop(key)
            else:
                self.reward_scales_agent[key] *= self.ll_env.dt
        # prepare list of functions
        self.reward_functions_agent = []
        self.reward_names_agent = []
        for name, scale in self.reward_scales_agent.items():
            if name == "termination":
                continue
            self.reward_names_agent.append(name)
            name = '_reward_' + name
            print("[_prepare_reward_function_agent]: reward = ", name)
            print("[_prepare_reward_function_agent]:     scale: ", scale)
            self.reward_functions_agent.append(getattr(self, name))

        # reward episode sums
        self.episode_sums_agent = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales_agent.keys()}

    def _prepare_reward_function_robot(self):
        """ Prepares a list of reward functions for the robot, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales_robot.keys()):
            scale = self.reward_scales_robot[key]
            if scale == 0:
                self.reward_scales_robot.pop(key)
            else:
                self.reward_scales_robot[key] *= self.ll_env.dt
        # prepare list of functions
        self.reward_functions_robot = []
        self.reward_names_robot = []
        for name, scale in self.reward_scales_robot.items():
            if name == "termination":
                continue
            self.reward_names_robot.append(name)
            name = '_reward_' + name
            print("[_prepare_reward_function_robot]: reward = ", name)
            print("[_prepare_reward_function_robot]:     scale: ", scale)
            self.reward_functions_robot.append(getattr(self, name))

        # reward episode sums
        self.episode_sums_robot = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales_robot.keys()}

    def _parse_cfg(self, cfg):
        # self.dt = self.cfg.control.decimation * self.sim_params.dt
        # self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales_robot = class_to_dict(self.cfg.rewards_robot.scales)
        self.reward_scales_agent = class_to_dict(self.cfg.rewards_agent.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.ll_env.dt)

        # self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.ll_env.dt)

    # ------------ reward functions----------------
    def _reward_evasion(self):
        """Reward for evading"""
        return torch.norm(self.agent_pos[:, :2] - self.robot_states[:, :2], p=1, dim=-1)

    def _reward_pursuit(self):
        """Reward for pursuing"""
        # print("agent_pos: ", self.agent_pos[:, :2], "   |   ll_env.root_state agent pos: ", self.ll_env.root_states[self.ll_env.agent_indices, :2])
        # print("robot_state pos: ", self.robot_states[:, :2], "   |   ll_env.root_state robot pos: ", self.ll_env.root_states[self.ll_env.robot_indices, :2])
        # error = torch.norm(self.agent_pos[:, :2] - self.robot_states[:, :2], p=1, dim=-1)
        # rew = torch.exp(-error) - 0.8 # L1 norm is better than L2

        # rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        # _, _, robot_yaw = get_euler_xyz(self.ll_env.base_quat)
        # robot_yaw = wrap_to_pi(robot_yaw)
        #
        # rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        # rel_yaw_global = wrap_to_pi(rel_yaw_global)
        #
        # scale = torch.exp(-torch.abs(rel_yaw_global))
        # rew = scale * dist
        rew = torch.square(torch.norm(self.agent_pos[:, :2] - self.robot_states[:, :2], p=2, dim=-1))
        return rew

    def _reward_robot_ctrl(self):
        """Reward for robot controls"""
        return torch.norm(self.ll_env.commands, dim=-1)

    def _reward_facing_agent(self):
        """Reward for the robot facing the agemt"""
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        _, _, robot_yaw = get_euler_xyz(self.ll_env.base_quat)
        robot_yaw = wrap_to_pi(robot_yaw)

        rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        rel_yaw_global = wrap_to_pi(rel_yaw_global)
        thresh = 0.1
        rew = torch.exp(-torch.abs(rel_yaw_global))
        rew[rew < thresh] = 0.0
        return rew
