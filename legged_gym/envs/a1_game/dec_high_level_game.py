
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
from legged_gym.filters.kalman_filter import KalmanFilter
from legged_gym.filters.ma_kalman_filter import MultiAgentKalmanFilter
from legged_gym.filters.kf_torchfilter import UKFTorchFilter
import sys

import pickle
from datetime import datetime

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
        ll_env_cfg.terrain.measure_heights = self.cfg.terrain.measure_heights

        # TODO: HACK! Bump up the torque penalty to simulate "instantaneous control effort cost"
        # before: -0.00001;
        # ll_env_cfg.rewards.scales.torques = -5.

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
        ll_env_cfg.commands.ranges.lin_vel_x = self.cfg.commands.ranges.lin_vel_x
        ll_env_cfg.commands.ranges.lin_vel_y = self.cfg.commands.ranges.lin_vel_y
        ll_env_cfg.commands.ranges.ang_vel_yaw = self.cfg.commands.ranges.ang_vel_yaw
        ll_env_cfg.commands.ranges.heading = self.cfg.commands.ranges.heading
        ll_env_cfg.domain_rand.randomize_friction = self.cfg.domain_rand.randomize_friction
        ll_env_cfg.domain_rand.friction_range = self.cfg.domain_rand.friction_range
        ll_env_cfg.domain_rand.randomize_base_mass = self.cfg.domain_rand.randomize_base_mass
        ll_env_cfg.domain_rand.added_mass_range = self.cfg.domain_rand.added_mass_range
        ll_env_cfg.domain_rand.push_robots = self.cfg.domain_rand.push_robots
        ll_env_cfg.domain_rand.push_interval_s = self.cfg.domain_rand.push_interval_s
        ll_env_cfg.domain_rand.max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        ll_env_cfg.noise.add_noise = self.cfg.noise.add_noise
        ll_env_cfg.noise.noise_level = self.cfg.noise.noise_level

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

        # parse the high-level config into appropriate dicts
        self._parse_cfg(self.cfg)

        self.gym = gymapi.acquire_gym()

        self.num_envs = self.cfg.env.num_envs

        # setup the capture distance between two agents
        self.capture_dist = self.cfg.env.capture_dist
        self.MAX_REL_POS = 100.

        # setup sensing params about robot
        self.robot_full_fov = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.robot_full_fov[:] = self.cfg.robot_sensing.fov

        # if using a FOV curriculum, initialize it to the starting FOV
        self.fov_curr_idx = 0
        if self.cfg.robot_sensing.fov_curriculum:
            self.robot_full_fov[:] = self.cfg.robot_sensing.fov_levels[self.fov_curr_idx]

        # if using a PREY curriculum, initialize it with starting prey relative angle range
        self.prey_curr_idx = 0
        self.max_rad = 6.0
        self.min_rad = 2.0 
        if self.cfg.robot_sensing.prey_curriculum:
            max_ang = self.cfg.robot_sensing.prey_angs[self.prey_curr_idx]
            min_ang = -max_ang
            rand_angle = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False).uniform_(min_ang, max_ang)
            rand_radius = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False).uniform_(self.min_rad, self.max_rad)
            self.agent_offset_xyz = torch.cat((rand_radius * torch.cos(rand_angle),
                                               rand_radius * torch.sin(rand_angle),
                                               torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
                                               ), dim=-1)
            self.ll_env.agent_offset_xyz = self.agent_offset_xyz

            # reset the low-level environment with the new initial prey positions 
            self.ll_env._reset_root_states(torch.arange(self.num_envs, device=self.device))


        # this keeps track of which training iteration we are in!
        self.training_iter_num = 0
        self.curriculum_target_iters = self.cfg.robot_sensing.curriculum_target_iters

        print("[DecHighLevelGame] robot full fov is: ", self.robot_full_fov[0])

        # robot policy info
        self.num_obs_robot = self.cfg.env.num_observations_robot
        self.num_privileged_obs_robot = self.cfg.env.num_privileged_obs_robot
        self.num_actions_robot = self.cfg.env.num_actions_robot

        # agent policy info
        self.num_obs_agent = self.cfg.env.num_observations_agent
        self.num_privileged_obs_agent = self.cfg.env.num_privileged_obs_agent
        self.num_actions_agent = self.cfg.env.num_actions_agent

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # *SIMPLER* PARTIAL obs space: setup the indicies of relevant observation quantities
        # self.pos_idxs_robot = list(range(0,12)) # all relative position history
        # self.visible_idxs_robot = list(range(12,16))  # all visible bools

        # PARTIAL obs space: setup the indicies of relevant observation quantities
        # self.pos_idxs_robot = list(range(0,12)) # all relative position history
        # self.ang_global_idxs_robot = list(range(12,20)) # all relative (global) angle
        # self.visible_idxs_robot = list(range(20, 24))  # all visible bools
        ## self.detect_idxs_robot = list(range(20, 24))  # all detected bools
        ## self.visible_idxs_robot = list(range(24, 28))  # all visible bools

        # FULL obs space: but with extra info
        # self.pos_idxs_robot = list(range(0,3)) # all relative position history
        # self.ang_global_idxs_robot = list(range(3,5)) # all relative (global) angle
        # self.detect_idxs_robot = list(range(5,6))  # all detected bools
        # self.visible_idxs_robot = list(range(6,7))  # all visible bools

        # KALMAN FILTER obs space
        # self.pos_idxs_robot = list(range(0,4))
        # self.cov_idxs_robot = list(range(4,20))

        self.pos_idxs_robot = list(range(0, 3))

        # allocate robot buffers
        # self.obs_buf_robot = self.MAX_REL_POS * torch.ones(self.num_envs, self.num_obs_robot, device=self.device, dtype=torch.float)
        # self.obs_buf_robot[:, -2:] = 0. # reset the robot's sense booleans to zero
        # self.obs_buf_robot = torch.zeros(self.num_envs, self.num_obs_robot, device=self.device, dtype=torch.float)
        # self.obs_buf_robot[:, self.pos_idxs_robot] = self.MAX_REL_POS
        self.obs_buf_robot = torch.zeros(self.num_envs, self.num_obs_robot, device=self.device, dtype=torch.float)
        self.obs_buf_robot[:, self.pos_idxs_robot] = self.MAX_REL_POS
        self.rew_buf_robot = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # setup the low-level observation buffer
        self.ll_obs_buf_robot = self.ll_env.obs_buf

        # allocate agent buffers
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

        # TODO: test!
        self.last_robot_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

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

        # Create the Kalman Filter
        self.filter_type = self.cfg.robot_sensing.filter_type
        self.real_traj = None
        self.z_traj = None
        self.est_traj = None
        self.P_traj = None 
        # self.num_states_kf = 3 # (x, y, dtheta)
        self.num_states_kf = 4  # (x, y, z, dtheta)
        self.num_actions_kf_r = self.num_actions_robot 
        self.num_actions_kf_a = self.num_actions_agent
        self.dyn_sys_type = "linear"
        self.filter_dt = self.ll_env.dt # we get measurements slower than sim: filter_dt = decimation * sim.dt
        print("[DecHighLevelGame] Filter type: ", self.filter_type)
        if self.filter_type == "kf":
            self.kf = KalmanFilter(self.filter_dt,
                                    self.num_states_kf,
                                    self.num_actions_kf_r,
                                    self.num_envs,
                                    state_type="pos_ang",
                                    device=self.device,
                                    dtype=torch.float)
        elif self.filter_type == "ukf":
            self.kf = UKFTorchFilter(state_dim=self.num_states_kf,
                                    control_dim=self.num_actions_kf_r,
                                    observation_dim=self.num_states_kf,
                                    num_envs=self.num_envs,
                                    dt=self.filter_dt,
                                    device=self.device,
                                    ang_dims=-1,
                                    dyn_sys_type=self.dyn_sys_type)
            self.kf_og = KalmanFilter(self.filter_dt,
                                    self.num_states_kf,
                                    self.num_actions_kf_r,
                                    self.num_envs,
                                    state_type="pos_ang",
                                    device=self.device,
                                    dtype=torch.float)
        elif self.filter_type == "makf":
            self.kf = MultiAgentKalmanFilter(self.filter_dt,
                                    self.num_states_kf,
                                    self.num_actions_kf_r,
                                    self.num_actions_kf_a,
                                    self.num_envs,
                                    state_type="pos_ang",
                                    device=self.device,
                                    dtype=torch.float)
        else:
            print("[ERROR]! Invalid filter type: ", self.filter_type)
            return -1

        # KF data saving info
        self.save_kf_data = False
        self.data_save_tstep = 0
        self.data_save_interval = 50

        self.weaving_tstep = 0
        self.weaving_ctrl_idx = 0

        # Reward debugging saving info
        self.save_rew_data = False
        self.rew_debug_tstep = 0
        self.rew_debug_interval = 100

        # Reward debugging
        self.fov_reward = None
        self.fov_rel_yaw = None
        self.ll_env_command_ang_vel = None
        # self.r_start = None
        # self.r_goal = None
        # self.r_curr = None
        # self.r_curr_proj = None
        # self.r_last = None
        # self.r_last_proj = None

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
        # command_agent *= 0
        # command_agent = self._straight_line_command_agent()
        command_agent = self._weaving_command_agent()

        # rel_yaw = self.get_rel_yaw_global_robot()
        # robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        # _, _, robot_yaw = get_euler_xyz(robot_base_quat)
        # robot_yaw = wrap_to_pi(robot_yaw)
        #
        # ang_error = wrap_to_pi(command_robot[:, 0] - robot_yaw)

        # command_robot_3d = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # command_robot_3d[:, 2] = command_robot[:, 0] #ang_error.squeeze(-1)
        # command_robot = self._straight_line_command_augmented_robot(command_robot)
        # command_robot = self._straight_line_command_robot(command_robot)
        # command_robot = self._turn_and_pursue_command_robot(command_robot)
        # TODO: HACK!! This is for debugging.

        # rel_pos = self.agent_pos[:, :3] - self.robot_states[:, :3]
        # rel_yaw_global = self.get_rel_yaw_global_robot()
        # print("[RAW] command robot: ", command_robot)
        # print("KF state estimate: ", self.kf.filter._belief_mean)
        # print("real state: ", torch.cat((rel_pos, rel_yaw_global), dim=-1))
        # print("[RAW] command agent: ", command_agent)

        # clip the robot and agent's commands
        command_robot = self.clip_command_robot(command_robot)
        # command_agent = self.clip_command_agent(command_agent)

        # print("[CLIPPED] command robot: ", command_robot)

        # NOTE: low-level policy requires 4D control
        # update the low-level simulator command since it deals with the robot
        ll_env_command_robot = torch.cat((command_robot,
                                          torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)), # heading
                                          dim=-1)
        self.ll_env.commands = ll_env_command_robot

        # record the last robot state before simulating physics
        self.last_robot_pos[:] = self.robot_states[:, :3]

        # get the low-level actions as a function of low-level obs
        # self.ll_env.compute_observations() # refresh the observation buffer!
        # self.ll_env._clip_obs()
        # ll_robot_obs = self.ll_env.get_observations()
        ll_robot_actions = self.ll_policy(self.ll_obs_buf_robot.detach())

        # forward simulate the low-level actions
        self.ll_obs_buf_robot, _, ll_rews, ll_dones, _ = self.ll_env.step(ll_robot_actions.detach())

        # forward simulate the agent action too
        # self.step_agent_single_integrator(command_agent)
        self.step_agent_dubins_car(command_agent)

        # take care of terminations, compute observations, rewards, and dones
        self.post_physics_step(ll_rews, ll_dones, command_robot, command_agent)

        # # update robot's visual sensing curriculum; do this for all envs
        # if self.cfg.robot_sensing.fov_curriculum:
        #     self._update_robot_sensing_curriculum(torch.arange(self.num_envs, device=self.device))

        self.weaving_tstep += 1

        return self.obs_buf_agent, self.obs_buf_robot, self.privileged_obs_buf_agent, self.privileged_obs_buf_robot, self.rew_buf_agent, self.rew_buf_robot, self.reset_buf, self.extras


    def _straight_line_command_augmented_robot(self, command_robot_1d):
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        command_robot = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        command_robot[:, 0] = torch.clip(rel_agent_pos_xyz[:, 0], min=self.command_ranges["lin_vel_x"][0],
                                         max=self.command_ranges["lin_vel_x"][1])
        command_robot[:, 1] = torch.clip(rel_agent_pos_xyz[:, 1], min=self.command_ranges["lin_vel_y"][0],
                                         max=self.command_ranges["lin_vel_y"][1])
        command_robot[:, 2] = 0
        command_robot[:, :2] += command_robot_1d

        return command_robot

    def _straight_line_command_robot(self, command_robot):
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        command_robot[:, 0] = torch.clip(rel_agent_pos_xyz[:, 0], min=self.command_ranges["lin_vel_x"][0],
                                         max=self.command_ranges["lin_vel_x"][1])
        command_robot[:, 1] = torch.clip(rel_agent_pos_xyz[:, 1], min=self.command_ranges["lin_vel_y"][0],
                                         max=self.command_ranges["lin_vel_y"][1])
        command_robot[:, 2] = 0

        return command_robot

    def _turn_and_pursue_command_robot(self, command_robot):
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        _, _, robot_yaw = get_euler_xyz(robot_base_quat)
        robot_yaw = wrap_to_pi(robot_yaw)

        rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        rel_yaw_global = wrap_to_pi(rel_yaw_global)

        for env_idx in range(self.num_envs):
            if torch.abs(rel_yaw_global[env_idx]) < 0.15: # pursue
                # command_robot[env_idx, 0] = 1
                # command_robot[env_idx, 1] = 0
                # command_robot[env_idx, 2] = 0
                command_robot[env_idx, :2] = -1 * torch.clip(rel_agent_pos_xyz[env_idx, :2], min=-1, max=1)
                command_robot[env_idx, 2] = 0
            else: # turn
                command_robot[env_idx, 0] = 0.
                command_robot[env_idx, 1] = 0. # small bias, just for fun
                command_robot[env_idx, 2] = 1
        return command_robot

    def _straight_line_command_agent(self):
        # TODO: This assumes agent is prey
        rel_robot_pos_xyz = self.robot_states[:, :3] - self.agent_pos[:, :3]
        command_agent = torch.zeros(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)
        command_agent[:, 0] = -1 * torch.clip(rel_robot_pos_xyz[:, 0], min=self.command_ranges["agent_lin_vel_x"][0],
                                         max=self.command_ranges["agent_lin_vel_x"][1])
        command_agent[:, 1] = -1 * torch.clip(rel_robot_pos_xyz[:, 1], min=self.command_ranges["agent_lin_vel_y"][0],
                                         max=self.command_ranges["agent_lin_vel_y"][1])

        return command_agent

    def _weaving_command_agent(self):
        # TODO: This assumes agent is prey
        command_agent = torch.zeros(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)
        switch_idx = 100
        if self.weaving_tstep % switch_idx == 0:
            self.weaving_ctrl_idx += 1
            self.weaving_ctrl_idx %= 2
            self.weaving_tstep = 0

        if self.weaving_ctrl_idx == 0:
            command_agent[:, 0] = 2*self.command_ranges["agent_lin_vel_x"][1]
            command_agent[:, 1] = self.command_ranges["agent_ang_vel_yaw"][0]
        elif self.weaving_ctrl_idx == 1:
            command_agent[:, 0] = 2*self.command_ranges["agent_lin_vel_x"][1]
            command_agent[:, 1] = self.command_ranges["agent_ang_vel_yaw"][1]

        return command_agent

    def clip_command_robot(self, command_robot):
        """Clips the robot's commands"""
        # clip the robot's commands
        command_robot[:, 0] = torch.clip(command_robot[:, 0], min=self.command_ranges["lin_vel_x"][0], max=self.command_ranges["lin_vel_x"][1])
        command_robot[:, 1] = torch.clip(command_robot[:, 1], min=self.command_ranges["lin_vel_y"][0], max=self.command_ranges["lin_vel_y"][1])
        command_robot[:, 2] = torch.clip(command_robot[:, 2], min=self.command_ranges["ang_vel_yaw"][0], max=self.command_ranges["ang_vel_yaw"][1])

        # set small commands to zero; this was originally used by lowlevel code to enable robot to learn to stand still
        command_robot[:, :2] *= (torch.norm(command_robot[:, :2], dim=1) > 0.2).unsqueeze(1)

        return command_robot

    def clip_command_agent(self, command_agent):
        """Clips the agent's commands"""
        command_agent[:, 0] = torch.clip(command_agent[:, 0], min=self.command_ranges["agent_lin_vel_x"][0], max=self.command_ranges["agent_lin_vel_x"][1])
        command_agent[:, 1] = torch.clip(command_agent[:, 1], min=self.command_ranges["agent_lin_vel_y"][0], max=self.command_ranges["agent_lin_vel_y"][1])
        command_agent[:, 2] = torch.clip(command_agent[:, 2], min=self.command_ranges["agent_ang_vel_yaw"][0], max=self.command_ranges["agent_ang_vel_yaw"][1])
        return command_agent

    def step_agent_omnidirectional(self, command_agent):
        """
        Steps agent modeled as an omnidirectional agent
            x' = x + dt * u1 * cos(theta)
            y' = y + dt * u2 * sin(theta)
            z' = z
            theta' = theta + dt * u3
        where the control is 3-dimensional:
            u = [u1 u2 u3] = [vx, vy, omega]
        """
        lin_vel_x = command_agent[:, 0]
        lin_vel_y = command_agent[:, 1]
        ang_vel = command_agent[:, 2]

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

    def step_agent_single_integrator(self, command_agent):
        """
        Steps agent modeled as a single integrator:
            x' = x + dt * u1/4
            y' = y + dt * u2
            z' = z          (assume z-dim is constant)
        """

        # agent that has full observability of the robot state
        command_lin_vel_x = command_agent[:, 0]
        command_lin_vel_y = command_agent[:, 1]

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

    def step_agent_dubins_car(self, command_agent):
        """
        Steps agent modeled as a single integrator:
            x' = x + dt * u1 * cos(yaw)
            y' = y + dt * u1 * cos(yaw)
            yaw' = yaw + dt * u2
        """

        # agent that has full observability of the robot state
        command_lin_vel = command_agent[:, 0]
        command_ang_vel = command_agent[:, 1]

        # TODO: agent gets simulated at the same Hz as the low-level controller!
        for _ in range(self.ll_env.cfg.control.decimation):
            self.agent_pos[:, 0] += self.ll_env.cfg.sim.dt * command_lin_vel * torch.cos(self.agent_heading)
            self.agent_pos[:, 1] += self.ll_env.cfg.sim.dt * command_lin_vel * torch.sin(self.agent_heading)
            self.agent_heading += self.ll_env.cfg.sim.dt * command_ang_vel
            self.agent_heading = wrap_to_pi(self.agent_heading)

        heading_quat = quat_from_angle_axis(self.agent_heading, self.z_unit_tensor)

        # update the simulator state for the agent!
        self.ll_env.root_states[self.ll_env.agent_indices, :3] = self.agent_pos
        self.ll_env.root_states[self.ll_env.agent_indices, 3:7] = heading_quat

        self.ll_env.root_states[self.ll_env.agent_indices, 7] = command_lin_vel # lin vel x
        self.ll_env.root_states[self.ll_env.agent_indices, 8] = 0. # lin vel y
        self.ll_env.root_states[self.ll_env.agent_indices, 9] = 0.  # lin vel z
        self.ll_env.root_states[self.ll_env.agent_indices, 10:13] = 0. # ang vel xyz

        # self.ll_env.gym.set_actor_root_state_tensor(self.ll_env.sim, gymtorch.unwrap_tensor(self.ll_env.root_states))
        agent_env_ids_int32 = self.ll_env.agent_indices.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.ll_env.sim,
                                                     gymtorch.unwrap_tensor(self.ll_env.root_states),
                                                     gymtorch.unwrap_tensor(agent_env_ids_int32),
                                                     len(agent_env_ids_int32))

    def post_physics_step(self, ll_rews, ll_dones, command_robot, command_agent):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
        """
        self.gym.refresh_actor_root_state_tensor(self.ll_env.sim)
        self.gym.refresh_net_contact_force_tensor(self.ll_env.sim)

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
        # TODO: THIS IS A HACK!
        # self.reset_buf |= self.ll_env.reset_buf # combine the low-level dones and the high-level dones
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.reset_idx(env_ids)

        # refresh the low-level policy's observation buf
        self.ll_env.compute_observations()
        self.ll_env._update_last_quantities()
        self.ll_env._clip_obs()
        self.ll_obs_buf_robot = self.ll_env.get_observations()

        # compute the high-level observation for each agent
        self.compute_observations_agent()               # compute high-level observations for agent
        self.compute_observations_robot(command_robot, command_agent)  # compute high-level observations for robot

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

        # reset low-level buffers
        self.ll_env.last_actions[env_ids] = 0.
        self.ll_env.last_dof_vel[env_ids] = 0.
        self.ll_env.feet_air_time[env_ids] = 0.
        self.ll_env.episode_length_buf[env_ids] = 0
        self.ll_env.reset_buf[env_ids] = 1

        # update the state variables
        self._update_agent_states()
        self.agent_heading[env_ids] = self.ll_env.init_agent_heading[env_ids].clone()

        # reset agent buffers
        self.obs_buf_agent[env_ids, :] = -self.MAX_REL_POS
        self.obs_buf_agent[env_ids, -2:] = 0.
        # print("reset obs buf agent: ", self.obs_buf_agent[env_ids, :])

        # reset robot buffers
        reset_robot_obs = torch.zeros(len(env_ids), self.num_obs_robot, device=self.device, requires_grad=False)
        reset_robot_obs[:, self.pos_idxs_robot] = self.MAX_REL_POS # only the initial relative position is reset different
        self.obs_buf_robot[env_ids, :] = reset_robot_obs
        # self.obs_buf_robot[env_ids, :] = self.MAX_REL_POS      # reset the robot's rel_agent_pos to max relative position
        # self.obs_buf_robot[env_ids, -2:] = 0                    # reset the robot's sense booleans to zero

        # reset the kalman filter
        if self.num_states_kf == 4:
            rel_pos = self.agent_pos[:, :3] - self.robot_states[:, :3]
        elif self.num_states_kf == 3:
            rel_pos = self.agent_pos[:, :2] - self.robot_states[:, :2]
        else:
            print("[DecHighLevelGame: reset_idx] ERROR: self.num_states_kf", self.num_states_kf, " is not supported.")
            return

        rel_yaw_global = self.get_rel_yaw_global_robot()
        rel_state = torch.cat((rel_pos, rel_yaw_global), dim=-1)
        
        # ---- choose right filter calls ---- #
        if self.filter_type == "kf":
            xhat0 = self.kf.sim_measurement(rel_state)
            self.kf.reset_xhat(env_ids=env_ids, xhat_val=xhat0[env_ids, :])
        elif self.filter_type == "ukf":
            xhat0 = self.kf.sim_observations(states=rel_state)
            self.kf.reset_mean_cov(env_ids=env_ids, mean=xhat0[env_ids, :])
            xhat0_og = self.kf_og.sim_measurement(rel_state)
            self.kf_og.reset_xhat(env_ids=env_ids, xhat_val=xhat0[env_ids, :])
            # import pdb; pdb.set_trace()

        # reset the high-level buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.curr_episode_step[env_ids] = 0
        self.capture_buf[env_ids] = 0
        self.detected_buf_agent[env_ids, :] = 0
        self.detected_buf_robot[env_ids, :] = 0

        self.last_robot_pos[env_ids, :] = self.ll_env.env_origins[env_ids, :3] # reset last robot position to the reset pos

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
        if self.ll_env.cfg.terrain.measure_heights:
            self.ll_env.measured_heights = self.ll_env._get_heights() # TODO: need to do this for the observation buffer to match in size!
        obs_agent, obs_robot, privileged_obs_agent, privileged_obs_robot, _, _, _, _ = self.step(actions_agent, actions_robot)
        return obs_agent, obs_robot, privileged_obs_agent, privileged_obs_robot

    def get_rel_yaw_global_robot(self):
        """Returns relative angle between the robot's local yaw angle and
        the angle between the robot's base and the agent's base:
            i.e., angle_btw_bases - robot_yaw
        """
        # from robot's POV, get its sensing
        rel_pos = self.agent_pos[:, :3] - self.robot_states[:, :3]

        # get relative yaw between the agent's heading and the robot's heading (global)
        robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        _, _, robot_yaw = get_euler_xyz(robot_base_quat)
        robot_yaw = wrap_to_pi(robot_yaw)

        rel_yaw_global = torch.atan2(rel_pos[:, 1], rel_pos[:, 0]) - robot_yaw
        rel_yaw_global = rel_yaw_global.unsqueeze(-1)
        rel_yaw_global = wrap_to_pi(rel_yaw_global)
        return rel_yaw_global

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

        # rew_name = "path_progress"
        # print("episode_sum_robot[", rew_name, "]: ", self.episode_sums_robot[rew_name])

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

    def compute_observations_robot(self, command_robot, command_agent):
        """ Computes observations of the robot robot
        """

        # self.compute_observations_pos_robot()             # OBS: (x_rel)
        # self.compute_observations_pos_angle_robot()       # OBS: (x_rel, yaw_rel)
        # self.compute_observations_full_obs_robot()        # OBS: (x_rel, cos(yaw_rel), sin(yaw_rel), d_bool, v_bool)
        # self.compute_observations_limited_FOV_robot()     # OBS: (x_rel^{t:t-4}, cos_yaw^{t:t-4}, sin_yaw_rel^{t:t-4}, d_bool^{t:t-4}, v_bool^{t:t-4})
        # self.compute_observations_state_hist_robot(limited_fov=True)          # OBS: (x_rel^{t:t-4}, v_bool^{t:t-4})
        self.compute_observations_KF_robot(command_robot, command_agent, limited_fov=True)   # OBS: (hat{x}_rel, hat{P})

        # print("[DecHighLevelGame] self.obs_buf_robot: ", self.obs_buf_robot)

    def compute_observations_pos_robot(self):
        """ Computes observations of the agent with full FOV
        """
        # from robot's POV, get the relative position to the agent
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        self.obs_buf_robot = rel_agent_pos_xyz

    def compute_observations_pos_angle_robot(self):
        """ Computes observations of the agent with full FOV
        """
        # from robot's POV, get the relative position to the agent
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        _, _, robot_yaw = get_euler_xyz(robot_base_quat)
        robot_yaw = wrap_to_pi(robot_yaw)

        rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        rel_yaw_global = rel_yaw_global.unsqueeze(-1)
        rel_yaw_global = wrap_to_pi(rel_yaw_global)

        # rel_yaw_global = self.get_rel_yaw_global_robot()

        # print("num positive angs: ", len((rel_yaw_global > 0).nonzero()))
        # print("num neg angs: ", len((rel_yaw_global <= 0).nonzero()))

        self.obs_buf_robot = torch.cat((rel_agent_pos_xyz * 0.1,
                                        rel_yaw_global # robot_yaw.unsqueeze(-1)
                                      #torch.cos(rel_yaw_global),
                                      #torch.sin(rel_yaw_global)
                                      ), dim=-1)

    def compute_observations_full_obs_robot(self):
        """ Computes observations of the robot with partial observability.
            obs_buf is laid out as:

           [s1, sin(h1_g), cos(h1_g), sin(h1_l), cos(h1_l), d1, v1]

           where s1 = (px1, py1, pz1) is the relative position (of size 3)
                 h1_g = is the *global* relative yaw angle between robot yaw and COM of agent (of size 1)
                 h1_l = is the *local* relative yaw angle between robot yaw and agent yaw (of size 1)
                 d1 = is "detected" boolean which tracks if the agent was detected once (of size 1)
                 v1 = is "visible" boolean which tracks if the agent was visible in FOV (of size 1)
        """
        sense_rel_pos, sense_rel_yaw_global, _, detected_bool, visible_bool = self.robot_sense_agent(limited_fov=False)

        self.obs_buf_robot = torch.cat((sense_rel_pos * 0.1,
                                        sense_rel_yaw_global,
                                        detected_bool.long(),
                                        visible_bool.long(),
                                        ), dim=-1)

    def compute_observations_KF_robot(self, command_robot, command_agent, limited_fov=False):
        """ Computes observations of the robot using a Kalman Filter state estimate.
            obs_buf is laid out as:

            [s^t, P^t]

            where: 
                s^t is the Kalman Filter estimated relative position (x, y, z) and dtheta.
                    of shape [num_envs, num_states]
                P^t is a posteriori estimate covariance, flattened
                    of shape [num_envs, num_states*num_states]
        """

        actions_kf_r = command_robot[:, :self.num_actions_kf_r]
        actions_kf_a = command_agent[:, :self.num_actions_kf_a]

        # ---- predict a priori state estimate  ---- #
        if self.filter_type == "kf":
            self.kf.predict(actions_kf_r)
            rel_state_a_priori = self.kf.xhat
        elif self.filter_type == "ukf":
            # import pdb; pdb.set_trace()
            self.kf_og.predict(actions_kf_r)
            rel_state_a_priori_og = self.kf_og.xhat
            self.kf.predict(actions_kf_r)
            rel_state_a_priori = self.kf.filter._belief_mean

        # from robot's POV, get its sensing
        rel_yaw_global = self.get_rel_yaw_global_robot()

        if self.num_states_kf == 4:
            rel_pos = self.agent_pos[:, :3] - self.robot_states[:, :3]
        elif self.num_states_kf == 3:
            rel_pos = self.agent_pos[:, :2] - self.robot_states[:, :2]
        else:
            print("[DecHighLevelGame: reset_idx] ERROR: self.num_states_kf", self.num_states_kf, " is not supported.")
            return

        # simulate getting a noisy measurement
        rel_state = torch.cat((rel_pos, rel_yaw_global), dim=-1)

        if limited_fov is True:
            half_fov = self.robot_full_fov / 2.

            # find environments where robot is visible
            leq = torch.le(torch.abs(rel_yaw_global), half_fov)
            fov_bool = torch.any(leq, dim=1)
            # fov_bool = torch.any(torch.abs(rel_yaw_global) <= half_fov, dim=1)
            visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()
        else:
            # if full FOV, then always get measurement and do update
            visible_env_ids = torch.arange(self.num_envs, device=self.device)

        
        # ---- perform Kalman update to only environments that can get measurements ---- #
        if self.filter_type == "kf":
            z = self.kf.sim_measurement(rel_state)
            self.kf.correct(z, env_ids=visible_env_ids)
            rel_state_a_posteriori = self.kf.xhat.clone()
            covariance_a_posteriori = self.kf.P_tensor.clone()
        elif self.filter_type == "ukf":
            z = self.kf.sim_observations(states=rel_state)
            self.kf.update(z[visible_env_ids, :], env_ids=visible_env_ids)
            rel_state_a_posteriori = self.kf.filter._belief_mean.clone()
            covariance_a_posteriori = self.kf.filter._belief_covariance.clone()
            z_og = self.kf_og.sim_measurement(rel_state)
            self.kf_og.correct(z, env_ids=visible_env_ids)
            rel_state_a_posteriori_og = self.kf_og.xhat.clone()
            covariance_a_posteriori_og = self.kf_og.P_tensor.clone()
            # pdb.set_trace()
        # --------------------------- #

        # print("robot ctrl: ", actions_kf_r)
        # print("real state:", rel_state)
        # print("esimated state: ", self.kf.filter._belief_mean)

        # TODO: Hack for logging!
        if limited_fov is True:
            hidden_env_ids = (~fov_bool).nonzero(as_tuple=False).flatten()
            z[hidden_env_ids, :] = 0

        # pack observation buf
        P_flattened = torch.flatten(covariance_a_posteriori, start_dim=1)
        pos_scale = 0.1
        scaled_rel_state_a_posteriori = rel_state_a_posteriori.clone() 
        scaled_rel_state_a_posteriori[:, :-1] *= pos_scale # don't scale the angular dimension since it is already small

        self.obs_buf_robot = torch.cat((scaled_rel_state_a_posteriori,
                                        P_flattened
                                        ), dim=-1)

        # ------------------------------------------------------------------- #
        # =========================== Book Keeping ========================== #
        # ------------------------------------------------------------------- #
        if self.save_kf_data:
            # record real data
            if self.real_traj is None:
                self.real_traj = np.array([rel_state.cpu().numpy()])
            else:
                self.real_traj = np.append(self.real_traj, [rel_state.cpu().numpy()], axis=0)

            # record state estimate
            if self.est_traj is None:
                self.est_traj = np.array([rel_state_a_posteriori.cpu().numpy()])
            else:
                self.est_traj = np.append(self.est_traj, [rel_state_a_posteriori.cpu().numpy()], axis=0)

            # record state covariance
            if self.P_traj is None:
                self.P_traj = np.array([covariance_a_posteriori.cpu().numpy()])
            else:
                self.P_traj = np.append(self.P_traj, [covariance_a_posteriori.cpu().numpy()], axis=0)

            # record measurements
            if self.z_traj is None:
                self.z_traj = np.array([z.cpu().numpy()])
            else:
                self.z_traj = np.append(self.z_traj, [z.cpu().numpy()], axis=0)

            if self.data_save_tstep % self.data_save_interval == 0: 
                print("Saving state estimation trajectory at tstep ", self.data_save_tstep, "...")
                now = datetime.now()
                dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
                filename = "kf_data_" + dt_string + "_" + str(self.data_save_tstep) + ".pickle"
                data_dict = {"real_traj": self.real_traj,
                                "est_traj": self.est_traj, 
                                "z_traj": self.z_traj, 
                                "P_traj": self.P_traj, 
                                "kf": self.kf}
                with open(filename, 'wb') as handle:
                    pickle.dump(data_dict, handle)

            # advance timestep
            self.data_save_tstep += 1
        # ------------------------------------------------------------------- # 

    def expand_with_zeros_at_dim(self, tensor_in, dim):
        """Expands the tensor_in by putting zeros at the old "dim" location.
        """
        tensor_out = torch.cat((tensor_in[:, :dim],
                                torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False),
                                tensor_in[:, dim:]
                                ), dim=-1)
        return tensor_out

    def compute_observations_state_hist_robot(self, limited_fov=False):
        """ Computes observations of the robot with a state history (instead of estimator)
                    obs_buf is laid out as:

                   [s^t, s^t-1, s^t-2, s^t-3,
                    v^t, v^t-1, v^t-2, v^t-3]

                   where s^i = (px^i, py^i, pz^i) is the relative position (of size 3) at timestep i
                         v^i = is "visible" boolean which tracks if the agent was visible in FOV (of size 1)
         """
        # from agent's POV, get its sensing
        sense_rel_pos, _, visible_bool = self.robot_sense_agent_simpler(limited_fov=limited_fov)

        old_sense_rel_pos = self.obs_buf_robot[:,
                            self.pos_idxs_robot[:-3]].clone()  # remove the oldest relative position observation
        old_visible_bool = self.obs_buf_robot[:,
                           self.visible_idxs_robot[:-1]].clone()  # remove the corresponding oldest visible bool

        # if robot is visible, then we need to scale; otherwise its an old measurement so no need
        # obs_pos_scale = 0.1
        # visible_env_ids = torch.any(visible_bool, dim=1).nonzero(as_tuple=False).flatten()
        # sense_rel_pos[visible_env_ids, :] *= obs_pos_scale

        # a new observation has the form: (rel_opponent_pos, rel_opponent_heading, detected_bool, visible_bool)
        self.obs_buf_robot = torch.cat((sense_rel_pos,
                                        old_sense_rel_pos,
                                        visible_bool.long(),
                                        old_visible_bool
                                        ), dim=-1)

    def compute_observations_limited_FOV_robot(self):
        """ Computes observations of the robot with partial observability.
            obs_buf is laid out as:

           [s^t, s^t-1, s^t-2, s^t-3,
            sin(h^t_g), cos(h^t_g), sin(h^t-1_g), cos(h^t-1_g), sin(h^t-2_g), cos(h^t-2_g), sin(h^t-3_g), cos(h^t-3_g),
            d^t, d^t-1, d^t-2, d^t-3,
            v^t, v^t-1, v^t-2, v^t-3]

           where s^i = (px^i, py^i, pz^i) is the relative position (of size 3) at timestep i
                 h^i_g = is the *global* relative yaw angle between robot yaw and COM of agent (of size 1)
                 d^i = is "detected" boolean which tracks if the agent was detected once (of size 1)
                 v^i = is "visible" boolean which tracks if the agent was visible in FOV (of size 1)
        """
        # from agent's POV, get its sensing
        sense_rel_pos, sense_rel_yaw_global, _, detected_bool, visible_bool = self.robot_sense_agent(limited_fov=True)

        old_sense_rel_pos = self.obs_buf_robot[:, self.pos_idxs_robot[:-3]].clone()  # remove the oldest relative position observation
        old_sense_rel_yaw_global = self.obs_buf_robot[:, self.ang_global_idxs_robot[:-2]].clone()  # removes the oldest global relative heading observation
        # old_detect_bool = self.obs_buf_robot[:, self.detect_idxs_robot[:-1]].clone()  # remove the corresponding oldest detected bool
        old_visible_bool = self.obs_buf_robot[:, self.visible_idxs_robot[:-1]].clone()  # remove the corresponding oldest visible bool

        # if robot is visible, then we need to scale; otherwise its an old measurement so no need
        obs_pos_scale = 0.1
        obs_yaw_global_scale = 1.0
        visible_env_ids = torch.any(visible_bool, dim=1).nonzero(as_tuple=False).flatten()
        sense_rel_pos[visible_env_ids, :] *= obs_pos_scale
        sense_rel_yaw_global[visible_env_ids, :] *= obs_yaw_global_scale

        # print("OLD self.obs_buf_robot: ", self.obs_buf_robot)
        # print("old_sense_rel_pos: ", old_sense_rel_pos)
        # print("sense_rel_pos: ", sense_rel_pos)
        # print("old_sense_rel_yaw_global: ", old_sense_rel_yaw_global)
        # print("sense_rel_yaw_global: ", sense_rel_yaw_global)
        # print("old_detect_bool: ", old_detect_bool)
        # print("detected_bool: ", detected_bool)
        # print("old_visible_bool: ", old_visible_bool)
        # print("visible_bool: ", visible_bool)

        # a new observation has the form: (rel_opponent_pos, rel_opponent_heading, detected_bool, visible_bool)
        self.obs_buf_robot = torch.cat((sense_rel_pos,
                                        old_sense_rel_pos,
                                        sense_rel_yaw_global,
                                        old_sense_rel_yaw_global,
                                        # detected_bool.long(),
                                        # old_detect_bool,
                                        visible_bool.long(),
                                        old_visible_bool
                                        ), dim=-1)

        # print("[DecHighLevelGame] in compute_observations_robot obs_buf_robot: ")
        # print("            pos_hist: ", self.obs_buf_robot[:, self.pos_idxs_robot])
        # print("            ang_global_hist: ", self.obs_buf_robot[:, self.ang_global_idxs_robot])
        # print("            detected_bool: ", self.obs_buf_robot[:, self.detect_idxs_robot])
        # print("            visible_bool: ", self.obs_buf_robot[:, self.visible_idxs_robot])

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

        self.obs_buf_agent = torch.cat((rel_robot_pos_xyz * 0.1,
                                      angle_btwn_agents_global
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
        # self.compute_observations_agent()
        return self.obs_buf_agent

    def get_observations_robot(self):
        # self.compute_observations_robot()
        return self.obs_buf_robot

    def get_privileged_observations_agent(self):
        return self.privileged_obs_buf_agent

    def get_privileged_observations_robot(self):
        return self.privileged_obs_buf_robot

    def robot_sense_agent_simpler(self, limited_fov):
        """
        Args:
            limited_fov (bool): true if limited fov, false if otherwise

        Returns: sensing information the POV of the robot. Returns 3 values:

            sense_rel_pos (torch.Tensor): [num_envs * 3] if is visible, contains the relative agent xyz-position;
                                                if not visible, copies the last relative agent xyz-position
            sense_rel_yaw_global (torch.Tensor): [num_envs * 1] if visible, contains global_yaw
                                                if not visible, copies the last global yaw
            visible_bool (torch.Tensor): [num_envs * 1] boolean if the agent has is currently visible

        TODO: I treat the FOV the same for horiz, vert, etc. For realsense camera:
                (HorizFOV)  = 64 degrees (~1.20428 rad)
                (DiagFOV)   = 72 degrees
                (VertFOV)   = 41 degrees
        """
        half_fov = self.robot_full_fov /2.

        # rel_agent_pos_xy = self.agent_pos[:, :2] - self.robot_states[:, :2]
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        # robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        # _, _, robot_yaw = get_euler_xyz(robot_base_quat)
        # robot_yaw = wrap_to_pi(robot_yaw)

        # relative yaw between the agent's heading and the robot's heading (global)
        # rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        # rel_yaw_global = rel_yaw_global.unsqueeze(-1)
        # rel_yaw_global = wrap_to_pi(rel_yaw_global)

        rel_yaw_global = self.get_rel_yaw_global_robot()

        # pack the final sensing measurements
        sense_rel_pos_xyz = rel_agent_pos_xyz.clone()
        # sense_rel_pos_xy = rel_agent_pos_xy.clone()
        sense_rel_yaw_global = torch.cat((torch.cos(rel_yaw_global), torch.sin(rel_yaw_global)), dim=-1)

        # find environments where robot is visible
        fov_bool = torch.any(torch.abs(rel_yaw_global) <= half_fov, dim=1)
        visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()
        hidden_env_ids = (fov_bool == 0).nonzero(as_tuple=False).flatten()

        # mark envs where we robot was visible
        visible_bool = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device, requires_grad=False)
        visible_bool[visible_env_ids, :] = 1

        if limited_fov:
            # if we didn't sense the robot now, copy over the agent's previous sensed position as our new "measurement"
            sense_rel_pos_xyz[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.pos_idxs_robot[:3]].clone()
            # sense_rel_pos_xy[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.pos_idxs_robot[:2]].clone()
            # sense_rel_yaw_global[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.ang_global_idxs_robot[:2]].clone()

        # print("===================================")
        # print("agent_pos: ", self.agent_pos[:, :3])
        # print("robot_pos: ", self.robot_states[:, :3])
        # print("sense_rel_pos_xy: ", sense_rel_pos_xy)
        # print("rel_yaw_global: ", rel_yaw_global)
        # print("cos(rel_yaw_global) and sin(rel_yaw_global): ", sense_rel_yaw_global)
        # # # print("detected_buf_robot: ", self.detected_buf_robot)
        # print("visible_bool: ", visible_bool)
        # print("===================================")

        return sense_rel_pos_xyz, None, visible_bool

    def robot_sense_agent(self, limited_fov):
        """
        Args:
            limited_fov (bool): true if limited field of view, false if otherwise

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
        half_fov = self.robot_full_fov /2.

        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        _, _, robot_yaw = get_euler_xyz(robot_base_quat)
        robot_yaw = wrap_to_pi(robot_yaw)

        # # relative yaw between the two agents (global)
        # rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        # rel_yaw_global = rel_yaw_global.unsqueeze(-1)
        # rel_yaw_global = wrap_to_pi(rel_yaw_global)
        rel_yaw_global = self.get_rel_yaw_global_robot()

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

        if limited_fov:
            # if we didn't sense the robot now, copy over the agent's previous sensed position as our new "measurement"
            sense_rel_pos[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.pos_idxs_robot[:3]].clone()
            sense_rel_yaw_global[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.ang_global_idxs_robot[:2]].clone()
            # sense_rel_yaw_local[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.ang_local_idxs[:2]].clone()

            # dist_outside_fov = torch.norm(wrap_to_pi(rel_yaw_global[hidden_env_ids] - half_fov), dim=-1)
            # zero_mean_pos = torch.zeros(len(dist_outside_fov), device=self.device, requires_grad=False)
            # zero_mean_ang = torch.zeros(len(dist_outside_fov), device=self.device, requires_grad=False)
            # noise_pos = torch.normal(mean=zero_mean_pos, std=dist_outside_fov).unsqueeze(-1)
            # noise_pos *= 0.1
            # noise_ang = torch.normal(mean=zero_mean_ang, std=dist_outside_fov).unsqueeze(-1)
            # noise_ang *= 0.01 # scale down the noise in the angular space

            # print("real real pos: ", sense_rel_pos[hidden_env_ids, :])
            # sense_rel_pos[hidden_env_ids, :] = torch.add(sense_rel_pos[hidden_env_ids, :], noise_pos)
            # print("*noisy* real pos: ", sense_rel_pos[hidden_env_ids, :])
            # sense_rel_yaw_global[hidden_env_ids, :] = torch.add(sense_rel_yaw_global[hidden_env_ids, :], noise_ang)  # TODO: should add this to angle directly, not sin/cos
            # sense_rel_yaw_global[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.ang_global_idxs_robot[:2]].clone()

        # print("===================================")
        # print("agent_pos: ", self.agent_pos[:, :3])
        # print("robot_pos: ", self.robot_states[:, :3])
        # print("sense_rel_pos: ", sense_rel_pos)
        # print("rel_yaw_global: ", rel_yaw_global)
        # print("cos(rel_yaw_global) and sin(rel_yaw_global): ", sense_rel_yaw_global)
        # # # print("detected_buf_robot: ", self.detected_buf_robot)
        # print("visible_bool: ", visible_bool)
        # print("===================================")

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
        robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        robot_forward = quat_apply_yaw(robot_base_quat, self.ll_env.forward_vec)
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

    def update_training_iter_num(self, iter):
        """Updates the internal variable keeping track of training iteration"""
        self.training_iter_num = iter

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

    def _update_robot_sensing_curriculum(self, env_ids):
        """ Implements the curriculum for the robot's FOV.

        Args:
            env_ids (List[int]): ids of environments being reset
        """

        if self.training_iter_num == self.curriculum_target_iters[self.fov_curr_idx]:
            self.fov_curr_idx += 1
            self.fov_curr_idx = min(self.fov_curr_idx, len(self.cfg.robot_sensing.fov_levels)-1)
            self.robot_full_fov[env_ids] = self.cfg.robot_sensing.fov_levels[self.fov_curr_idx]
            print("[DecHighLevelGame] in update_robot_sensing_curriculum():")
            print("                   training iter num: ", self.training_iter_num)
            print("                   robot FOV is: ", self.robot_full_fov[0])

    def _update_prey_curriculum(self, env_ids):
        """ Implements the curriculum for the prey initial position

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if self.training_iter_num == self.curriculum_target_iters[self.prey_curr_idx]:
            self.prey_curr_idx += 1 
            self.prey_curr_idx = min(self.prey_curr_idx, len(self.cfg.robot_sensing.prey_angs)-1)
            max_ang = self.cfg.robot_sensing.prey_angs[self.prey_curr_idx]
            min_ang = -max_ang
            rand_angle = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False).uniform_(min_ang, max_ang)
            rand_radius = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False).uniform_(self.min_rad, self.max_rad)
            self.agent_offset_xyz = torch.cat((rand_radius * torch.cos(rand_angle),
                                               rand_radius * torch.sin(rand_angle),
                                               torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
                                               ), dim=-1)
            self.ll_env.agent_offset_xyz = self.agent_offset_xyz

            print("[DecHighLevelGame] in _update_prey_curriculum():")
            print("                   training iter num: ", self.training_iter_num)
            print("                   prey ang is: ", max_ang)

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
        rew = torch.square(torch.norm(self.agent_pos[:, :2] - self.robot_states[:, :2], p=2, dim=-1))
        return rew

    def _reward_pursuit(self):
        """Reward for pursuing"""
        rew = torch.square(torch.norm(self.agent_pos[:, :2] - self.robot_states[:, :2], p=2, dim=-1))
        return rew

    def _reward_robot_foveation(self):
        """Reward for the robot facing the agemt"""
        rel_yaw_global = self.get_rel_yaw_global_robot()
        rel_yaw_global = rel_yaw_global.squeeze(-1)

        # Exponential-type reward
        # rew = torch.exp(-torch.abs(rel_yaw_global))

        # "Relu"-type reward
        offset = np.pi / 3
        max_rew_val = 0.9

        slope = 1.0 #0.45
        diff_left = slope * rel_yaw_global + offset
        diff_right = slope * rel_yaw_global - offset

        # relu_left = torch.clamp(diff_left, min=0)   # max(0, diff_left)
        # relu_right = -torch.clamp(diff_right, max=0) # -min(0, diff_right)

        relu_left = diff_left  # max(0, diff_left)
        relu_right = -diff_right # -min(0, diff_right)

        val = torch.zeros_like(relu_left)
        val[rel_yaw_global > 0] = relu_right[rel_yaw_global > 0]
        val[rel_yaw_global <= 0] = relu_left[rel_yaw_global <= 0]
        rew = torch.clamp(val, max=max_rew_val) # min(min(a,b), max_rew_val)

        # ------------------------------------------------------------------- #
        # =========================== Book Keeping ========================== #
        # ------------------------------------------------------------------- #
        if self.save_rew_data:
            if self.fov_reward is None:
                self.fov_reward = rew.unsqueeze(0)
                self.fov_rel_yaw = rel_yaw_global.unsqueeze(0)
                self.ll_env_command_ang_vel = self.ll_env.commands[:, 2].unsqueeze(0)
            else:
                self.fov_reward = torch.cat((self.fov_reward, rew.unsqueeze(0)), dim=0)
                self.fov_rel_yaw = torch.cat((self.fov_rel_yaw, rel_yaw_global.unsqueeze(0)), dim=0)
                self.ll_env_command_ang_vel = torch.cat((self.ll_env_command_ang_vel, self.ll_env.commands[:, 2].unsqueeze(0)), dim=0)

            if self.rew_debug_tstep % self.rew_debug_interval == 0:
                print("[Foveation] Saving reward debugging information at ", self.rew_debug_tstep, "...")
                now = datetime.now()
                dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
                filename = "foveation_rew_debug_" + dt_string + "_" + str(self.rew_debug_tstep) + ".pickle"
                data_dict = {"fov_reward": self.fov_reward.cpu().numpy(),
                             "fov_rel_yaw": self.fov_rel_yaw.cpu().numpy(),
                             "ll_env_command_ang_vel": self.ll_env_command_ang_vel.cpu().numpy()}
                with open(filename, 'wb') as handle:
                    pickle.dump(data_dict, handle)

            self.rew_debug_tstep += 1

        # print("foveation reward:", rew)
        return rew

    def _reward_robot_ang_vel(self):
        """Reward for the robot's angular velocity"""
        rew = torch.norm(self.ll_env.commands[:, 2], p=2, dim=-1)
        return rew

    def _reward_path_progress(self):
        """Reward for progress along the path that connects the initial robot state to the agent state.
        r(t) = proj(s(t)) - proj(s(t-1))
        """
        curr_robot_pos = self.robot_states[:, :3]
        robot_start_pos = self.ll_env.env_origins
        robot_goal_pos = self.agent_pos[:, :3] # TODO: NOTE THIS ASSUMES THE AGENT IS STATIC

        curr_progress_to_agent = self.proj_state_to_path(curr_robot_pos, robot_start_pos, robot_goal_pos)
        last_progress_to_agent = self.proj_state_to_path(self.last_robot_pos, robot_start_pos, robot_goal_pos)

        # if self.save_rew_data:
        #     if self.r_start is None:
        #         self.r_start = robot_start_pos.unsqueeze(0)
        #         self.r_goal = robot_goal_pos.unsqueeze(0)
        #         self.r_curr = curr_robot_pos.unsqueeze(0)
        #         self.r_curr_proj = curr_progress_to_agent.unsqueeze(0)
        #         self.r_last = self.last_robot_pos.unsqueeze(0)
        #         self.r_last_proj = last_progress_to_agent.unsqueeze(0)
        #     else:
        #         self.r_start = torch.cat((self.r_start, robot_start_pos.unsqueeze(0)), dim=0)
        #         self.r_goal = torch.cat((self.r_goal, robot_goal_pos.unsqueeze(0)), dim=0)
        #         self.r_curr = torch.cat((self.r_curr, curr_robot_pos.unsqueeze(0)), dim=0)
        #         self.r_curr_proj = torch.cat((self.r_curr_proj, curr_progress_to_agent.unsqueeze(0)), dim=0)
        #         self.r_last = torch.cat((self.r_last, self.last_robot_pos.unsqueeze(0)), dim=0)
        #         self.r_last_proj = torch.cat((self.r_last_proj, last_progress_to_agent.unsqueeze(0)), dim=0)

        #     if self.rew_debug_tstep % self.rew_debug_interval == 0:
        #         print("Saving reward debugging information at ", self.rew_debug_tstep, "...")
        #         now = datetime.now()
        #         dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
        #         filename = "rew_debug_" + dt_string + "_" + str(self.rew_debug_tstep) + ".pickle"
        #         data_dict = {"r_start": self.r_start.cpu().numpy(),
        #                      "r_goal": self.r_goal.cpu().numpy(),
        #                      "r_curr": self.r_curr.cpu().numpy(),
        #                      "r_curr_proj": self.r_curr_proj.cpu().numpy(),
        #                      "r_last": self.r_last.cpu().numpy(),
        #                      "r_last_proj": self.r_last_proj.cpu().numpy()}
        #         with open(filename, 'wb') as handle:
        #             pickle.dump(data_dict, handle)

        #     self.rew_debug_tstep += 1

        return curr_progress_to_agent - last_progress_to_agent

    def proj_state_to_path(self, curr_pos, start, goal):
        """Projects point curr_pos onto the line formed by start and goal points"""
        gs = goal - start
        cs = curr_pos - start
        gs_norm = torch.norm(gs, dim=-1)
        # gs_norm = torch.sum(gs * gs, dim=-1)
        progress = torch.sum(cs * gs, dim=-1) / gs_norm
        return progress