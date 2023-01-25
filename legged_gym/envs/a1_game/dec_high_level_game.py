
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

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

        # setup the capture distance between predator-prey
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

        self.num_envs = cfg.env.num_envs

        # prey policy info
        self.num_obs_prey = cfg.env.num_observations_prey
        self.num_privileged_obs_prey = cfg.env.num_privileged_obs_prey
        self.num_actions_prey = cfg.env.num_actions_prey

        # predator policy info
        self.num_obs_pred = cfg.env.num_observations_predator
        self.num_privileged_obs_pred = cfg.env.num_privileged_obs_predator
        self.num_actions_pred = cfg.env.num_actions_predator

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate prey buffers
        self.obs_buf_prey = self.MAX_REL_POS * torch.ones(self.num_envs, self.num_obs_prey, device=self.device, dtype=torch.float)
        self.obs_buf_prey[:, 12:16] = 0 # reset the prey's sense booleans to zero
        self.rew_buf_prey = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # allocate predator buffers
        self.obs_buf_pred = self.MAX_REL_POS * torch.ones(self.num_envs, self.num_obs_pred, device=self.device, dtype=torch.float)
        self.rew_buf_pred = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.curr_episode_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        if self.num_privileged_obs_pred is not None:
            self.privileged_obs_buf_pred = torch.zeros(self.num_envs, self.num_privileged_obs_pred, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf_pred = None
            # self.num_privileged_obs = self.num_obs

        if self.num_privileged_obs_prey is not None:
            self.privileged_obs_buf_prey = torch.zeros(self.num_envs, self.num_privileged_obs_prey, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf_prey = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # TODO: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        print("[DecHighLevelGame] initializing buffers...")
        self._init_buffers()
        print("[DecHighLevelGame] preparing PREDATOR reward functions...")
        self._prepare_reward_function_pred()
        print("[DecHighLevelGame] preparing PREY reward functions...")
        self._prepare_reward_function_prey()
        print("[DecHighLevelGame] done with initializing!")
        self.init_done = True

    def step(self, command_pred, command_prey):
        """Applies a high-level command to a particular agent, simulates, and calls self.post_physics_step()

        Args:
            command_pred (torch.Tensor): Tensor of shape (num_envs, num_actions_pred) for the high-level command
            command_prey (torch.Tensor): Tensor of shape (num_envs, num_actions_prey) for the high-level command

        Returns:
            obs_buf_pred, obs_buf_prey, priviledged_obs_buf_pred, priviledged_obs_buf_prey, ...
            rew_buf_pred, rew_buf_prey, reset_buf, extras
        """

        # clip the prey's commands
        command_prey[:, 0] = torch.clip(command_prey[:, 0], min=self.command_ranges["lin_vel_x"][0], max=self.command_ranges["lin_vel_x"][1])
        command_prey[:, 1] = torch.clip(command_prey[:, 1], min=self.command_ranges["lin_vel_y"][0], max=self.command_ranges["lin_vel_y"][1])
        if self.cfg.commands.heading_command:
            command_prey[:, 2] = wrap_to_pi(command_prey[:, 2])

        # clip the predator's commands
        command_pred[:, 0] = torch.clip(command_pred[:, 0], min=self.command_ranges["predator_lin_vel_x"][0], max=self.command_ranges["predator_lin_vel_x"][1])
        command_pred[:, 1] = torch.clip(command_pred[:, 1], min=self.command_ranges["predator_lin_vel_y"][0], max=self.command_ranges["predator_lin_vel_y"][1])

        # print("PREY CMD: ", command_prey)
        # print("PREDATOR CMD: ", command_pred)

        # update the low-level simulator command since it deals with the prey robot
        self.ll_env.commands = command_prey 

        # get the low-level actions as a function of low-level obs
        ll_prey_obs = self.ll_env.get_observations()
        ll_prey_actions = self.ll_policy(ll_prey_obs.detach())

        # forward simulate the low-level actions
        ll_obs, _, ll_rews, ll_dones, ll_infos = self.ll_env.step(ll_prey_actions.detach())

        # forward simulate the predator action too
        self.step_predator_single_integrator(command=command_pred)

        # take care of terminations, compute observations, rewards, and dones
        self.post_physics_step(ll_rews, ll_dones)

        return self.obs_buf_pred, self.obs_buf_prey, self.privileged_obs_buf_pred, self.privileged_obs_buf_prey, self.rew_buf_pred, self.rew_buf_prey, self.reset_buf, self.extras

    def step_predator_single_integrator(self, command=None):
        """
        Steps predator modeled as a single integrator:
            x' = x + dt * u1
            y' = y + dt * u2
            z' = z          (assume z-dim is constant)
        """

        # predator that has full observability of the prey state
        if command is None:
            command_lin_vel_x, command_lin_vel_y = self.full_obs_predator(dyn_type='integrator')
        else:
            command_lin_vel_x = command[:, 0]
            command_lin_vel_y = command[:, 1]

        # TODO: predator gets simulated at the same Hz as the low-level controller!
        for _ in range(self.ll_env.cfg.control.decimation):
            self.predator_pos[:, 0] += self.ll_env.cfg.sim.dt * command_lin_vel_x
            self.predator_pos[:, 1] += self.ll_env.cfg.sim.dt * command_lin_vel_y

        # update the simulator state for the predator!
        self.ll_env.root_states[self.ll_env.predator_indices, :3] = self.predator_pos
        self.ll_env.gym.set_actor_root_state_tensor(self.ll_env.sim, gymtorch.unwrap_tensor(self.ll_env.root_states))

    def post_physics_step(self, ll_rews, ll_dones):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
        """
        # updates the local copies of agent state info
        self._update_agent_states()

        self.episode_length_buf += 1
        self.curr_episode_step += 1 # this is used to model the progress of "time" in the episode

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward_prey(ll_rews) # prey (robot) combines the high-level and low-level rewards
        self.compute_reward_pred()

        # NOTE: ll_env.step() already reset based on the low-level reset criteria
        self.reset_buf |= self.ll_env.reset_buf # combine the low-level dones and the high-level dones
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # compute the high-level observation for each agent
        self.compute_observations_pred()
        self.compute_observations_prey()

        print("PREY OBS: ", self.obs_buf_prey)
        print("PREDATOR OBS: ", self.obs_buf_pred)

    def check_termination(self):
        """ Check if environments need to be reset under various conditions.
        """
        # reset predator-prey if they are within the capture distance
        self.reset_buf = torch.norm(self.prey_states[:, :2] - self.predator_pos[:, :2], dim=-1) < self.capture_dist
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
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

        # resets the predator and prey states in the low-level simulator environment
        self.ll_env._reset_dofs(env_ids)
        self.ll_env._reset_root_states(env_ids)

        # update the state variables
        self._update_agent_states()

        # reset buffers
        self.obs_buf_prey[env_ids, 0:12] = self.MAX_REL_POS      # reset the prey's rel_predator_pos to max relative position
        self.obs_buf_prey[env_ids, 12:16] = 0                    # reset the prey's sense booleans to zero 
        self.obs_buf_pred[env_ids, :] = -self.MAX_REL_POS      # reset the predator's rel_prey_pos to -max relative position
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.curr_episode_step[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums_pred.keys():
            self.extras["episode"]['rew_pred_' + key] = torch.mean(self.episode_sums_pred[key][env_ids]) / self.max_episode_length_s
            self.episode_sums_pred[key][env_ids] = 0.
        for key in self.episode_sums_prey.keys():
            self.extras["episode"]['rew_prey_' + key] = torch.mean(self.episode_sums_prey[key][env_ids]) / self.max_episode_length_s
            self.episode_sums_prey[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        actions_pred = torch.zeros(self.num_envs, self.num_actions_pred, device=self.device, requires_grad=False)
        actions_prey = torch.zeros(self.num_envs, self.num_actions_prey, device=self.device, requires_grad=False)
        obs_pred, obs_prey, privileged_obs_pred, privileged_obs_prey, _, _, _, _ = self.step(actions_pred, actions_prey)
        return obs_pred, obs_prey, privileged_obs_pred, privileged_obs_prey

    def compute_reward_prey(self, ll_rews):
        """ Compute rewards for the PREY
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward

            Args: ll_rews (torch.Tensor) of size num_envs containing low-level reward per environment
        """
        ll_rew_weight = 2.0
        self.rew_buf_prey = ll_rew_weight * ll_rews # sum together the low-level reward and the high-level reward
        for i in range(len(self.reward_functions_prey)):
            name = self.reward_names_prey[i]
            rew = self.reward_functions_prey[i]() * self.reward_scales_prey[name]
            self.rew_buf_prey += rew
            self.episode_sums_prey[name] += rew
        if self.cfg.rewards_prey.only_positive_rewards:
            self.rew_buf_prey[:] = torch.clip(self.rew_buf_prey[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales_prey:
            rew = self._reward_termination() * self.reward_scales_prey["termination"]
            self.rew_buf_prey += rew
            self.episode_sums_prey["termination"] += rew
        # print("high level rewards + low-level rewards: ", self.rew_buf)

    def compute_reward_pred(self):
        """ Compute rewards for the PREDATOR
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf_pred[:] = 0.
        for i in range(len(self.reward_functions_pred)):
            name = self.reward_names_pred[i]
            rew = self.reward_functions_pred[i]() * self.reward_scales_pred[name]
            self.rew_buf_pred += rew
            self.episode_sums_pred[name] += rew
        if self.cfg.rewards_predator.only_positive_rewards:
            self.rew_buf_pred[:] = torch.clip(self.rew_buf_pred[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales_pred:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf_pred += rew
            self.episode_sums_pred["termination"] += rew
        # print("high level rewards + low-level rewards: ", self.rew_buf)

    def compute_observations_prey(self):
        """ Computes observations of the robot prey
        """

        # from prey's POV, get its sensing
        rel_predator_pos, sense_bool = self.prey_sense_predator()

        # obs_buf is laid out as:
        #   [s1, s2, s3, s4, bool1, bool2, bool3, bool4]
        #   where s1 = (px1, py1, pz1) is the relative position
        old_rel_pred_pos = self.obs_buf_prey[:, 3:12].clone()    #  remove the oldest relative position observation
        old_sense_bool = self.obs_buf_prey[:, 13:16].clone()     # [num_envs x hist_len-1] remove the oldest sense bool

        self.obs_buf_prey = torch.cat((old_rel_pred_pos,
                                  rel_predator_pos,
                                  old_sense_bool,
                                  sense_bool.long()), dim=-1)

        # print("[DecHighLevelGame] in compute_observations_prey: ", self.obs_buf_prey[0, :])

    def compute_observations_pred(self):
        """ Computes observations of the predator
        """

        # from predator's POV, get the relative position to the prey
        rel_prey_pos = self.prey_states[:, :3] - self.predator_pos

        self.obs_buf_pred = rel_prey_pos 
        # print("[DecHighLevelGame] in compute_observations_pred: ", self.obs_buf_pred[0, :])

    def get_observations_pred(self):
        self.compute_observations_pred()
        return self.obs_buf_pred

    def get_observations_prey(self):
        self.compute_observations_prey()
        return self.obs_buf_prey 

    def get_privileged_observations_pred(self):
        return self.privileged_obs_buf_pred

    def get_privileged_observations_prey(self):
        return self.privileged_obs_buf_prey

    def prey_sense_predator(self):
        """
        Returns: relative predator state if within FOV, else it returns MAX_DIST for each state component.

        TODO: I treat the FOV the same for horiz, vert, etc. For realsense camera:
                (HorizFOV)  = 64 degrees (~1.20428 rad)
                (DiagFOV)   = 72 degrees
                (VertFOV)   = 41 degrees
        """
        half_fov = 1.20428 / 2.

        # forward = quat_apply(self.ll_env.base_quat, self.ll_env.forward_vec) # direction vector of prey
        # rel_predator_pos = self.ll_env.root_states[self.ll_env.predator_indices, :3] - self.ll_env.root_states[self.ll_env.prey_indices, :3]
        rel_predator_pos = self.predator_pos - self.prey_states[:, :3] # relative position of predator w.r.t. prey
        forward = quat_apply_yaw(self.ll_env.base_quat, self.ll_env.forward_vec)

        # TODO: only use  xy dim here?
        dot_vec = torch.sum(forward * rel_predator_pos, dim=-1)
        dot_vec = dot_vec.unsqueeze(-1)
        n_forward = torch.norm(forward, p=2, dim=-1, keepdim=True)
        n_rel_predator_pos = torch.norm(rel_predator_pos, p=2, dim=-1, keepdim=True)
        denom = n_forward * n_rel_predator_pos

        # angle of predator w.r.t prey
        pred_angle = torch.acos(dot_vec / denom)
        pred_angle = wrap_to_pi(pred_angle)

        # sense_rel_predator_pos = self.MAX_DIST * torch.ones_like(rel_predator_pos, device=self.device, requires_grad=False)
        sense_rel_predator_pos = rel_predator_pos.clone()

        # see which predators are visible within the half_fov
        sense_bool = torch.any(torch.abs(pred_angle) <= half_fov, dim=1)
        visible_env_ids = sense_bool.nonzero(as_tuple=False).flatten()

        # mark all environments where we could sense the predator
        visible_bool = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        visible_bool[visible_env_ids, :] = 1

        # if we didn't sense the predator now, copy over the prey's previous sensed position as our new "measurement"
        occluded_env_ids = (visible_bool == 0).nonzero(as_tuple=False).flatten()
        sense_rel_predator_pos[occluded_env_ids, :] = self.obs_buf_prey[occluded_env_ids, 9:12].clone()

        # target_tensor = torch.tensor([8., 8., -0.1200], device=self.device, requires_grad=False)
        # print(sense_rel_predator_pos[0,0:3])
        # print(target_tensor)
        # print(torch.allclose(sense_rel_predator_pos[0,0:3], target_tensor))
        # if torch.allclose(sense_rel_predator_pos[0,0:3], target_tensor):
        #     print("WHY AM I GETTING HERE? pred_angle = ", pred_angle)
        #     import pdb; pdb.set_trace()

        # print("sense_bool: ", sense_bool)
        # print("visible_env_ids: ", visible_env_ids)
        # print("occluded_env_ids: ", occluded_env_ids)

        # if visible, record their relative position, otherwise set their relative position to MAX_DIST
        # sense_rel_predator_pos[sense_env_ids, :] = rel_predator_pos[sense_env_ids, :]

        # print("in prey_sense_predator: ")
        # if occluded_env_ids.nelement() == 0:
        #     print(" prey can see!")
        # else:
        #     print(" prey can't see predator...")
        # print("  obs_buf: ", self.obs_buf)

        return sense_rel_predator_pos, visible_bool


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

    # ------------- Callbacks --------------

    def _update_agent_states(self):
        """Goes to the low-level environment and grabs the most recent simulator states for the predator and prey."""
        self.prey_states = self.ll_env.root_states[self.ll_env.prey_indices, :]
        self.base_quat = self.prey_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.prey_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.prey_states[:, 10:13])
        self.predator_pos = self.ll_env.root_states[self.ll_env.predator_indices, :3]

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.prey_states = self.ll_env.root_states[self.ll_env.prey_indices, :]
        self.base_quat = self.prey_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.prey_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.prey_states[:, 10:13])

        # the predator is initialized at a random offset from the prey
        # init_prey_pos = self.prey_states[:, :3].detach().clone()
        # offset = 2.0
        self.init_predator_pos = self.ll_env.init_predator_pos.clone()
        # self.init_predator_pos = init_prey_pos - (torch.randn_like(init_prey_pos) - offset)
        # self.init_predator_pos[:, 2] = self.cfg.init_state.predator_pos[2] # keep the same z, only randomize xy-offset
        self.predator_pos = self.init_predator_pos

    def _prepare_reward_function_pred(self):
        """ Prepares a list of reward functions for the predator, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales_pred.keys()):
            scale = self.reward_scales_pred[key]
            if scale == 0:
                self.reward_scales_pred.pop(key)
            else:
                self.reward_scales_pred[key] *= self.ll_env.dt
        # prepare list of functions
        self.reward_functions_pred = []
        self.reward_names_pred = []
        for name, scale in self.reward_scales_pred.items():
            if name == "termination":
                continue
            self.reward_names_pred.append(name)
            name = '_reward_' + name
            print("[_prepare_reward_function_pred]: reward = ", name)
            self.reward_functions_pred.append(getattr(self, name))

        # reward episode sums
        self.episode_sums_pred = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales_pred.keys()}

    def _prepare_reward_function_prey(self):
        """ Prepares a list of reward functions for the prey, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales_prey.keys()):
            scale = self.reward_scales_prey[key]
            if scale == 0:
                self.reward_scales_prey.pop(key)
            else:
                self.reward_scales_prey[key] *= self.ll_env.dt
        # prepare list of functions
        self.reward_functions_prey = []
        self.reward_names_prey = []
        for name, scale in self.reward_scales_prey.items():
            if name == "termination":
                continue
            self.reward_names_prey.append(name)
            name = '_reward_' + name
            print("[_prepare_reward_function_prey]: reward = ", name)
            self.reward_functions_prey.append(getattr(self, name))

        # reward episode sums
        self.episode_sums_prey = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales_prey.keys()}

    def _parse_cfg(self, cfg):
        # self.dt = self.cfg.control.decimation * self.sim_params.dt
        # self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales_prey = class_to_dict(self.cfg.rewards_prey.scales)
        self.reward_scales_pred = class_to_dict(self.cfg.rewards_predator.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.ll_env.dt)

        # self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.ll_env.dt)

    # ------------ reward functions----------------
    def _reward_evasion(self):
        # [POV of prey] Rewards *maximizing* distance between predator and prey
        return torch.norm(self.predator_pos - self.prey_states[:, :3], dim=1)

    def _reward_pursuit(self):
        # [POV of predator] Rewards *minimizing* distance between predator and prey
        return -torch.norm(self.predator_pos - self.prey_states[:, :3], dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
