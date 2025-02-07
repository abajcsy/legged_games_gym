
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

class HighLevelGame():
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
        print("IN HIGH LEVEL GAME...")
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
        print("--> Preparing low level environment...")
        self.ll_env, _ = task_registry.make_env(name="low_level_game", args=None, env_cfg=ll_env_cfg)

        # load low-level policy
        print("--> Loading low level policy... for: ", ll_train_cfg.runner.experiment_name)
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
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        # self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.obs_buf = self.MAX_REL_POS * torch.ones(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.curr_episode_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # TODO: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        print("--> Initializing buffers...")
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, command):
        """ Apply high-level command, simulate, and call self.post_physics_step()

        Args:
            command (torch.Tensor): Tensor of shape (num_envs, 4) for the high-level command

        Returns:
            obs_buf, priviledged_obs_buf, rew_buf, reset_buf, extras
        """

        # in the step function we hold the high-level command constant
        # high-level control is "hl_control_decimation" times slower than the low-level controller
        # hl_control_decimation = 3  # running @ 12.5 Hz
        hl_control_decimation = 1  # running @ 50 Hz

        # clip the prey's commands
        command[:, 0] = torch.clip(command[:, 0], min=self.command_ranges["lin_vel_x"][0], max=self.command_ranges["lin_vel_x"][1])
        command[:, 1] = torch.clip(command[:, 1], min=self.command_ranges["lin_vel_y"][0], max=self.command_ranges["lin_vel_y"][1])
        if self.cfg.commands.heading_command:
            command[:, 2] = wrap_to_pi(command[:, 2])

        # clip the predator's commands
        command[:, 4] = torch.clip(command[:, 4], min=self.command_ranges["predator_lin_vel_x"][0], max=self.command_ranges["predator_lin_vel_x"][1])
        command[:, 5] = torch.clip(command[:, 5], min=self.command_ranges["predator_lin_vel_y"][0], max=self.command_ranges["predator_lin_vel_y"][1])

        # high-level control loop
        for _ in range(hl_control_decimation):

            self.ll_env.commands = command[:, :4] # the first four commands are for the prey

            # get the low-level actions as a function of obs
            ll_obs = self.ll_env.get_observations()
            actions = self.ll_policy(ll_obs.detach())

            # forward simulate the low-level actions
            ll_obs, _, ll_rews, ll_dones, ll_infos = self.ll_env.step(actions.detach())
            self.curr_episode_step += 1

            # simulate the predator action too
            print("observations: ", self.obs_buf)
            print("prey commands: ", command[:, 0:4])
            print("predator commands: ", command[:, 4:])
            self.step_predator_single_integrator(command=command[:, 4:])

            # update the agent states after the forward simulation step
            self._update_agent_states()

            # compute the high-level reward as a function of the low-level reward
            self.compute_reward(ll_rews) # TODO: when hl_ctrl_decimation > 1, this should be outside the for-loop

            # reset predator-prey if they are within the capture distance or if the low-level policy detected failure
            pred_prey_dist = torch.norm(self.prey_states[:, :2] - self.predator_pos[:, :2], dim=1)
            hl_dones = pred_prey_dist < self.capture_dist

            hl_env_ids = hl_dones.nonzero(as_tuple=False).flatten()
            ll_env_ids = ll_dones.nonzero(as_tuple=False).flatten()

            all_env_ids = hl_env_ids

            if self.cfg.env.env_radius is not None:
                # check if the predator and prey have left the restricted environment zone.
                # NOTE: need to account for the unique env_origins of each environment
                prey_env_dones = torch.norm(self.prey_states[:, :2] - self.ll_env.env_origins[:, :2], dim=1) > self.cfg.env.env_radius
                predator_env_dones = torch.norm(self.predator_pos[:, :2] - self.ll_env.env_origins[:, :2], dim=1) > self.cfg.env.env_radius

                # print(self.prey_states[:, :2])

                # if either predator or prey is outside the env_radius, reset
                env_dist_dones = torch.logical_or(prey_env_dones, predator_env_dones)
                env_dist_ids = env_dist_dones.nonzero(as_tuple=False).flatten()
                all_env_ids = torch.cat((all_env_ids, env_dist_ids), dim=-1)

                # get the unique indicies of environments to be reset
                all_env_ids = torch.unique(all_env_ids)

            # print("     ll_env_ids: ", ll_env_ids)
            # print("     hl_env_ids: ", hl_env_ids)
            # print("     all_env_ids: ", all_env_ids)
            # print("     env_ids: ", env_ids)

            # add in all the low-level envs that were reset, and construct the dones vector
            all_env_ids = torch.cat((all_env_ids, ll_env_ids), dim=-1)
            env_ids = torch.unique(all_env_ids)

            # NOTE: only need to reset the high-level envs here,
            # because ll_env.step() resets based on the low-level reset criteria
            self.reset_idx(env_ids)

            all_dones = torch.zeros_like(ll_dones, device=self.device, requires_grad=False)
            all_dones[env_ids] = True
            self.reset_buf = all_dones

            # compute the high-level observation: relative predator-prey state
            self.compute_observations()

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def step_predator_dubins_car(self):
        """
        Steps predator modeled as a dubins' car like vehicle:
            x' = x + dt * u1 * cos(theta)
            y' = y + dt * u1 * sin(theta)
            z' = z          (assume z-dim is constant)
            theta' = theta + dt * u2
            u = [u1, u2] where u1 is linear velocity and u2 is angular velocity
        """

        command_lin_vel, command_ang_vel = self.full_obs_predator(dyn_type='dubins')

        # TODO: predator gets simulated at the same Hz as the low-level controller!
        for _ in range(self.ll_env.cfg.control.decimation):
            self.predator_pos[:, 0] += self.ll_env.cfg.sim.dt * command_lin_vel * torch.cos(self.predator_pos[:, 3])
            self.predator_pos[:, 1] += self.ll_env.cfg.sim.dt * command_lin_vel * torch.sin(self.predator_pos[:, 3])
            self.predator_pos[:, 3] += self.ll_env.cfg.sim.dt * command_ang_vel

        # update the simulator state for the predator!
        self.ll_env.root_states[self.ll_env.predator_indices, :] = self.predator_pos
        self.ll_env.gym.set_actor_root_state_tensor(self.ll_env.sim, gymtorch.unwrap_tensor(self.ll_env.root_states))

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

    def full_obs_predator(self, dyn_type):
        """
        Computes the predator's "policy" assume it gets full observability of the prey at all times.
        dyn_type: supported types are 'dubins' or 'integrator'

        Returns:
            command_lin_vel_x, command_lin_vel_y actions for predator to take.
        """
        dxy = self.ll_env.root_states[self.ll_env.prey_indices, :2] - self.predator_pos[:, :2]
        distxy = torch.norm(dxy, dim=1)

        yaw = torch.atan2(dxy[:, 1], dxy[:, 0])
        max_lin_vel = 2

        command_u1 = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        command_u2 = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        if dyn_type is 'integrator':

            dxy *= 2 # TODO: hack --makes the predator more aggressive

            # simulate the predator "losing steam" and getting slower over time.
            # note:  curr_episode_step is of size [num_envs x 1]
            alpha = (self.max_episode_length - self.curr_episode_step) / self.max_episode_length
            curr_vel_limit = 0.01 * (1 - alpha) + max_lin_vel * alpha

            command_u1 = torch.clamp(dxy[:, 0], min=-curr_vel_limit, max=curr_vel_limit) # lin_vel_x
            command_u2 = torch.clamp(dxy[:, 1], min=-curr_vel_limit, max=curr_vel_limit) # lin_vel_y
        elif dyn_type is 'dubins':
            # TODO THIS IS A HACK!
            command_u1 = max_lin_vel
            command_u2 = np.pi/8.
        else:
            print("ERROR: dynamics type <", dyn_type, "> not supported for predator!")
            return 0., 0.

        return command_u1, command_u2

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
        self.ll_env._reset_root_states(env_ids)

        # update the state variables
        self._update_agent_states()

        # reset buffers
        self.obs_buf[env_ids, 0:12] = self.MAX_REL_POS      # reset the prey's rel_predator_pos to max relative position
        self.obs_buf[env_ids, 12:16] = 0                    # reset the prey's sense booleans to zero and the episode tstep to zero
        self.obs_buf[env_ids, 16:] = -self.MAX_REL_POS      # reset the predator's rel_prey_pos to -max relative position
        self.episode_length_buf[env_ids] = 0
        self.curr_episode_step[env_ids] = 0

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def compute_reward(self, ll_rews):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward

            Args: ll_rews (torch.Tensor) of size num_envs containing low-level reward per environment
        """
        ll_rew_weight = 2.0
        self.rew_buf = ll_rew_weight * ll_rews # sum together the low-level reward and the high-level reward
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        # print("high level rewards + low-level rewards: ", self.rew_buf)

    def compute_observations(self):
        """ Computes observations
        """
        # # add relative predator state
        # rel_predator_pos = self.predator_pos - self.prey_states[:, :3]
        # self.obs_buf = rel_predator_pos

        # from prey's POV, get its sensing
        rel_predator_pos, sense_bool = self.sense_predator()

        # from predator's POV, get the relative position to the prey
        rel_prey_pos = self.prey_states[:, :3] - self.predator_pos

        # obs_buf is laid out as:
        #   [s1, s2, s3, s4, bool1, bool2, bool3, bool4]
        #   where s1 = (px1, py1, pz1) is the relative position
        old_rel_pred_pos = self.obs_buf[:, 3:12].clone()    #  remove the oldest relative position observation
        old_sense_bool = self.obs_buf[:, 13:16].clone()     # [num_envs x hist_len-1] remove the oldest sense bool

        # self.obs_buf = torch.cat((old_rel_pred_pos,
        #                           rel_predator_pos,
        #                           old_sense_bool,
        #                           sense_bool.long(),
        #                           self.curr_episode_step.unsqueeze(-1)), dim=-1)

        self.obs_buf = torch.cat((old_rel_pred_pos,
                                  rel_predator_pos,
                                  old_sense_bool,
                                  sense_bool.long(),
                                  rel_prey_pos), dim=-1)

    def get_observations(self):
        self.compute_observations()
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def sense_predator(self):
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

        # if we didn't sense the predator now, copy over the previous sensed position as our new "measurement"
        occluded_env_ids = (visible_bool == 0).nonzero(as_tuple=False).flatten()
        sense_rel_predator_pos[occluded_env_ids, :] = self.obs_buf[occluded_env_ids, 9:12].clone()

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

        # print("in sense_predator: ")
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

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.ll_env.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def _parse_cfg(self, cfg):
        # self.dt = self.cfg.control.decimation * self.sim_params.dt
        # self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
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
