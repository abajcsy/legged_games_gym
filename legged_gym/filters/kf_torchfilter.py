from typing import Tuple

import torch
import torchfilter
from torchfilter import types

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import multivariate_normal

import pdb

from datetime import datetime
import os

from legged_gym.filters.filter_helpers import wrap_to_pi, plot_state_traj, plot_state_cov_mat

from legged_gym.filters.models.linear_system_models import (
    LinearDynamicsModel,
    LinearKalmanFilterMeasurementModel
)
from legged_gym.filters.models.nonlinear_system_models import (
    NonlinearDynamicsModel,
    NonlinearKalmanFilterMeasurementModel
)

class UKFTorchFilter(object):
    """
    This class is just a lightweight wrapper around the TorchFilter library's UnscentedKalmanFilter. 
    """
    def __init__(self, 
                state_dim, 
                control_dim, 
                observation_dim, 
                num_envs, 
                dt, 
                device,
                ang_dims=None,
                dyn_sys_type="linear"):
        self.dt = dt 
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.observation_dim = observation_dim
        self.num_envs = num_envs
        self.ang_dims = ang_dims
        self.device = device

        # initial value of a posteriori estimate covariance matrix
        self.P = torch.eye(self.state_dim, device=self.device, requires_grad=False)
        self.P[self.ang_dims,self.ang_dims] *= 0.1 # scale down the covariance of the angular dimension

        if dyn_sys_type == "linear" or "Linear":
            dyn_model, measure_model = self._linear_system_model()
        elif dyn_sys_type == "nonlinear" or "Nonlinear":
            dyn_model, measure_model = self._nonlinear_system_model()
        else:
            print("[ERROR: UKFTorchFilter] No such dynamical systems type as: ", dyn_sys_type)
            return -1 

        # create the Kalman Filter
        sigma_point_strategy = None
        self.filter = torchfilter.filters.UnscentedKalmanFilter(
                dynamics_model=dyn_model,
                measurement_model=measure_model,
                ang_dims=self.ang_dims,
                sigma_point_strategy=sigma_point_strategy
            )

        # self.filter = torchfilter.filters.ExtendedKalmanFilter(
        #         dynamics_model=dyn_model,
        #         measurement_model=measure_model
        #     )

        # initialize the filter with some default values.
        self.filter.initialize_beliefs(
            mean=torch.zeros(self.num_envs, self.state_dim, device=self.device, requires_grad=False),
            covariance=torch.zeros(size=(
                                    self.num_envs, 
                                    self.state_dim, 
                                    self.state_dim), device=self.device, requires_grad=False) 
                                    + self.P[None, :, :],
        )     

    def _linear_system_model(self):
        A = torch.eye(self.state_dim, device=self.device, requires_grad=False)
        B = -self.dt * torch.eye(self.state_dim, self.control_dim, device=self.device, requires_grad=False)
        C = torch.eye(self.observation_dim, self.state_dim, device=self.device, requires_grad=False)
        C_pinv = torch.pinverse(C)
        q_val = 0.01 
        r_val = 0.2

        # need to remap B matrix so (theta) is evolution is properly handled
        B[-1, -1] = self.dt

        dyn_model = LinearDynamicsModel(self.state_dim, 
                            self.control_dim, 
                            self.observation_dim,
                            A, B, q_val, self.device, ang_dims=self.ang_dims)
        measure_model = LinearKalmanFilterMeasurementModel(self.state_dim, 
                            self.observation_dim, 
                            C, r_val, self.device, ang_dims=self.ang_dims)

        return dyn_model, measure_model

    def _nonlinear_system_model(self):
        C = torch.eye(self.observation_dim, self.state_dim, device=self.device, requires_grad=False)
        q_val = 0.01 
        r_val = 0.2
        dyn_model = NonlinearDynamicsModel(self.state_dim, 
                            self.control_dim, 
                            self.observation_dim,
                            q_val, self.dt, self.device)
        measure_model = NonlinearKalmanFilterMeasurementModel(self.state_dim, 
                            self.observation_dim, 
                            C, r_val, self.device)

        return dyn_model, measure_model

    def reset_mean_cov(self, env_ids, mean=None):
        """Resets the initial estimate.
        Args:
            env_ids [list of Ints]: environment indicies to be reset
            mean [torch.Tensor]: of length [num_env_ids, num_states] containing values to reset to
        """
        # reset initial state estimate
        if mean is None:
            self.filter._belief_mean[env_ids, :] = 0.
        else:
            self.filter._belief_mean[env_ids, :] = mean

        # reset state covariance
        self.filter._belief_covariance[env_ids, :] = self.P

    # def sim_observations(self, states):
    #     """Simulates observations with measurement model.
    #     Args: 
    #         states [torch.Tensor]: of length [num_envs, num_states]
    #     """
    #     pred_observations, R_tril = self.filter.measurement_model(states=states)
    #     observations = pred_observations + (
    #                 R_tril @ torch.randn(size=(self.num_envs, self.observation_dim, 1), device=self.device, requires_grad=False)
    #                 ).squeeze(-1)

    #     # wrap the angular dimensions to -pi to pi
    #     observations[:, self.ang_dims] = wrap_to_pi(observations[:, self.ang_dims])
    #     return observations

    def sim_observations(self, states):
        """Simulates observations with measurement model.
        Args: 
            states [torch.Tensor]: of length [num_envs, num_states]

        Returns:
            z (torch.Tensor): measurement of shape [num_env_ids, num_states]
        """
        pred_observations, R_tril = self.filter.measurement_model(states=states)
        # R_tensor = R_tril.repeat(self.num_envs, 1).reshape(self.num_envs, self.state_dim, self.state_dim)

        zero_mean = np.zeros((self.num_envs, self.state_dim))
        v_tensor = self.sample_batch_mvn(zero_mean, R_tril.cpu().numpy(), self.num_envs)

        if self.ang_dims is not None:
            # if we have an angular component to state, sample it separately
            v_tensor[:, self.ang_dims] = wrap_to_pi(v_tensor[:, self.ang_dims])

        # use y = Hx + v to simulate measurement where v ~ N(0, R)
        z = pred_observations + v_tensor

        # wrap the angular dimensions to -pi to pi
        if self.ang_dims is not None:
            z[:, self.ang_dims] = wrap_to_pi(z[:, self.ang_dims])

        if z.dtype != states.dtype:
            z = z.to(states.dtype)

        return z

    def sample_batch_mvn(self, mean, cov, batch_size) -> np.ndarray:
        """
        Batch sample multivariate normal distribution.

        Arguments:

            mean (np.ndarray): expected values of shape (B, D)
            cov (np.ndarray): covariance matrices of shape (B, D, D)
            batch_size (int): additional batch shape (B)

        Returns: torch.Tensor or shape: (B, D)
                 with one samples from the multivariate normal distributions
        """
        L = np.linalg.cholesky(cov)
        X = np.random.standard_normal((batch_size, mean.shape[-1], 1))
        Y = (L @ X).reshape(batch_size, mean.shape[-1]) + mean
        Y_tensor = torch.tensor(Y, device=self.device, requires_grad=False)
        return Y_tensor

    def dynamics(self, states, commands, add_noise=False):
        """Wrapper for predict function. Handles angles."""
        next_states, Q_tril = self.filter.dynamics_model(initial_states=states, controls=commands)
        # add noise
        if add_noise:
            next_states = next_states + (
                    Q_tril @ torch.randn(size=(self.num_envs, self.state_dim, 1))
                ).squeeze(-1)
        
        next_states[:, self.ang_dims] = wrap_to_pi(next_states[:, self.ang_dims])
        return next_states, Q_tril

    def predict(self, commands_robot):
        """Wrapper for predict function. Handles angles."""
        self.filter._predict_step(controls=commands_robot)
        self.filter._belief_mean[:, self.ang_dims] = wrap_to_pi(
                                                self.filter._belief_mean[:, self.ang_dims])

    def update(self, observations, env_ids):
        """Wrapper for update function. Handles angles."""
        if len(env_ids) == 0:
            return
        self.filter._update_step_visible(observations, env_ids)
        # self.filter._update_step(observations=observations)
        self.filter._belief_mean[env_ids, self.ang_dims] = wrap_to_pi(
                                                        self.filter._belief_mean[env_ids, self.ang_dims])

    # ----- Helpers ----- #
    def _turn_and_pursue_command_robot(self, rel_state):
        command_robot = torch.zeros(self.num_envs, self.control_dim, device=self.device)
        print("rel state: ", rel_state[:, :-1])
        print("dtheta: ", rel_state[:, -1])
        eps = 0.15
        for env_idx in range(self.num_envs):
            if torch.abs(rel_state[env_idx, -1]) < eps:
                print("GOING STRAIGHT..")
                command_robot[env_idx, :-1] = torch.clip(rel_state[env_idx, :2], min=-1, max=1)[:self.control_dim-1]
            else:
                print("TURNING..")
                command_robot[env_idx, -1] = 1 #torch.clip(rel_state[env_idx, -1],  min=-1, max=1) # just turn
        return command_robot

    def _dubins_command_robot(self, rel_state):
        command_robot = torch.zeros(self.num_envs, self.control_dim, device=self.device)
        print("rel state: ", rel_state[:, :-1])
        print("dtheta: ", rel_state[:, -1])
        eps = 0.15
        for env_idx in range(self.num_envs):
            command_robot[env_idx, 0] = torch.clip(rel_state[env_idx, 0], min=-1, max=1)
            command_robot[env_idx, 1] = -1 * torch.clip(rel_state[env_idx, -1],  min=-1, max=1)
        return command_robot

def sim_linear_sys_ukf():
    """Simulate data for linear system robot."""
    state_dim = 4
    control_dim = 3
    observation_dim = 4
    num_envs = 1
    dt = 0.02
    robot_full_fov = 1.20428
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the real system
    max_s = 4
    t = np.arange(0, max_s, dt)
    print("Simulating for ", max_s, " seconds...")

    kf = UKFTorchFilter(state_dim,
                        control_dim,
                        observation_dim,
                        num_envs,
                        dt,
                        device=device,
                        ang_dims=-1,
                        dyn_sys_type="linear")

    xa0 = torch.zeros(num_envs, state_dim)
    xa0[0, 0] = 0
    xa0[0, 1] = 0
    xa0[0, 3] = 0
    # xa0[1, 0] = 1
    # xa0[1, 1] = 1
    # xa0[1, 3] = 0
    # xa0[2, 0] = 0
    # xa0[2, 1] = 0
    # xa0[2, 3] = 0
    xr0 = torch.zeros(num_envs, state_dim)
    xr0[0, 0] = 3
    xr0[0, 1] = 2
    xr0[0, 3] = 0 #np.pi / 2
    # xr0[1, 0] = -3
    # xr0[1, 1] = -2
    # xr0[1, 3] = np.pi / 4
    # xr0[2, 0] = 1
    # xr0[2, 1] = 4
    # xr0[2, 3] = -np.pi / 2

    # setup the dtheta state
    xrel0 = xa0 - xr0
    xrel0[:, -1] = torch.atan2(xrel0[:, 1], xrel0[:, 0]) - xr0[:, -1]  # fourth state is dtheta
    xrel0[:, -1] = wrap_to_pi(xrel0[:, -1])

    observations = kf.sim_observations(states=torch.tensor(xrel0))
    kf.reset_mean_cov(env_ids=torch.arange(num_envs), mean=torch.tensor(xrel0)) # TODO: here gets perfect initial state!

    print("belief mean: ", kf.filter._belief_mean)
    print("belief cov: ", kf.filter._belief_covariance)

   # create the fake ground-truth data
   #  xrel = xrel0
   #  real_traj = np.array([xrel.numpy()])
   #  for tidx in range(len(t)):
   #      commands_robot = kf._turn_and_pursue_command_robot(xrel)
   #      # action = torch.clip(xrel[:, :2], min=min_lin_vel, max=max_lin_vel)
   #      xrel, Q_tril = kf.dynamics(xrel, commands_robot)
   #      real_traj = np.append(real_traj, [xrel.numpy()], axis=0)
        
    # do the prediction and simulation loop
    xrel = xrel0
    pred_traj = np.array([kf.filter._belief_mean.numpy()])
    est_traj = np.array([kf.filter._belief_mean.numpy()])
    real_traj = np.array([xrel.numpy()])
    z_traj = np.zeros(0)
    P_traj = np.array([kf.filter._belief_covariance.numpy()])
    for tidx in range(len(t)):
        print("predicting...")
        # import pdb; pdb.set_trace()
        commands_robot = kf._turn_and_pursue_command_robot(kf.filter._belief_mean)
        commands_robot[:, 0] = 1
        commands_robot[:, 1] = -1
        commands_robot[:, 2] = 1

        xrel, Q_tril = kf.dynamics(xrel, commands_robot, add_noise=False)
        real_traj = np.append(real_traj, [xrel.numpy()], axis=0)

        # print("commands_robot: ", commands_robot)
        # action = torch.clip(kf.xhat[:, :2], min=min_lin_vel, max=max_lin_vel)
        # if command[:, 2] == 0:
        #     import pdb; pdb.set_trace()

        kf.predict(commands_robot)
        pred_traj = np.append(pred_traj, [kf.filter._belief_mean.numpy()], axis=0)
        print("predicted state: ", kf.filter._belief_mean)


        # find environments where robot is visible
        dtheta = xrel[:, -1]
        half_fov = robot_full_fov / 2.
        leq = torch.le(torch.abs(torch.tensor(dtheta.reshape(num_envs, 1))), half_fov)
        fov_bool = torch.any(leq, dim=1)
        visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()

        # simulate observations
        observations = kf.sim_observations(states=torch.tensor(xrel))

        if tidx == 0:
            z_traj = np.array([observations.numpy()])
        else:
            z_traj = np.append(z_traj, [observations.numpy()], axis=0)

        # print("visible_env_ids: ", visible_env_ids)

        # update the mean and covariance
        # visible_env_ids = torch.arange(num_envs) # TODO: HACK
        kf.update(observations[visible_env_ids, :], visible_env_ids)

        print("updated state: ", kf.filter._belief_mean)
        print("real state: ", xrel)

        est_traj = np.append(est_traj, [kf.filter._belief_mean.numpy()], axis=0)
        P_traj = np.append(P_traj, [kf.filter._belief_covariance.numpy()], axis=0)


    print("Plotting state traj...")
    plot_state_traj(real_traj, z_traj, pred_traj, est_traj)
    print("Plotting covariance matrix traj...")
    plot_state_cov_mat(P_traj, est_traj)

    print("est_traj: ", est_traj)
    print("P_traj: ", P_traj)

def sim_nonlinear_sys_ukf():
    """Simulate data for nonlinear system robot."""
    state_dim = 4
    control_dim = 2 # u = (lin_vel, ang_yaw_vel)
    observation_dim = 4
    num_envs = 2
    dt = 0.02
    robot_full_fov = 1.20428
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the real system
    max_s = 7
    t = np.arange(0, max_s, dt)
    print("Simulating for ", max_s, " seconds...")

    kf = UKFTorchFilter(state_dim,
                        control_dim,
                        observation_dim,
                        num_envs,
                        dt,
                        device=device,
                        ang_dims=-1,
                        dyn_sys_type="nonlinear")

    xa0 = torch.zeros(num_envs, state_dim)
    xa0[0, 0] = 0
    xa0[0, 1] = 0
    xa0[0, 3] = 0
    xa0[1, 0] = 1
    xa0[1, 1] = 1
    xa0[1, 3] = 0
    xr0 = torch.zeros(num_envs, state_dim)
    xr0[0, 0] = 3
    xr0[0, 1] = 2
    xr0[0, 3] = np.pi / 2
    xr0[1, 0] = -3
    xr0[1, 1] = -2
    xr0[1, 3] = np.pi/4

    # setup the dtheta state
    xrel0 = xa0 - xr0
    xrel0[:, -1] = torch.atan2(xrel0[:, 1], xrel0[:, 0]) - xr0[:, -1]  # fourth state is dtheta
    xrel0[:, -1] = kf._wrap_to_pi(xrel0[:, -1])

    observations = kf.sim_observations(states=torch.tensor(xrel0))
    kf.reset_mean_cov(env_ids=torch.arange(num_envs), mean=observations) # TODO: here gets perfect initial state!

    print("belief mean: ", kf.filter._belief_mean)
    print("belief cov: ", kf.filter._belief_covariance)

   # create the fake ground-truth data
    xrel = xrel0
    real_traj = np.array([xrel.numpy()])
    for tidx in range(len(t)):
        commands_robot = kf._dubins_command_robot(xrel)
        # action = torch.clip(xrel[:, :2], min=min_lin_vel, max=max_lin_vel)
        xrel, Q_tril = kf.dynamics(xrel, commands_robot)

        # add noise
        xrel = xrel + (
                Q_tril @ torch.randn(size=(num_envs, state_dim, 1))
            ).squeeze(-1)
        real_traj = np.append(real_traj, [xrel.numpy()], axis=0)
        
    # do the prediction and simulation loop
    pred_traj = np.array([kf.filter._belief_mean.numpy()])
    est_traj = np.array([kf.filter._belief_mean.numpy()])
    z_traj = np.zeros(0)
    P_traj = np.array([kf.filter._belief_covariance.numpy()])
    for tidx in range(len(t)):
        print("predicting...")
        command = kf._dubins_command_robot(kf.filter._belief_mean)
        # action = torch.clip(kf.xhat[:, :2], min=min_lin_vel, max=max_lin_vel)
        kf.predict(command)
        pred_traj = np.append(pred_traj, [kf.filter._belief_mean.numpy()], axis=0)

        # get a measurement
        if tidx % 1 == 0:
            print("got measurement, doing corrective update...")
            x = real_traj[tidx, :]
            observations = kf.sim_observations(states=torch.tensor(x))
            if tidx == 0:
                z_traj = np.array([observations.numpy()])
            else:
                z_traj = np.append(z_traj, [observations.numpy()], axis=0)

            kf.update(observations, torch.arange(num_envs))

        est_traj = np.append(est_traj, [kf.filter._belief_mean.numpy()], axis=0)
        P_traj = np.append(P_traj, [kf.filter._belief_covariance.numpy()], axis=0)

    print("Plotting state traj...")
    plot_state_traj(real_traj, z_traj, pred_traj, est_traj)
    print("Plotting covariance matrix traj...")
    plot_state_cov_mat(P_traj, est_traj)

    print("est_traj: ", est_traj)
    print("P_traj: ", P_traj)

if __name__ == "__main__":
    sim_linear_sys_ukf()
    # sim_nonlinear_sys_ukf()
    