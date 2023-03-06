import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import multivariate_normal
import pdb
from datetime import datetime
import os

class KalmanFilter(object):
    def __init__(self, dt, num_states, num_actions, num_envs, device='cpu', dtype=torch.float64):
        self.dt = dt
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype

        # dynamics matricies x' = Ax + Bu + w
        self.A = torch.eye(self.num_states, dtype=self.dtype, device=self.device)
        self.B = -self.dt * torch.eye(self.num_states, self.num_actions, dtype=self.dtype, device=self.device)

        self.H = torch.eye(self.num_states, dtype=self.dtype, device=self.device)          # observation matrix y = Hx + v
        self.Q = 0.01 * torch.eye(self.num_states, dtype=self.dtype, device=self.device)    # process covariance (w ~ N(0,Q))
        self.R = 0.2 * torch.eye(self.num_states, dtype=self.dtype, device=self.device)    # measurement covariance (v ~ N(0,R))
        self.P = torch.eye(self.num_states, dtype=self.dtype, device=self.device)          # a posteriori estimate covariance matrix
        self.I = torch.eye(self.num_states, device=self.device)

        # current state estimate
        self.xhat = torch.zeros(self.num_envs, self.num_states, dtype=self.dtype, device=self.device)

        # torchify everything
        self.A_tensor = self.A.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)
        self.B_tensor = self.B.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_actions)
        self.H_tensor = self.H.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)
        self.Q_tensor = self.Q.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)
        self.R_tensor = self.R.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)
        self.P_tensor = self.P.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)
        self.I_tensor = self.I.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)

        # create N(0, R) to sample measurements from
        zero_mean = torch.zeros(self.num_envs, self.num_states, dtype=self.dtype, device=self.device)
        self.normal_dist = MultivariateNormal(loc=zero_mean, covariance_matrix=self.R)

    def set_init_xhat(self, xhat):
        """Initializes estimated state.
        Args: 
            xhat (torch.Tensor): initial estimated state of shape [num_envs, num_states]
        """
        self.xhat = xhat

    def reset_xhat(self, env_ids, xhat_val=None):
        """Resets the initial estimate.
        Args:
            env_ids [list of Ints]: environment indicies to be reset
            reset_val [torch.Tensor]: of length [num_env_ids x num_states] containing values to reset to
        """
        # reset initial state estimate
        if xhat_val is None:
            self.xhat[env_ids, :] = 0.
        else:
            self.xhat[env_ids, :] = xhat_val
        # reset state covariance
        self.P_tensor[env_ids, :] = self.P

    def dynamics(self, x, command_robot):
        """
        Applies linear dynamics to evolve state: x' = Ax + Bu

        Args: 
            x (torch.Tensor): current state of shape [num_envs, num_states]
            command_robot (torch.Tensor): current action of shape [num_envs, num_actions]
        Returns:
            x' (torch.Tensor): next state of shape [num_envs, num_states]
        """
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if command_robot.dtype != self.dtype:
            command_robot = command_robot.to(self.dtype)

        Ax = torch.bmm(self.A_tensor, x.unsqueeze(-1)).squeeze(-1)
        Bu = torch.bmm(self.B_tensor, command_robot.unsqueeze(-1)).squeeze(-1)
        xnext = Ax + Bu
        return xnext

    def predict(self, command_robot):
        """
        Precition, i.e., time update.

        Args: 
            command_robot (torch.Tensor): current action of shape [num_envs, num_actions]
        Returns:
            xhat (torch.Tensor): predicted (a priori) state estimate of shape [num_envs, num_states]
        """

        if command_robot.dtype != self.dtype:
            command_robot = command_robot.to(self.dtype)

        # Predict the next state by applying dynamics
        #   xhat' = Axhat + Bu
        self.xhat = self.dynamics(self.xhat, command_robot)

        # Predict the error covariance 
        #   Phat = A * P * A' + Q
        AP = torch.bmm(self.A_tensor, self.P_tensor)
        A_top = torch.transpose(self.A_tensor, 1, 2)
        self.P_tensor = torch.bmm(AP, A_top) + self.Q_tensor
        return self.xhat

    def correct(self, z, env_ids):
        """
        Measurement update.

        Args:
            z (torch.Tensor): measurement of shape [num_envs, num_obs]
            env_ids (torch.Tensor): of length [num_env_ids] environments to be corrected
        """

        if len(env_ids) == 0:
            return

        if z.dtype != self.dtype:
            z = z.to(self.dtype)

        # Compute Kalman gain:
        #   S = H * Phat H^T + R
        #   K = Phat * H^T S^-1
        H_top = torch.transpose(self.H_tensor, 1, 2)
        S = torch.bmm(torch.bmm(self.H_tensor, self.P_tensor), H_top) + self.R_tensor
        K = torch.bmm(self.P_tensor, torch.bmm(H_top, torch.pinverse(S)))

        # Measurement residual
        #   y = z - H * xhat
        y = (z - torch.bmm(self.H_tensor, self.xhat.unsqueeze(-1)).squeeze(-1))
        
        # Get a posteriori estimate using measurement z
        #   xhat = xhat + K * y
        self.xhat[env_ids, :] = self.xhat[env_ids, :] + torch.bmm(K[env_ids, :], y[env_ids, :].unsqueeze(-1)).squeeze(-1)

        # Get a posteriori estimate covariance
        #   P = (I - K * H) * Phat
        self.P_tensor[env_ids, :] = torch.bmm(self.I_tensor[env_ids, :] -
                                              torch.bmm(K[env_ids, :], self.H_tensor[env_ids, :]), self.P_tensor[env_ids, :])

    def sim_measurement(self, xreal):
        """
        Simulate a measurement given the ground-truth state.

        Args:
            xreal (torch.Tensor): real state of shape [num_envs, num_states]

        Returns:
            z (torch.Tensor): measurement of shape [num_env_ids, num_states]
        """
        # measurements are drawn from v ~ N(0, R)
        # zero_mean = np.zeros(self.num_states)
        # v = np.random.multivariate_normal(zero_mean, self.R)
        # v = v.reshape(self.num_states, 1)

        if xreal.dtype != self.dtype:
            xreal = xreal.to(self.dtype)

        # torchify everything
        v_tensor = self.normal_dist.rsample()

        # use y = Hx + v to simulate measurement
        Hx = torch.bmm(self.H_tensor, xreal.unsqueeze(-1)).squeeze(-1)
        z = Hx + v_tensor
        return z

    def _plot_state_traj(self, real_traj, z_traj, pred_traj, est_traj):
        """
        Plots trajectories.

        Args:
            real_traj (np.array): ground truth state traj of shape [num_tsteps, num_envs, num_states]
            z_traj (np.array): measurement traj of shape [num_tsteps, num_envs, num_obs] 
            pred_traj (np.array): predicted state traj of shape [num_tsteps, num_envs, num_states] 
            est_traj (np.array): estimated state traj of shape [num_tsteps, num_envs, num_states]  
        """
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)

        colors_z = np.linspace(0, 1, len(z_traj[:,0,0]))
        colors_r = np.linspace(0, 1, len(real_traj[:,0,0]))
        colors_e = np.linspace(0, 1, len(est_traj[:, 0, 0]))
        if pred_traj is not None:
            colors_p = np.linspace(0, 1, len(pred_traj[:, 0, 0]))
        
        for eidx in range(self.num_envs):
            # plot measurements
            plt.scatter(z_traj[:,eidx,0], z_traj[:,eidx,1], c=colors_z, cmap='Greens', label="measurements")
            if pred_traj is not None:
                # plot predicted state trajectory (a priori)
                plt.scatter(pred_traj[:,eidx,0], pred_traj[:,eidx,1], c=colors_p, cmap='Blues', label="preds")
            # plot estimated state trajectory (a posteriori)
            plt.scatter(est_traj[:,eidx,0], est_traj[:,eidx,1], c=colors_e, cmap='Reds', label="estimates")
            # plot GT state trajectory
            plt.scatter(real_traj[:,eidx,0], real_traj[:,eidx,1], c=colors_r, cmap='Greys', label="real state")
            # plot initial state explicitly
            plt.scatter(real_traj[0,eidx,0], real_traj[0,eidx,1], c=colors_r[0], edgecolors='k', cmap='Greys')
        
        # plot the origin (i.e., the goal of the controller)
        plt.scatter(0, 0, c='m', edgecolors='k', marker=",", s=100)

        # plt.ylim([-5, 5])
        # plt.xlim([-5, 5])
        plt.xlabel("relative x-pos")
        plt.ylabel("relative y-pos")

        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels)
        # ax.legend()

        red_patch = mpatches.Patch(color='red', label='estimates')
        green_patch = mpatches.Patch(color='green', label='measurements')
        black_patch = mpatches.Patch(color='black', label='real state')
        # if pred_traj is not None:
        #     blue_patch = mpatches.Patch(color='blue', label='preds')
        plt.legend(handles=[red_patch, green_patch, black_patch])

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
        path = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(path + '/imgs/' + 'kalman_filter'+dt_string+'.png')
        # plt.show()

    def _plot_state_cov_mat(self, P_traj, est_traj, env_id):
        """
        Plots state covariance matrix P over time. 
        Args:
            P_traj (np.array): of shape [num_tsteps, num_envs, num_states, num_states]
        """
        num_tsteps = len(P_traj[:,0,0,0])
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        colors = np.linspace(0,1,num_tsteps)

        # Initializing the random seed
        random_seed = 1000

        # Setting mean of the distributino to
        # be at (0,0)
        mean = np.array([0, 0])

        for env_id in range(P_traj.shape[1]):
            for tidx in range(0, num_tsteps, 15):
                # Generating a Gaussian bivariate distribution
                # with given mean and covariance matrix
                distr = multivariate_normal(cov=P_traj[tidx,env_id,0,0], mean=mean,
                                            seed=random_seed)

                # Generating samples out of the distribution
                data = distr.rvs(size=500)

                # Plotting the generated samples
                xhat = est_traj[tidx, env_id, :]
                plt.scatter(xhat[0] + data[:, 0], xhat[1] + data[:, 1], color=[1-colors[tidx], 0, colors[tidx]], alpha=0.1)

            plt.scatter(est_traj[:, env_id, 0], est_traj[:, env_id, 1], c='k', s=70)
            plt.scatter(est_traj[0, env_id, 0], est_traj[0, env_id, 1], c='w', facecolor='w', edgecolors='k', s=70)

        # plot the origin (i.e., the goal of the controller)
        plt.scatter(0, 0, c='m', edgecolors='k', marker=",", s=100)

        plt.title('State Estimate Covariance (P)')
        # plt.show()
        path = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(path + '/imgs/' + 'kalman_filter_P_mat.png')


    def _compute_mse(self, xrel_traj, pred_traj, est_traj):
        """Compute MSE per each state dimension."""
        for eidx in range(self.num_envs):
            num_tsteps = len(xrel_traj[:,0,0])
            diff_sq_est = (xrel_traj[:, eidx, :] - est_traj[:, eidx, :])**2
            mse_est = (1 / num_tsteps) * np.sum(diff_sq_est, axis=0)
            diff_sq_pred = (xrel_traj[:, eidx, :] - pred_traj[:, eidx, :])**2
            mse_pred = (1 / num_tsteps) * np.sum(diff_sq_pred, axis=0)
            print("ENV ", eidx, " | MSE[real - est] (x):",  mse_est[0], "(y): ", mse_est[1], "(z): ", mse_est[2])
            print("ENV ", eidx, " | MSE[real - pred] (x):",  mse_pred[0], "(y): ", mse_pred[1], "(z): ", mse_pred[2])

if __name__ == '__main__':
    dt = 0.02
    num_states = 3
    num_actions = 2
    num_envs = 1 #4
    min_lin_vel = -1 
    max_lin_vel = 1
    device = 'cpu'
    dtype = torch.float

    # define the real system
    max_s = 4
    t = np.arange(0, max_s, dt)
    print("Simulating for ", max_s, " seconds..." )

    xa0 = torch.ones(num_envs, num_states, dtype=dtype, device=device)
    xr0 = torch.ones(num_envs, num_states, dtype=dtype, device=device)
    xr0[0, 0] = 3
    xr0[0, 1] = 2
    # xr0[1, 0] = -3
    # xr0[1, 1] = -2
    # xr0[2, 0] = 3
    # xr0[2, 1] = -3
    # xr0[3, 0] = -3
    # xr0[3, 1] = 3
    xrel0 = xa0 - xr0

    # define the kalman filter and set init state
    kf = KalmanFilter(dt, num_states, num_actions, num_envs, device, dtype)
    xhat0 = kf.sim_measurement(xrel0)
    xhat0[:, 0] = -2
    xhat0[:, 1] = -2
    xhat0[:, 2] = 0
    kf.set_init_xhat(xhat0)

    # create the fake ground-truth data
    xrel = xrel0
    real_traj = np.array([xrel.numpy()])
    for tidx in range(len(t)):
        action = torch.clip(xrel[:, :2], min=min_lin_vel, max=max_lin_vel)
        xrel = kf.dynamics(xrel, action)
        real_traj = np.append(real_traj, [xrel.numpy()], axis=0)

    # do the prediction and simulation loop
    pred_traj = np.array([xhat0.numpy()])
    est_traj = np.array([xhat0.numpy()])
    z_traj = np.zeros(0)
    P_traj = np.array([kf.P_tensor.numpy()])
    for tidx in range(len(t)):
        print("predicting...")
        action = torch.clip(kf.xhat[:, :2], min=min_lin_vel, max=max_lin_vel)
        xhat = kf.predict(action)
        pred_traj = np.append(pred_traj, [xhat.numpy()], axis=0)
    
        # get a measurement
        if tidx % 20 == 0:
            print("got measurement, doing corrective update...")
            x = real_traj[tidx, :]
            z = kf.sim_measurement(torch.tensor(x, dtype=torch.float64))
            if tidx == 0:
                z_traj = np.array([z.numpy()])
            else:
                z_traj = np.append(z_traj, [z.numpy()], axis=0)
            kf.correct(z)

        est_traj = np.append(est_traj, [kf.xhat.numpy()], axis=0)
        P_traj = np.append(P_traj, [kf.P_tensor.numpy()], axis=0)

    print("Plotting state traj...")
    kf._plot_state_traj(real_traj, z_traj, pred_traj, est_traj)
    print("Plotting covariance matrix traj...")
    kf._plot_state_cov_mat(P_traj, 0)
    print("Computing MSE...")
    kf._compute_mse(real_traj, pred_traj, est_traj)
