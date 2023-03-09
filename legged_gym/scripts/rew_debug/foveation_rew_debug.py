import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def plot_foveation_rew():
    ths = np.linspace(-np.pi, np.pi, 100)
    a = np.zeros(len(ths))
    b = np.zeros(len(ths))

    # RELU-like Reward
    offset = np.pi / 3
    # offset = np.pi / 3
    max_rew_val = 0.9  # 0.7

    # m = 1.0
    # m = 0.6
    m = 0.45
    diff_left = m * ths + offset
    diff_right = m * ths - offset

    # a = np.maximum(0, diff_left)
    # b = -np.minimum(0, diff_right)
    a = diff_left
    b = -diff_right
    func = np.minimum(np.minimum(a, b), max_rew_val)

    plt.plot(ths, a, '--')
    plt.plot(ths, b, '--')
    plt.plot(ths, func)

    # Exponentiated reward
    # a = np.exp(-np.abs(ths))
    # plt.plot(ths, a)

    plt.plot((np.pi / 8) * np.ones(len(ths)), np.linspace(-2, 4, len(ths)), '--k')
    plt.plot(-(np.pi / 8) * np.ones(len(ths)), np.linspace(-2, 4, len(ths)), '--k')
    plt.plot(np.linspace(-4, 4, len(ths)), np.zeros(len(ths)), 'k')
    plt.ylim(-0.5, 1.5)
    plt.show()


def plot_data(angles, rews, commands=None):
    print("shape of angles: ", angles.shape)
    print("shape of rews: ", rews.shape)

    num_axs = 2
    ratios = [1, 1]
    if commands is not None:
        num_axs = 3
        ratios = [1, 1, 1]

    fig, axs = plt.subplots(num_axs, 1, figsize=(6, 9), gridspec_kw={'height_ratios': ratios})

    num_tsteps = len(angles[:, 0])
    tsteps = np.linspace(0, num_tsteps, num_tsteps)

    for env_id in range(len(angles[0, :])):
        # axs[0].plot(angles[:, env_id], rews[:, env_id])
        axs[0].plot(tsteps, angles[:, env_id], 'r')
        axs[0].set_xlabel("time step")
        axs[0].set_ylabel("rel yaw")
        axs[1].plot(tsteps, rews[:, env_id], 'b')
        axs[1].set_xlabel("time step")
        axs[1].set_ylabel("reward")
        if commands is not None:
            axs[2].plot(tsteps, commands[:, env_id], 'k')
            axs[2].set_xlabel("time step")
            axs[2].set_ylabel("angular velocity cmd")
            axs[2].set_ylim([-1,1])
    plt.show()


if __name__ == '__main__':
    # plot_foveation_rew()

    path = os.path.dirname(os.path.abspath(__file__))
    # file = '/foveation/foveation_rew_debug_08_03_2023-13-30-55_200.pickle' # 1000 iter policy
    # file = '/foveation/foveation_rew_debug_08_03_2023-13-31-28_200.pickle'  # 200 iter policy

    file = '/foveation/foveation_rew_debug_08_03_2023-14-24-45_200.pickle'

    # file = '/foveation/foveation_rew_debug_08_03_2023-13-28-32_200.pickle'
    # file = '/foveation/foveation_rew_debug_08_03_2023-12-02-52_200.pickle'
    # file = '/foveation/foveation_rew_debug_08_03_2023-11-39-58_200.pickle' # 2 agents
    filename = path + file
    with open(filename, "rb") as f:
        data_dict = pickle.load(f)

    fov_reward = data_dict["fov_reward"]
    fov_rel_yaw = data_dict["fov_rel_yaw"]
    ll_env_command_ang_vel = data_dict["ll_env_command_ang_vel"]

print("avg foveation rew per env: ", np.mean(fov_reward, axis=0))
print("avg foveation rew: ", np.mean(fov_reward))
print("avg command rew: ", np.mean(ll_env_command_ang_vel))

plot_data(fov_rel_yaw, fov_reward, ll_env_command_ang_vel)
