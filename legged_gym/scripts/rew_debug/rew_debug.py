import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def proj_point_on_line(s, g, p):
    sp = p - s
    sg = g - s
    coeff = np.sum(sp * sg, axis=-1) / np.sum(sg * sg, axis=-1)
    proj = s + np.expand_dims(coeff, axis=-1) * sg
    # perp_line = p - result
    # bla = np.sum(perp_line * sg, axis=-1)
    # ang = np.arccos(bla / np.sqrt(np.sum(perp_line ** 2, axis=-1)) * np.sqrt(np.sum(sg ** 2, axis=-1)))
    # print(ang * 180 / np.pi )
    return proj, coeff

def plot_data(r_start, r_goal, r_curr, r_curr_proj, r_last_proj):
    proj_curr, _ = proj_point_on_line(r_start, r_goal, r_curr)

    print("shape of r_start: ", r_start.shape)
    print("shape of r_goal: ", r_goal.shape)
    print("shape of r_curr: ", r_curr.shape)
    print("shape of r_curr_proj: ", r_curr_proj.shape)

    fig, axs = plt.subplots(2, 1, figsize=(6, 9), gridspec_kw={'height_ratios': [2, 1]})

    tsteps = list(range(0, len(r_curr[:, 0])))
    colors = np.linspace(0, 1, len(r_curr[:, 0]))
    # colors_r = np.linspace(0, 1, len(real_traj[:, 0, 0]))
    # colors_e = np.linspace(0, 1, len(est_traj[:, 0, 0]))

    colormaps = ["Blues", "Greens", "Reds", "Purples"]
    color_bla = ['b', 'g', 'r', 'o']

    for env_id in range(r_start.shape[1]):
        axs[0].plot([r_start[0, env_id, 0], r_goal[0, env_id, 0]], [r_start[0, env_id, 1], r_goal[0, env_id, 1]], 'k--')

        # plot measurements
        axs[0].scatter(r_curr[:, env_id, 0], r_curr[:, env_id, 1], c=colors, cmap=colormaps[env_id])
        axs[0].scatter(proj_curr[:, env_id, 0], proj_curr[:, env_id, 1], c=colors, cmap='Greys')
        axs[0].scatter(r_start[:, env_id, 0], r_start[:, env_id, 1], c='k')
        axs[0].scatter(r_goal[:, env_id, 0], r_goal[:, env_id, 1], c='r')

        # initial condition
        axs[0].scatter(r_curr[0, env_id, 0], r_curr[0, env_id, 1], c='w', edgecolor='k')

        axs[0].scatter(r_curr[40, env_id, 0], r_curr[40, env_id, 1], c='m', edgecolor='k')
        axs[0].scatter(proj_curr[40, env_id, 0], proj_curr[40, env_id, 1], c='m', edgecolor='k')

        # axs[0].scatter(r_curr[200, 0], r_curr[200, 1], c='g', edgecolor='k')
        # axs[0].scatter(proj_curr[200, 0], proj_curr[200, 1], c='g', edgecolor='k')

        # ax.scatter(r_curr[400, 0], r_curr[400, 1], c='g')
        # ax.scatter(proj_curr[400, 0], proj_curr[400, 1], c='g')

        # axs[0].scatter(r_curr[20, 0], r_curr[20, 1], c='y', edgecolor='k')
        # axs[0].scatter(proj_curr[20, 0], proj_curr[20, 1], c='y', edgecolor='k')
        axs[0].set_xlim([-5, 8])
        axs[0].set_ylim([-5, 8])

        # print("problematic r_curr_proj: ", r_curr_proj[30:32])
        # print("problematic proj_curr: ", proj_curr[30:32])
        rew = r_curr_proj[:, env_id] - r_last_proj[:, env_id]
        axs[1].scatter(tsteps, rew, c=colors, cmap=colormaps[env_id])
        axs[1].scatter(40, rew[40], c='m', edgecolor='k')
        # axs[1].scatter(200, rew[200], c='g', edgecolor='k')
        # axs[1].scatter(20, rew[20], c='y', edgecolor='k')
        axs[1].set_xlabel('timestep')
        axs[1].set_ylabel('rew')
        # ax2.scatter(tsteps, r_curr_proj, c=colors, cmap='Blues_r')
        # ax2.scatter(tsteps, r_last_proj)

    plt.show()

def plot_rew_func(r_start, r_goal):
    nx, ny = (10, 10)
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    xv, yv = np.meshgrid(x, y)

    # plt.plot(xv, yv, marker='o', color='k', linestyle='none')
    rew_arr = np.zeros((nx, ny))
    rew_arr_x = np.zeros(nx)
    rew_arr_y = np.zeros(ny)
    for i in range(nx):
        coeff_str = ""
        for j in range(ny):
            # treat xv[i,j], yv[i,j]
            r_curr = np.array([xv[i,j], yv[i,j]])
            _, coeff = proj_point_on_line(r_start, r_goal, r_curr)
            rew_arr[i, j] = coeff
            rew_arr_x[i] = coeff
            rew_arr_y[i] = coeff
            coeff_str += ", " + str(round(coeff,2))
        print(coeff_str)

    fig, ax = plt.subplots()
    # im = plt.matshow(rew_arr)
    # heatmap, xedges, yedges = np.histogram2d(rew_arr[:,0], rew_arr[0,:])
    # extent = [x[0], x[-1], y[0], y[-1]]

    # plt.clf()
    # im = plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=plt.cm.jet)
    # im = ax.imshow(rew_arr, cmap="RdYlGn")

    # for i in range(nx):
    #     for j in range(ny):
    #         text = ax.text(j, i, rew_arr[i, j],
    #                        ha="center", va="center", color="w")

    cbar = ax.figure.colorbar(im, ax=ax)
    # ax.set_xticks(x, minor=True)
    # ax.set_yticks(y, minor=True)
    # fig.tight_layout()
    plt.show()

if __name__ == '__main__':

    path = os.path.dirname(os.path.abspath(__file__))
    # file = '/rew_debug_28_02_2023-18-12-36_300.pickle' # 2 agents
    # file = '/rew_debug_28_02_2023-18-14-41_300.pickle'
    # file = '/rew_debug_28_02_2023-12-03-39_300.pickle' # 2 agents
    # file = '/rew_debug_28_02_2023-11-31-01_400.pickle' # 1 agent
    # file = '/rew_debug_28_02_2023-11-58-57_300.pickle' # 1 agent
    filename = path + file
    with open(filename, "rb") as f:
        data_dict = pickle.load(f)

    r_start = data_dict["r_start"]
    r_goal = data_dict["r_goal"]
    r_curr = data_dict["r_curr"]
    r_curr_proj = data_dict["r_curr_proj"]
    r_last = data_dict["r_last"]
    r_last_proj = data_dict["r_last_proj"]
    rew = r_curr_proj - r_last_proj

    print("avg rew per env: ", np.mean(rew, axis=0))
    print("avg rew: ", np.mean(rew))

    plot_data(r_start, r_goal, r_curr, r_curr_proj, r_last_proj)

    # plot_rew_func(r_start[0,0:2], r_goal[0,0:2])