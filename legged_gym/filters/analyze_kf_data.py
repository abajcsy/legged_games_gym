import pickle
import os


if __name__ == '__main__':

	path = os.path.dirname(os.path.abspath(__file__))
	# bla = 'kf_data_03_03_2023-17-13-18_200.pickle'
	# bla = 'kf_data_03_03_2023-17-13-14_100.pickle'
	# bla = 'kf_data_06_03_2023-09-57-44_100.pickle'
	bla = 'kf_data_06_03_2023-10-28-17_400.pickle' # +1.5
	# bla = 'kf_data_06_03_2023-10-20-42_300.pickle'  # -1.5
	filename = path + '/data/' + bla #kf_data_27_02_2023-17-18-12_200.pickle'
	with open(filename, "rb") as f:
		data_dict = pickle.load(f)

	kf = data_dict["kf"]
	real_traj = data_dict["real_traj"]
	est_traj = data_dict["est_traj"]
	z_traj = data_dict["z_traj"]
	P_traj = data_dict["P_traj"]
	pred_traj = None

	print("shape of real_traj: ", real_traj.shape)
	print("shape of est_traj: ", est_traj.shape)
	print("shape of z_traj: ", z_traj.shape)
	print("shape of P_traj: ", P_traj.shape)
	kf._plot_state_traj(real_traj, z_traj, pred_traj, est_traj)
	kf._plot_state_cov_mat(P_traj, est_traj, env_id=0)

