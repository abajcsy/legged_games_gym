
from legged_gym.envs.base.base_config import BaseConfig

class DecHighLevelGameCfg( BaseConfig ):
    class env:
        # note:
        #   48 observations for nominal A1 setup
        #   + 187 for non-flat terrain observations
        #   + 3 for relative xyz-state to point-predator
        num_envs = 3000 # 4096
        num_observations_prey = 16      # prey:     (3 rel pred-prey pos * 4-sample long history + 4 for binary occlusion variable) = 16
        num_observations_predator = 5
        # num_observations_predator = 36 # 24
        num_privileged_obs_prey = None
        num_privileged_obs_predator = None
        num_actions_prey = 4         # prey (lin_vel_x, lin_vel_y, ang_vel_yaw, heading) = 4
        num_actions_predator = 3     # predator (vx, vy, omega) = 3
        env_spacing = 3.        # not used with heightfields/trimeshes
        send_timeouts = True    # send time out information to the algorithm
        episode_length_s = 20   # episode length in seconds
        capture_dist = 0.8      # if predator is closer than this dist to prey, they are captured

    class terrain:
        mesh_type = 'plane' # 'trimesh'
        curriculum = False
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)

    class commands: # note: commands and actions are the same for the high-level policy
        # num_prey_commands = 4        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        heading_command = True       # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0]     # min max [m/s]
            lin_vel_y = [-1.0, 1.0]     # min max [m/s]
            ang_vel_yaw = [-1, 1]       # min max [rad/s]
            heading = [-3.14, 3.14]
            predator_lin_vel_x = [-2.0, 2.0] # min max [m/s]
            predator_lin_vel_y = [-2.0, 2.0] # min max [m/s]
            predator_ang_vel_yaw = [-1.0, 1.0] # min max [rad/s]

    class init_state:
        predator_pos = [0.0, 0.0, 0.3] # x, y, z (predator pos)
        predator_rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        predator_lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        predator_ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        prey_pos = [0.0, 0.0, 0.42]  # x,y,z [m] (prey pos)
        prey_rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        prey_lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        prey_ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards_prey:
        only_positive_rewards = True
        class scales:
            evasion = 0.9

    class rewards_predator:
        only_positive_rewards = False
        class scales:
            pursuit = -1.0
            termination = 10.0 #5.0

    # class normalization:
    #     class obs_scales:
    #         lin_vel = 2.0
    #         ang_vel = 0.25
    #         dof_pos = 1.0
    #         dof_vel = 0.05
    #         height_measurements = 5.0
    #     clip_observations = 100.
    #     clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [0, 0, 0]
        # lookat = [11., 5, 3.]  # [m]self.cfg.terrain.num_rows

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class DecHighLevelGameCfgPPO( BaseConfig ):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24          # per iteration
        max_iterations = 1000           # number of policy updates per agent
        max_evolutions = 1            # number of times the predator-prey alternate policy updates (e.g., if 100, then each agent gets to be updated 50 times)  

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        run_name = ''
        experiment_name = 'dec_high_level_game'