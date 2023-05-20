
from legged_gym.envs.base.base_config import BaseConfig

class RMADecHighLevelGameCfg( BaseConfig ):
    class env:
        debug_viz = False
        robot_hl_dt = 0.2   # 1 / robot_hl_dt is the Hz

        num_envs = 3000 # 4096

        num_actions_robot = 3           # robot (lin_vel_x, lin_vel_y, ang_vel_yaw) = 3
        num_actions_agent = 2           # other agent (lin_vel, ang_vel) = 2
        num_priv_robot_states = None    # ONLY FOR PO: (xhat, cov_diag) = 8
        num_robot_states = 4            # x = (px, py, pz, theta)
        num_agent_states = 3            # x = (px, py, pz)
        num_pred_steps = 8              # prediction length
        num_hist_steps = 8              # history length

        # interaction scenario options:
        #       'nav' if doing goal-reaching w/agent avoidance
        #       'game' pursuit-evasion interaction
        interaction_type = 'game'

        # robot policy type options:
        #       'reaction' uses only current relative state
        #       'estimation' uses history of relative state and robot actions
        #       'prediction_phase1' uses privileged info about future relative state
        #       'prediction_phase2' uses history of relative state and robot actions with privileged future predictions
        robot_policy_type = 'reaction'

        # ====== [Pursuit-Evasion Game] ====== #
        if interaction_type == 'game':
            # BASELINE - REACTION
            if robot_policy_type == 'reaction':
                num_observations_robot = num_robot_states
                num_observations_agent = 4
                num_privileged_obs_robot = None
                num_privileged_obs_agent = None
            # BASELINE - ESTIMATION
            elif robot_policy_type == 'estimation':
                num_observations_robot = num_robot_states * (num_hist_steps + 1) + num_actions_robot * num_hist_steps
                num_observations_agent = 4
                num_privileged_obs_robot = None
                num_privileged_obs_agent = None
            # PREDICTION - PHASE 1
            elif robot_policy_type == 'prediction_phase1':
                num_observations_robot = num_robot_states * (num_pred_steps + 1)        # PREDICTIONS: pi(x^t, x^t+1:t+N)
                num_observations_agent = 4
                num_privileged_obs_robot = None
                num_privileged_obs_agent = None
            # PREDICTION - PHASE 2
            elif robot_policy_type == 'prediction_phase2':
                num_observations_robot = num_robot_states * (num_hist_steps + 1) + num_actions_robot * num_hist_steps # HISTORY: pi(x^t, x^t-N:t, uR^t-N:t-1)
                num_observations_agent = 4
                num_privileged_obs_robot = num_robot_states * (num_pred_steps + 1)        # PREDICTIONS: pi(x^t, x^t+1:t+N)
                num_privileged_obs_agent = None
            # PARTIAL OBSERVABILITY PREDICTION - PHASE 2
            elif robot_policy_type == 'po_prediction_phase2':
                # Used for the partially-observable STUDENT
                num_observations_priv_robot = num_priv_robot_states * (num_hist_steps + 1) + num_actions_robot * num_hist_steps # HISTORY: pi(x^t, x^t-N:t, uR^t-N:t-1)
                num_observations_priv_agent = 4
                num_privileged_obs_priv_robot = num_priv_robot_states * (num_pred_steps + 1)        # PREDICTIONS: pi(x^t, x^t+1:t+N)

                # Used for the fully-observable TEACHER
                num_observations_robot = num_robot_states * (num_hist_steps + 1) + num_actions_robot * num_hist_steps # HISTORY: pi(x^t, x^t-N:t, uR^t-N:t-1)
                num_observations_agent = 8
                num_privileged_obs_robot = num_robot_states * (num_pred_steps + 1)        # PREDICTIONS: pi(x^t, x^t+1:t+N)
                num_privileged_obs_agent = None
        # =========== END ========== #

        # ====== [Navigation] ====== #
        if interaction_type == 'nav':
            if robot_policy_type == 'prediction_phase1':
                # PHASE 1 INFO
                num_observations_robot = num_robot_states * (num_pred_steps + 1) + 3       # PREDICTIONS + GOAL: pi(goal_xyth, x^t, x^t+1:t+N)
                num_observations_agent = 4
                num_privileged_obs_robot = None
                num_privileged_obs_agent = None
            elif robot_policy_type == 'prediction_phase2':
                # PHASE 2 INFO
                num_observations_robot = num_robot_states * (num_hist_steps + 1) + num_actions_robot * num_hist_steps + 3 # HISTORY: pi(goal_xyth, x^t, x^t-N:t, uR^t-N:t-1)
                num_observations_agent = 4
                num_privileged_obs_robot = num_robot_states * (num_pred_steps + 1) + 3       # PREDICTIONS: pi(goal_xyth, x^t, x^t+1:t+N)
                num_privileged_obs_agent = None
        # =========== END ========== #

        env_spacing = 3.            # not used with heightfields / trimeshes
        send_timeouts = False       # send time out information to the algorithm
        send_BC_actions = True      # send optimal robot actions for the BC loss in the algorithm; TODO: need to clean up this var
        episode_length_s = 20       # episode length in seconds
        capture_dist = 0.8          # if the two agents are closer than this dist, they are captured

        # [Navigation]
        goal_dist = 0.8             # if the robot is closer than this dist to the goal, it is done
        collision_dist = 1.5

        # simulated agent info
        # options for agent's dynamics:
        #       'dubins' (u = linvel, angvel)
        #       'integrator' (u = xvel, yvel)
        agent_dyn_type = 'dubins'

        # options for agent policy type:
        #       'simple_weaving' it follows dubins' curves
        #       'complex_weaving' it follows random linear and angular velocity combinations
        #       'static' just stands still
        agent_policy_type = 'complex_weaving'
        agent_ang = [-3.14, 3.14]       # initial condition: [min, max] relative angle to robot
        agent_rad = [2.0, 6.0]          # initial condition: [min, max] spawn radius away from robot

        # [Pursuit-Evasion Game] for 'simple_weaving' agent policy only
        agent_turn_freq = [50, 100]  # sample how long to turn (tsteps) from [min, max]
        agent_straight_freq = [100, 200]  # sample how long to keep straight (tsteps) from [min, max]
        randomize_init_turn_dir = True  # if True, then initial turn going left or right is randomized
        # agent_turn_freq = [50, 100]                   # sample how long to turn (tsteps) from [min, max]
        # agent_straight_freq = [100, 200]              # sample how long to keep straight (tsteps) from [min, max]
        # randomize_init_turn_dir = True                # if True, then initial turn going left or right is randomized

        # [Navigation] for goal initialization
        goal_ang = [-3.14, 3.14]        # goal location specs: [min, max] relative angle to robot
        goal_rad = [3.0, 7.0]           # goal location specs: [min, max] goal distance away from robot
        agent_init_bias = False          # bias the agent to spawn "in the way" of the robot towards its goal
        agent_policy_bias = False         # bias a subset of the agent random policies to "pursue" the robot

    class robot_sensing:
        filter_type = "kf" # options: "ukf" or "kf"
        # fov = 6.28      # = 360, full FOV
        # fov = 1.20428  # = 64 degrees, RealSense;
        fov = 1.54    # = 88 degrees, ZED 2 HD1080

        fov_curriculum = False
        fov_levels = [6.28, 4.71, 3.14, 1.57, 1.20428] # 360, 270, 180, 90, 64 degrees

        prey_curriculum = False
        prey_angs = [0.52, 1.04, 1.57, 2.4, 3.14] # prey's initial relative angle will be in [-prey_ang, prey_ang]

        obstacle_curriculum = False
        obstacle_heights = [0., 0.1, 0.5, 1, 5] # [m]

        prey_policy_curriculum = False
        prey_policy = ['static', 'active']

        curriculum_target_iters = [200, 400]
        # curriculum_target_iters = [200, 400, 600, 800, 1000] #[400, 800, 1200, 1600, 1800]

    class terrain:
        mesh_type = 'plane'
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        # obstacle terrain only:
        fov_measure_heights = False

        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1m x 1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        # max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 10
        terrain_width = 10
        num_rows = 3  # number of terrain rows (levels)
        num_cols = 3  # number of terrain cols (types)

        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete obstacles, stepping stones, forest]
        terrain_proportions = [0., 0., 0., 0., 0., 0., 1.]

        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

        # forest terrain type only:
        num_obstacles = 0  # number of "trees" in the environment
        obstacle_height = 0  # in [units]; e.g. 500 is very tall, 20 is managable by robot

    class commands: # note: commands and actions are the same for the high-level policy
        # num_robot_commands = 4        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        heading_command = False         # if true: compute ang vel command from heading error
        command_clipping = False        # if true: clip robot + agent commands to the ranges below
        use_joypad = False
        class ranges:
            lin_vel_x = [-3,3]#[0., 3.0]     # min max [m/s]
            lin_vel_y = [-1,1]#[0., 0.]     # min max [m/s]
            ang_vel_yaw = [-2, 2] #[-3.14, 3.14]       # min max [rad/s]
            heading = [-3.14, 3.14]
            agent_lin_vel_x = [-3, 3] # min max [m/s]
            agent_lin_vel_y = [-1, 1] # min max [m/s]
            agent_ang_vel_yaw = [-2.0, 2.0] # min max [rad/s]

    class init_state:
        agent_pos = [0.0, 0.0, 0.3] # x, y, z (agent pos)
        agent_rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        agent_lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        agent_ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        robot_pos = [0.0, 0.0, 0.42]  # x,y,z [m] (robot pos)
        robot_rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        robot_lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        robot_ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
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
        curriculum = False
        max_init_dists = [2.0, 4.0, 6.0, 8.0, 10.0]

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards_robot: # ROBOT!
        only_positive_rewards = False
        class scales:
            pursuit = -1.0
            exp_pursuit = 0. #1.0
            command_norm = -0.0
            robot_foveation = 0.0
            robot_ang_vel = -0.0
            path_progress = 0.0
            time_elapsed = -0.0
            termination = 100.0

            # [Navigation]
            robot_dist_to_goal = 0. #-1.0
            robot_collision_avoid = 0. #-100.0

    class rewards_agent: # CUBE!
        only_positive_rewards = False
        class scales:
            evasion = 1.0
            termination = 0.0

    class normalization:
        class obs_scales:
            height_measurements = 5.0

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

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
            max_gpu_contact_pairs = 2**15 #2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class RMADecHighLevelGameCfgPPO( BaseConfig ):
    seed = 1
    runner_class_name = 'DecGamePolicyRunner' # 'OnPolicyRunner'

    class policy:
        # robot policy type options:
        #       'reaction' uses only current relative state
        #       'estimation' uses history of relative state and robot actions
        #       'prediction_phase1' uses privileged info about future relative state
        #       'prediction_phase2' uses history of relative state and robot actions with privileged future predictions
        robot_policy_type = 'reaction'
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        privilege_enc_hidden_dims = [512, 256, 128]  # i.e. encoder_hidden_dims
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        eval_time = False

        # ActorCriticGamesRMA: information about the estimator(s)
        if robot_policy_type == 'estimation' or robot_policy_type == 'prediction_phase1':
            use_estimator = False   # True uses the learned estimator: zhat = E(x^history, uR^history)
            use_privilege_enc = True         # True uses the teacher estimator: z* = T(x^future)
            init_noise_std = 0.5
        elif robot_policy_type == 'prediction_phase2' or robot_policy_type == 'po_prediction_phase2':
            use_estimator = True   # True uses the learned estimator: zhat = E(x^history, uR^history)
            use_privilege_enc = True         # True uses the teacher estimator: z* = T(x^future)
            init_noise_std = 0.01
        else:
            use_estimator = False   # True uses the learned estimator: zhat = E(x^history, uR^history)
            use_privilege_enc = False         # True uses the teacher estimator: z* = T(x^future)
            init_noise_std = 0.5 

        future_len = 8
        history_len = 8
        num_robot_states = 4
        num_robot_actions = 3
        num_agent_states = 3
        num_latent = 8          # i.e., embedding sz

        # ===== [Pursuit-Evasion Game] ===== #
        if robot_policy_type == 'estimation':
            num_privilege_enc_obs = num_robot_states*(history_len+1) + num_robot_actions*history_len  # i.e., 8-step future relative state
            num_estimator_obs = None
        elif robot_policy_type == 'prediction_phase1':
            num_privilege_enc_obs = num_robot_states * future_len  # i.e., 8-step future relative state
            num_estimator_obs = None
        elif robot_policy_type == 'prediction_phase2':
            num_privilege_enc_obs = num_robot_states * future_len  # i.e., 8-step future relative state
            num_estimator_obs = num_robot_states * (history_len + 1) + num_robot_actions * history_len    # i.e., 8-step past rel-state and robot controls + present state

        #  ===== [Navigation] ===== #
        # num_privilege_enc_obs = num_robot_states * future_len # only the 8-step future relative state is privileged (current state and goal is not)
        # num_estimator_obs = num_robot_states * (history_len + 1) + num_robot_actions * history_len # the estimator treats only the 8-step past rel-state and robot controls + present state as privileged # TODO: FIX THIS IN ACTOR CRITIC RMA

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0. #0.01
        bc_coef = 0. #0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCriticGamesRMA'
        algorithm_class_name = 'PPO'
        eval_time = False
        num_steps_per_env = 10 #24          # per iteration
        max_iterations = 1601           # number of policy updates per agent
        max_evolutions = 1            # number of times the two agents alternate policy updates (e.g., if 100, then each agent gets to be updated 50 times)

        # logging
        save_learn_interval = 200  # check for potential saves every this many iterations
        save_evol_interval = 1
        # load and resume
        resume_robot = True
        resume_agent = False
        load_run = 'phase_1_policy_v3' #nav_phase_1_policy' #'phase_1_policy'  # -1 = last run
        evol_checkpoint_robot = 0       
        learn_checkpoint_robot = 1600   # -1 = last saved model
        evol_checkpoint_agent = 0
        learn_checkpoint_agent = 1400
        resume_path = None  # updated from load_run and chkpt
        run_name = ''
        experiment_name = 'dec_high_level_game'
