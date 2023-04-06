# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import torch
class A1WXVRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 5200
        num_hist_obs = 2
        num_vis_obs = 51
        num_pri_obs = 3+3+4+1+1 + num_vis_obs
        
        # num_observations = (62+num_pri_obs)*num_hist_obs
        # num_actions = 19
        
        num_observations = (41+num_pri_obs)*num_hist_obs
        num_observations_prior = 116 # 2+36+4+4+3+12+1+2+1+51
        num_actions = 12

        # delay_len = 2
        # delay_curriulum = False
        # push_steps = 1
        # vis = True
        # reach = True
        # no_privilege = False
        prior = True
        # train_dog = True
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FR_hip_joint': 0.05 ,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.4,  # [rad]
            
            'FL_hip_joint': -0.05,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.4,   # [rad]
            
            'RR_hip_joint': 0.05,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            'RR_calf_joint': -1.4,    # [rad]
            
            'RL_hip_joint': -0.05,   # [rad]
            'RL_thigh_joint': .8,   # [rad]
            'RL_calf_joint': -1.4,    # [rad]
        }
    
    
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
            
    class commands:
        curriculum = False
        discrete = [] # other option: ["lin_vel_x"]
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = True #True # if true: compute ang vel command from heading error
        active = 0.5 #1 # TODO: what does this mean?
        class ranges:
            ratio = 0.5 #ratio of sample 0 
            lin_vel_x = [0.5, 0.5] # min max [m/s] 0.2/0.8
            lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0] #or [-pi,pi] when yaw=0
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 55.,'waist': 200.,'shoulder': 300.,'elbow': 250.,'wrist_angle': 50.,'wrist_rotate': 30.,'left_finger': 50.,'right_finger': 50.}  # [N*m/rad]
        # damping = {'joint': 0.6,'waist': 0.,'shoulder': 0.,'elbow': 0.,'wrist_angle': 0.,'wrist_rotate': 1.,'left_finger': 5.,'right_finger': 5.}     # [N*m*s/rad]
        stiffness = {'joint': 45.} #  [N*m/rad]
        damping = {'joint': 0.8} #0.6}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = torch.ones(12)*0.4
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1wx/urdf/a1wx_nc.urdf'
        # name = "a1wx"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        # hip_name = 'hip'
        
        penalize_contacts_on = []
        # terminate_after_contacts_on = ["base","wx200/left_finger_link"]
        terminate_after_contacts_on = ["base"]
        terminate_condition = {}
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # fix_base_link = True
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.97
        base_height_target = 0.25
        # exponential_decay = ["tracking_ang_vel","total_work","action_norm"]
        exponential_decay = []
        max_iter = 1000
        # arm_weight = [2., 0.8, 0.8, 0.8, 0.8, 0.8,0.8]
        work_weight = [1., 1., 1., 1, 1, 1,1]
        dog_mask = [1,1,1, 1,1,1, 1,1,1, 1,1,1]
        
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.000
            dof_pos_limits = -0.0
            feet_air_time =  0.0
            termination = -5.0
            tracking_lin_vel = 7.0
            tracking_ang_vel = -1
            high_vel = -2
            lin_vel_z = -0.0
            # lin_vel_y = -5.0
            ang_vel_xy = -0.0
            orientation = 1.
            dof_vel = -0.
            dof_acc = -0
            # base_height = -3
            collision = -0.0
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            dog_work = -0.004
            dog_action_norm = -0.004
            slip = -0.6
            # arm_work = -0.002
            # arm_action_norm = -0.003

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 10.
        clip_actions = 100.
        
    class noise:
        add_noise = False
        noise_level = 0.2# scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.01
            lin_vel = 0.01
            ang_vel = 0.02
            gravity = 0.01
            height_measurements = 0.1
            
    class domain_rand:
        curriculum = []
        randomize_friction = False
        friction_range = [-0.1, 0.1]
        randomize_base_mass = False
        added_mass_range = [-2, 2]
        push_robots = True
        start_push_iter = 0
        push_interval_s = 4
        push_duration_s = 2
        push_interval_range = [1,1]
        push_duration_range = [0.5,1]
        min_push_vel_xy = [80.,80]
        max_push_vel_xy = [110.,110]
        offset = [[-0.12,0.12],[-0.12,0.12],[-0.12,0.12]]
        impulse = False
        # robot_frame = True
        force_only=True
        
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.05 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = False
        static_friction = 0.5
        dynamic_friction = 0.5
        restitution = 0.
        # rough terrain only:
        rough_plane = False
        roughness = 0.025
        
        slope = [0.15,0.5]
        slope_portion = 0. #this determines how many slopes to generate
        slope_roughness = 0.01
        
        rough_stair = False
        stair_size = [1.5,0.35]
        
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 1 #5 # number of terrain rows (levels)
        num_cols = 1 #10 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, plane, gaps]
        terrain_proportions =  [0., 0., 0., 0., 0., 1.1, 0.] # [0., 0.2, 0.25, 0.25, 0.2, 0.2, 0.4]
        # trimesh only:
        slope_treshold = 0. #0.75 # slopes above this threshold will be corrected to vertical surfaces
        
        # mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        # measure_heights = False
class A1WXVRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.5
        # init_noise_std = 0.1
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'lrelu'
        
        RMA = True
        RMA_hidden_dims = [256,256]
        privilege_obs_dim = A1WXVRoughCfg.env.num_pri_obs
        num_hist_obs = A1WXVRoughCfg.env.num_hist_obs
        num_privilege_obs = privilege_obs_dim*num_hist_obs
        num_latent = 32
        
        # estimator=True
        estimator = False
        
        decouple = False
        dog = True
        dog_policy_path = "/home/ravenhuang/locomani/locomani/legged_gym/logs/a1wx_vision/Mar23_18-02-20_arm_small_push/model_30000.pt"
        
        teacher_policy = False
        teacher_policy_path = ""
        
        start_arm = 0
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        adaptive_weight_teacher = 0.9991
        adaptive_weight_alpha = 0.9991
        adaptive_weight_beta = 0.998
        start_teacher = 0.8
        end_teacher = 0.3
        start_alpha = 0.3
        end_alpha = 0.3
        start_beta = 0.5
        end_beta = 0.05
        cmd = True #this will only us bc when cmd is nonzero for prior policy, ppo, hardcoded
        
    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 24
        load_run = "Mar23_18-02-20_arm_small_push"
        resume = True
        reset_std = True
        load_optimizer = False
        checkpoint = -1
        max_iterations = 10000 # number of policy updates
        experiment_name = 'a1wx_vision'
        run_name = 'arm_small_push'
        

  