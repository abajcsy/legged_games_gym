# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.a1 import A1RoughCfg, A1RoughCfgPPO

class A1GameFlatCfg( A1RoughCfg ):
    class env ( A1RoughCfg.env ):
        # note:
        #   48 observations for nominal A1 setup
        #   + 187 for non-flat terrain observations
        #   + 2 for point-predator (x,y) state
        num_observations = 50
        num_actions = 12 + 2

    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class init_state (A1RoughCfg.init_state):
        rel_predator_pos = [0.0, 0.0]

    # class rewards( A1RoughCfg.rewards ):
    #
    #     class scales( A1RoughCfg.rewards.scales ):

class A1GameFlatCfgPPO( A1RoughCfg ):
    class runner( A1RoughCfg.runner ):
        run_name = ''
        experiment_name = 'flat_a1_game'