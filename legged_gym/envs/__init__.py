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

from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .a1.a1wxv_config import A1WXVRoughCfg, A1WXVRoughCfgPPO
from .a1.legged_robot_arm_vision_prior import LeggedRobotVision

from .a1_game.high_level_game import HighLevelGame
from .a1_game.dec_high_level_game import DecHighLevelGame
from .a1_game.low_level_game import LowLevelGame
from .a1_game.low_level_game_config import LowLevelGameCfg, LowLevelGamePPO
from .a1_game.high_level_game_flat_config import HighLevelGameFlatCfg, HighLevelGameFlatCfgPPO
from .a1_game.dec_high_level_game_config import DecHighLevelGameCfg, DecHighLevelGameCfgPPO
from .a1_game.rma_dec_high_level_game_config import RMADecHighLevelGameCfg, RMADecHighLevelGameCfgPPO
from .a1_game.dec_high_level_game_obstacles_config import DecHighLevelGameObstaclesCfg, DecHighLevelGameObstaclesCfgPPO

from legged_gym.utils.task_registry import  task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "a1wxv_rough", LeggedRobotVision, A1WXVRoughCfg(), A1WXVRoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "low_level_game", LowLevelGame, LowLevelGameCfg(), LowLevelGamePPO() )
task_registry.register( "high_level_game", HighLevelGame, HighLevelGameFlatCfg(), HighLevelGameFlatCfgPPO() )
task_registry.register( "dec_high_level_game", DecHighLevelGame, DecHighLevelGameCfg(), DecHighLevelGameCfgPPO() )
task_registry.register( "dec_high_level_game_obstacles", DecHighLevelGame, DecHighLevelGameObstaclesCfg(), DecHighLevelGameObstaclesCfgPPO() )
task_registry.register( "rma_dec_high_level_game", DecHighLevelGame, RMADecHighLevelGameCfg(), RMADecHighLevelGameCfgPPO() )