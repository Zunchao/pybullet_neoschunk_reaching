# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example configurations using the PPO algorithm."""

import functools

from agent import ppo
from agent import networks
import tensorflow as tf

from env.neobotixschunkGymEnv import NeobotixSchunkGymEnv

def default():
  """Default configuration for PPO."""
  # General
  algorithm = ppo.PPOAlgorithm
  num_agents = 30
  eval_episodes = 30
  use_gpu = True
  # Network
  network = networks.feed_forward_gaussian
  weight_summaries = dict(all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
  policy_layers = 256, 128
  value_layers = 256, 128
  init_mean_factor = 0.1
  init_logstd = -1
  # Optimization
  update_every = 30
  update_epochs = 25
  optimizer = tf.train.AdamOptimizer
  update_epochs_policy = 64
  update_epochs_value = 64
  learning_rate = 5e-5
  # Losses
  discount = 0.99
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 200
  kl_init_penalty = 1
  return locals()

def pybullet_neoschunk_reaching():
  """Configuration for Bullet neo+schunk mm reaching task."""
  locals().update(default())
  env = functools.partial(NeobotixSchunkGymEnv,
                          renders=False,
                          is_discrete=False,
                          if_prioritized=False,
                          reward_type='rdense',
                          action_repeat=1,
                          enable_self_collision_flag=True,
                          max_steps=500,
                          action_dim=6,
                          ws_boundary=1,
                          random_initial=True,
                          if_obstacle=False,
                          if_obstacle_moving=False,
                          if_goal_moving_type='static')
  # Environment
  max_length = 500
  steps = 2e8  # 100M
  return locals()
