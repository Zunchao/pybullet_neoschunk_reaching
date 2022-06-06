'''
developed by Z. Zheng, @KIT-IPR
python -m ray_scripts.rllib_ppo_neobotixschunk train
python -m ray_scripts.rllib_ppo_neobotixschunk rollout .../checkpoint_xx/checkpoint-xx --run PPO --env env_mp500lwa4dpg70 --steps 1000 --out ~/ray_results/rollout.pkl
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import inspect
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from env.neobotixschunkGymEnv import NeobotixSchunkGymEnv

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from ray.rllib import train
from ray.rllib import rollout
from ray import tune
from ray.tune import grid_search
from ray.rllib.agents.ppo import PPOTrainer


def env_input_config(train_or_rollout):
    envInputs = {
        'renders': False,
        'is_discrete': False,
        'if_prioritized': False,
        'reward_type': 'rdense',
        'action_repeat': 1,
        'enable_self_collision_flag': True,
        'max_steps': 500,
        'action_dim': 6,
        'ws_boundary': 1,
        'random_initial': True,
        'if_obstacle': True,
        'if_obstacle_moving': False,
        'if_goal_moving_type': 'static'
    }
    if not train_or_rollout:
        envInputs['renders'] = True
    return envInputs

def env_creator(env_config):
    env = NeobotixSchunkGymEnv(**env_config)
    return env  # return an env instance

def train_agent(config):
    parser = argparse.ArgumentParser(
        description="Train or Run an RLlib Agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subcommand_group = parser.add_subparsers(
        help="Commands to train or run an RLlib agent.", dest="command")
    train_parser = train.create_parser(
        lambda **kwargs: subcommand_group.add_parser("train", **kwargs))
    rollout_parser = rollout.create_parser(
        lambda **kwargs: subcommand_group.add_parser("rollout", **kwargs))
    options = parser.parse_args()

    register_env('env_mp500lwa4dpg70', env_creator)
    configFromYaml = {
        'train-ppo':{
            'env': 'env_mp500lwa4dpg70',
            'run': 'PPO',
            'config': config,
            'checkpoint_freq': 2000,
            #'local_dir': "~/train_results",
            'stop':{
                "timesteps_total": 1e9,
                #'training_iteration': 100000,
            },
        }
    }
    configTune = {
            "env": 'env_mp500lwa4dpg70',
            'config': config,
    }
    if options.command == "train":
        ray.init(object_store_memory=int(5e9))
        configFromYaml['train-ppo']['config']['env_config'] = env_input_config(True)
        configTune['config']['env_config'] = env_input_config(True)
        tune.run(PPOTrainer,
                 stop={
                     "timesteps_total": 1e8,
                 },
                 config={
                     "env": 'env_mp500lwa4dpg70',
                     "env_config": env_input_config(True)
                 })
        #run_experiments(configFromYaml)
        #train.run(options, train_parser)
    elif options.command == "rollout":
        options.run = configFromYaml['train-ppo']['run']
        options.env = configFromYaml['train-ppo']['env']
        options.no_render = True
        options.steps = 10000
        options.out = None
        configFromYaml['train-ppo']['config']['env_config'] = env_input_config(False)
        configFromYaml['train-ppo']['config']['monitor'] = True
        options.config = configFromYaml['train-ppo']['config']
        rollout.run(options, rollout_parser)
    else:
        parser.print_help()


if __name__ == '__main__':
    # print('default common', COMMON_CONFIG)
    # print('default model', MODEL_DEFAULTS)
    config_to_use = {
        "num_workers": 8,
        # Whether to rollout "complete_episodes" or "truncate_episodes"
        "batch_mode": "complete_episodes",
        "num_gpus": 0,
        "gamma": 0.99,
        "lr": grid_search([1e-4, 5e-5, 1e-6]),#5e-5,
        "monitor": False,
        "use_critic": True,
        "use_gae": True,
        "lambda": 0.995,
        "kl_coeff": 0.2,
        "kl_target": 0.01,
        "rollout_fragment_length": 200,
        "train_batch_size": 1600,
        "sgd_minibatch_size": 64,
        "num_sgd_iter": 16,

        "vf_share_layers": False,
        # tune this if set vf_share_layers: True.
        "vf_loss_coeff": 0.5,

        "clip_param": 0.2,
        "vf_clip_param": 500,

        "simple_optimizer": False,
        "entropy_coeff": 0.01,
        # "synchronize_filters": True,
        "model":
            {
                "fcnet_activation": "tanh",
                "fcnet_hiddens": [256, 256],
                'use_lstm': False,
            },
    }
    train_agent(config_to_use)