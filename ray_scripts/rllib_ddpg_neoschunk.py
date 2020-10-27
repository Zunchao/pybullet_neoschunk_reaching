'''
developed by Z. Zheng, @KIT-IPR

python -m ray_scripts.rllib_ddpg_neoschunk train

python -m ray_scripts.rllib_ppo_neoschunk rollout .../checkpoint_xx/checkpoint-xx --run PPO --env env_mp500lwa4dpg70 --steps 1000 --out ~/ray_results/rollout.pkl

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


def env_input_config(train_or_rollout):
    envInputs = {
        'urdfRoot': parentdir,
        'renders': False,
        'isDiscrete': False,
        'actionDim': 10,
        'rewardType': 'rdense',
        'randomInitial': True,
        'actionRepeat': 1,
        'isEnableSelfCollision': True,
        'maxSteps': 2e3,
        'wsBoundary': 1,
        'colliObj': True,
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
            'run': 'DDPG',
            'config': config,
            'checkpoint_freq': 2000,
            #'local_dir': "~/train_results",
            'stop':{
                "timesteps_total": 1e10,
            },
        }
    }

    if options.command == "train":
        ray.init(object_store_memory=int(5e9))
        configFromYaml['train-ppo']['config']['env_config'] = env_input_config(True)
        run_experiments(configFromYaml)
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
        'twin_q': True,
        'policy_delay': 2,
        'smooth_target_policy': True,

        'actor_hiddens': [256, 128],
        'critic_hiddens': [256, 128],

        #'n_step': 1,
        'gamma': 0.999,
        "num_gpus": 0,
        "sample_batch_size": 20,
        "per_worker_exploration": True,
        #"min_iter_time_s": 30,
        # === Exploration ===
        #'schedule_max_timesteps': 100000,
        "target_network_update_freq": 3000,
        "timesteps_per_iteration": 1000,
        'tau': 0.001,

        # === Replay buffer ===
        'buffer_size': 200000,
        'prioritized_replay': True,
        #"parameter_noise": True,

        # === Optimization ===
        'use_huber': True,
        'huber_threshold': 1.0,
        'l2_reg': 1e-5,
        'learning_starts': 1000,
        #'sample_batch_size': 20,
        'train_batch_size': 32,
        "critic_lr": 5e-5,
        # Learning rate for the actor (policy) optimizer.
        "actor_lr": 5e-5,

        # === Parallelism ===
        'num_workers': 16,
        #'num_gpus_per_worker': 0,
        #'per_worker_exploration': False,
        "worker_side_prioritization": True,

        # === Evaluation ===
    }
    train_agent(config_to_use)