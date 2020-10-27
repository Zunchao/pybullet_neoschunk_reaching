'''
developed by Z. Zheng, @KIT-IPR

python3 -m ddpg.rllib_train_2drobot train

python3 -m ddpg.rllib_train_2drobot rollout .../checkpoint_xx/checkpoint-xx --run PPO --env env_mp500lwa4dpg70 --steps 1000 --out ~/ray_results/rollout.pkl

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import inspect
import random
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from ddpg.mmRobotGymEnv import MMRobotGymEnv
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from ray.rllib import train
from ray.rllib import rollout

MAX_EP_STEPS = 1000
DSPACE = 10

def env_input_config(train_or_rollout):
    envInputs = {
        'maxstep': MAX_EP_STEPS,
        'space': DSPACE,
    }
    #if not train_or_rollout:
        #envInputs['renders'] = True
    return envInputs

def env_creator(env_config):
    env = MMRobotGymEnv(**env_config)
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

    register_env('env_2drobot', env_creator)
    configFromYaml = {
        'train-ppo':{
            'env': 'env_2drobot',
            'run': 'PPO',
            'config': config,
            'checkpoint_freq': 1000,
            #'local_dir': "~/train_results",
            'stop':{
                "timesteps_total": 1e7,
            },
        }
    }

    if options.command == "train":
        ray.init()
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
        "use_gae": True,
        "lambda": 0.995,
        "num_workers": 2,
        "num_gpus": 0,
        "monitor": False,
        # "lambda": lambda: random.uniform(0.9, 1.0),
        "kl_coeff": 100,
        "kl_target": 0.02,
        "sample_batch_size": 200,
        "train_batch_size": 400,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 30,
        "lr": 5e-5,
        "vf_loss_coeff": 1.0,
        "vf_share_layers": True,
        "clip_param": 0.2,
        "vf_clip_param": 1,
        "simple_optimizer": False,
        # Whether to rollout "complete_episodes" or "truncate_episodes"
        "batch_mode": "truncate_episodes",
        # "synchronize_filters": True,
        "model":
            {
                "fcnet_activation": "tanh",
                "fcnet_hiddens": [64, 64],
            },
    }
    train_agent(config_to_use)