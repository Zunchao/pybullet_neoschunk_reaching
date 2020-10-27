'''
python stable_baselines_scripts/zoo/mmtrain.py --algo ppo2 --env NeobotixSchunkEnv-v0 --tensorboard-log ~/results_server/stable-baselines/ -optimize --n-trials 1000 --n-jobs 2 --sampler random --pruner median
'''
import os
import inspect
import warnings

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from env.neobotixschunkGymEnv import NeobotixSchunkGymEnv

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import optuna

from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv

ENVIRONMENT = NeobotixSchunkGymEnv(renders=False,
                                   is_discrete=False,
                                   reward_type='rdense',
                                   action_repeat=1,
                                   enable_self_collision_flag=True,
                                   max_steps=500,
                                   action_dim=6,
                                   ws_boundary=1,
                                   random_initial=1,
                                   if_obstacle=0,
                                   if_obstacle_moving=0)


def optimize_ppo2(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_ppo2(trial)
    envs = SubprocVecEnv([ENVIRONMENT for i in range(1)])
    env = make_vec_env(lambda: envs, n_envs=1, seed=0)

    model = PPO2('MlpPolicy', env, verbose=0, nminibatches=1, **model_params)
    model.learn(10000)
    mean_reward, _ = evaluate_policy(model, ENVIRONMENT, n_eval_episodes=10)

    return -1 * mean_reward


if __name__ == '__main__':
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=4)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')