from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from env.neobotixschunkGymEnv import NeobotixSchunkGymEnv
#from env.neobotixschunkGymEnv_new import NeobotixSchunkGymEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

from stable_baselines.common.cmd_util import make_vec_env

# Instantiate the env
envs = NeobotixSchunkGymEnv(renders=False,
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
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
# It will check your custom environment and output additional warnings if needed
check_env(envs, warn=True)
# Hyperparameters for learning identity for each RL model

# wrap it
#env = make_vec_env(lambda: envs, n_envs=1)

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: envs])

model = PPO2(MlpPolicy, env, verbose=1, n_steps=256, nminibatches=256, lam=0.92, gamma=0.99, noptepochs=5,
             ent_coef=0.0, learning_rate=5e-5, cliprange=0.2)
model.learn(total_timesteps=int(1e6))
model.save(currentdir+'results/train_model')
obs = env.reset()
for i in range(1e4):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()

'''
LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", learning_rate=1e-3, n_steps=1,
                         gamma=0.7, env=e, seed=0).learn(total_timesteps=10000),
    'acer': lambda e: ACER(policy="MlpPolicy", env=e, seed=0,
                           n_steps=1, replay_ratio=1).learn(total_timesteps=15000),
    'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e, seed=0,
                             learning_rate=5e-4, n_steps=1).learn(total_timesteps=20000),
    'dqn': lambda e: DQN(policy="MlpPolicy", batch_size=16, gamma=0.1,
                         exploration_fraction=0.001, env=e, seed=0).learn(total_timesteps=40000),
    'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e, seed=0, lam=0.5,
                           optim_batchsize=16, optim_stepsize=1e-3).learn(total_timesteps=15000),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e, seed=0,
                           learning_rate=1.5e-3, lam=0.8).learn(total_timesteps=20000),
    'trpo': lambda e: TRPO(policy="MlpPolicy", env=e, seed=0,
                           max_kl=0.05, lam=0.7).learn(total_timesteps=10000),
}


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ['a2c', 'acer', 'acktr', 'dqn', 'ppo1', 'ppo2', 'trpo'])
def test_identity(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    :param model_name: (str) Name of the RL model
    """
    env = DummyVecEnv([lambda: envs])

    model = LEARN_FUNC_DICT[model_name](env)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)

    obs = env.reset()
    assert model.action_probability(obs).shape == (1, 10), "Error: action_probability not returning correct shape"
    action = env.action_space.sample()
    action_prob = model.action_probability(obs, actions=action)
    assert np.prod(action_prob.shape) == 1, "Error: not scalar probability"
    action_logprob = model.action_probability(obs, actions=action, logp=True)
    assert np.allclose(action_prob, np.exp(action_logprob)), (action_prob, action_logprob)

    # Free memory
    del model, env
'''

