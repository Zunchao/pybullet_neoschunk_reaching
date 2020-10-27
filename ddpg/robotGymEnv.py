"""
2D mobile robot reaching
built on 17.06.2019
by Zunchao
"""

import numpy as np
import pyglet
import gym
from gym import spaces, logger
from gym.utils import seeding
import time
largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 720
pyglet.clock.set_fps_limit(10000)
THRESHHOLD = 0.1
VIEWBOUND = 1.2
VIEWLINE = 1.06
CARLENGTH = 0.12
CARWIDTH = 0.08
LINKLENGTH = 0.12
LINKWIDTH = 0.03

class RobotGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self, maxstep=1e2, space=1):
        self._isDiscrete = False
        self._maxSteps = maxstep
        self.stepsCounter = 0
        self.episodeCount = 0
        self.successCount = 0
        self._action_dim = 2
        self.dt = 0.1
        self.spacemax = space
        self.spacemin = -space
        self._observation = []
        self._dis = 0
        self._goal = []
        self.r_penalty = 0
        self.joint_state = []

        self._terminated = False

        self.robot_view = [0.0, 0.0, 0.0, 20.0, 40.0]  # robot in viewer (x, y, r, w, l)
        self.robot_state = np.zeros(5)  # robot state [px, py, vx, vy, w]
        self.viewer = None
        self.seed()
        self.reset()
        daction = 1
        if not self._isDiscrete:
            self.action_bound = np.ones(self._action_dim) * daction
            self.action_space = spaces.Box(low=-self.action_bound, high=self.action_bound, dtype=np.float32)
        
        self.observation_dim = len(self.get_observation())
        observation_high = np.array([largeValObservation] * self.observation_dim)
        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float32)

    def reset(self):
        self._terminated = False
        self.stepsCounter = 0
        self.r_penalty = 0
        self.robot_state[0:2] = np.random.uniform(self.spacemin, self.spacemax, 2)
        self.robot_state[2:4] = np.zeros(2)
        self.robot_state[4] = np.random.uniform(-np.pi, np.pi)
        self.robot_state = np.zeros(5)
        self.robot_view[0:2] = self.robot_state[0:2]
        self.robot_view[2] = self.robot_state[4]
        self.joint_state = np.zeros(3)

        self._goal = np.random.uniform(self.spacemin, self.spacemax, 2)
        self._dis = np.linalg.norm(self._goal - self.robot_state[0:2])

        self._observation = self.get_observation()

        return np.array(self._observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_observation(self):
        observation = []
        #observation[0:2] = self.robot_state[0:2]#np.subtract(self._goal, self.robot_state[0:2])
        observation[0:2] = np.subtract(self._goal, self.robot_state[0:2])
        observation[2:5] = self.robot_state[2:5]
        #observation[5:7] = self._goal
        self._observation = observation
        return self._observation

    def step(self, action):
        # action = [^v, ^w]
        action = action
        self.stepsCounter += 1
        self.robot_state[4] = self.check_w(self.robot_state[4], action[1]*self.dt)
        self.robot_state[2:4] = self.check_v(self.robot_state[2:4], action[0]*self.dt * np.array([np.cos(self.robot_state[4]), np.sin(self.robot_state[4])]))
        self.robot_state[0:2] = self.check_boundary(self.robot_state[0:2], self.robot_state[2:4]*self.dt)
        self.robot_view[0:2] = self.robot_state[0:2]
        self.robot_view[2] = self.robot_state[4]
        done = self.terminator()
        if done:
            self.episodeCount += 1
        # observation = self.get_observation()
        reward = self.reward()
        return np.array(self._observation), reward, self._terminated, {}
    
    def reward(self):
        reward = np.exp(-self._dis**2) + self.r_penalty #- self.stepsCounter
        return reward
    
    def terminator(self):
        observations = self.get_observation()
        relative_dis = np.subtract(self._goal, self.robot_state[0:2])#self.get_observation()
        self._dis = np.linalg.norm(relative_dis)
        if self.stepsCounter >= self._maxSteps-1:
            self.r_penalty = 100
            return True        
        if self._dis < THRESHHOLD:
            print('Reaching : ', self._dis, self._goal, self.robot_state)
            self._terminated = True
            self.r_penalty = 0
            self.successCount += 1
            return True        
        return False

    def check_boundary(self, p, dp):
        p += dp
        if np.abs(p[0]) > 1:
            p[0] -= dp[0]
        if np.abs(p[1]) > 1:
            p[1] -= dp[1]
        return p

    def check_w(self, w, dw):
        w += dw
        if np.abs(w) > np.pi:
            w -= dw
        return w

    def check_v(self, v, dv):
        v += dv
        if np.abs(v[0]) > 1:
            v[0] -= dv[0]
        if np.abs(v[1]) > 1:
            v[1] -= dv[1]
        return v

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(RENDER_HEIGHT, RENDER_WIDTH)
            self.viewer.set_bounds(-VIEWBOUND, VIEWBOUND, -VIEWBOUND, VIEWBOUND)
            self.viewer.add_geom(rendering.Line((-VIEWLINE, VIEWLINE), (-VIEWLINE, -VIEWLINE)))
            self.viewer.add_geom(rendering.Line((VIEWLINE, VIEWLINE), (VIEWLINE, -VIEWLINE)))
            self.viewer.add_geom(rendering.Line((VIEWLINE, -VIEWLINE), (-VIEWLINE, -VIEWLINE)))
            self.viewer.add_geom(rendering.Line((VIEWLINE, VIEWLINE), (-VIEWLINE, VIEWLINE)))
            # draw goal point
            rgoal = self.viewer.draw_circle(0.025)
            rgoal.set_color(0.8, 0.1, 0.1)
            self.viewer.add_geom(rgoal)
            self.goaltrans = rendering.Transform(translation=self._goal)
            rgoal.add_attr(self.goaltrans)
            # draw reaching bounding box
            rcircle = self.viewer.draw_circle(THRESHHOLD, filled=False)
            rcircle.set_color(0.1, 0.1, 0.1)
            self.viewer.add_geom(rcircle)
            self.circletrans = rendering.Transform(translation=self._goal)
            rcircle.add_attr(self.circletrans)
            # draw mobile robot
            l, w = CARLENGTH/2, CARWIDTH/2
            rpolygon = ((-l, w), (l, w), (l, -w), (-l, -w))
            rrobot = rendering.FilledPolygon(rpolygon)
            rrobot.set_color(0.1, 0.1, 0.8)
            self.viewer.add_geom(rrobot)
            self.cartrans = rendering.Transform()
            rrobot.add_attr(self.cartrans)
        # self.goaltrans.set_translation(self._goal[0], self._goal[1])
        # self.circletrans.set_translation(self._goal[0], self._goal[1])
        self.cartrans.set_translation(self.robot_view[0], self.robot_view[1])
        self.cartrans.set_rotation(self.robot_view[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def sample_action(self):
        if not self._isDiscrete:
            a = np.random.uniform(-1, 1, size=self._action_dim)
        return a

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)


if __name__ == '__main__':
    #np.random.seed(11)
    env = RobotGymEnv(maxstep=5e2, space=1)
    env.set_fps(30)
    for ep in range(1):
        s = env.reset()
        #print(env.transform_joint_position([0,0,0],[1,2,1]))
        #print('reset')
        for _ in range(500):
            env.render()
            time.sleep(0.1)
            a = env.sample_action()
            #a = [1,1]
            s, r, done, info = env.step(a)
            # print((s,r,done,info))
            if done:
                break
        env.close()