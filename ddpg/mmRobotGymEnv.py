"""
2D mobile manipulation reaching
mobile box with 3-link arm
built on 21.06.2019
by Zunchao
"""

import numpy as np
import pyglet
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers import Monitor
import time
import csv
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 720
#pyglet.clock.set_fps_limit(10000)
THRESHOLD = 0.2#*1e1
VIEWBOUND = 1.2*2
VIEWLINE = 1.06*2
CARLENGTH = 0.16
CARWIDTH = 0.1
LINKLENGTH = 0.12
LINKWIDTH = 0.03


class MMRobotGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, maxstep=1e2, space=1):
        self._isDiscrete = False
        self._maxSteps = maxstep
        self.stepsCounter = 0
        self.episodeCount = 0
        self.successCount = 0
        self._action_dim = 5
        self._observation_dim = 10
        self._state_dim = 8
        self.dt = 0.1
        self.spacemax = space
        self.spacemin = -space
        self._observation = []
        self._dis_ee = 0
        self._dis_base = 0
        self._dis_init = 0
        self._goal = []
        self.r_penalty = 0
        self.joint_state = []
        self._action = []

        self.linklength = [LINKLENGTH, LINKLENGTH, LINKLENGTH]

        self._terminated = False
        # robot in viewer (w, l)
        self.robot_view = [20.0, 40.0]
        # robot state [pbx, pby, vbx, vby, bw, j1, j2, j3]
        # base position, base velocity, joint positions
        self.robot_state = np.zeros(self._state_dim)
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
        #self.seed()
        #while True:
        self.robot_state[0:2] = np.random.uniform(self.spacemin, self.spacemax, 2)
        self.robot_state[2:4] = np.zeros(2)
        self.robot_state[4:8] = np.random.uniform(-np.pi, np.pi)
        Tpos0 = self.transform_joint_position(self.robot_state[5:8], self.linklength, self.robot_state[0:2], self.robot_state[4])
        #if not self.check_boundary(Tpos0):
            #break
        # self.robot_state = np.zeros(self._state_dim)
        # self.robot_state[4] = -np.pi/2
        self.joint_state = self.robot_state[5:8]

        self._goal = np.random.uniform(self.spacemin, self.spacemax, 2)
        #self._goal = [2,2]
        self._observation = self.get_observation()
        self._dis_ee = np.linalg.norm(self._observation[-2:])
        self._dis_init = self._dis_ee
        self._dis_base = np.linalg.norm(self._observation[0:2])
        self.datawriterfile = csv.writer(
            open(os.path.join(parentdir, 'neobotix_schunk_pybullet/results/robot2d1.csv'), "a"))

        return np.array(self._observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_observation(self):
        observation = []# [^pbx, ^pby, vbx, vby, bw, j1, j2, j3, ^pex, ^pey]
        Tpos = self.transform_joint_position(self.robot_state[5:8], self.linklength, self.robot_state[0:2],
                                             self.robot_state[4])
        # observation[0:2] = np.subtract(self._goal, Tpos[4:6])
        observation[0:8] = self.robot_state
        observation[0:2] = self._goal - self.robot_state[0:2]
        observation[8:10] = self._goal - Tpos[4:6]

        # observation[10:16] = Tpos
        # observation[5:7] = self._goal
        self._observation = observation
        return self._observation

    def step(self, action):
        # action = [^j1, ^j2, ^j3, ^v, ^w]
        # joint state difference, base vel and ang difference
        self._action = action
        self.stepsCounter += 1
        # check the base angular, base vel, base position and arm position on this base config
        wbasenew, wbaseold = self.check_w([self.robot_state[4]], [x * self.dt for x in [action[4]]])
        self.robot_state[4] = wbasenew[0]
        vbasenew, vbaseold = self.check_v(self.robot_state[2:4], action[3] * self.dt * np.array(
            [np.cos(self.robot_state[4]), np.sin(self.robot_state[4])]))
        self.robot_state[2:4] = vbasenew
        self.robot_state[0:2] += self.robot_state[2:4]*self.dt
        #Tpos0 = self.transform_joint_position(self.robot_state[5:8], self.linklength, self.robot_state[0:2], self.robot_state[4])
        if self.check_boundary(self.robot_state[0:2]):# or self.check_boundary(Tpos0):
            self.robot_state[0:2] -= self.robot_state[2:4]*self.dt
            self.r_penalty = -10#*np.linalg.norm(self.robot_state[0:2])
            self.robot_state[2:4] = -vbasenew#np.zeros(2)#vbaseold
            #self.robot_state[4] = wbaseold[0]
            # check tha arm joint position
        jointstatesnew, jointstatesold = self.check_w(self.robot_state[5:8], [x * self.dt for x in action[0:3]])
        self.robot_state[5:8] = jointstatesnew
        #Tpos1 = self.transform_joint_position(self.robot_state[5:8], self.linklength, self.robot_state[0:2], self.robot_state[4])
        #if self.check_boundary(Tpos1):
            #self.robot_state[5:8] = jointstatesold

        done = self.terminator()
        if done:
            self.episodeCount += 1
            self.datawriterfile.writerow([self.successCount, self.episodeCount, self.successCount/self.episodeCount])
            print('success rate : ', self.successCount,self.episodeCount,self.successCount/self.episodeCount)
        # observation = self.get_observation()
        reward = self.reward()
        return np.array(self._observation), reward, self._terminated, {}

    def terminator(self):
        observations = self.get_observation()
        # relative_dis = np.subtract(self._goal, observations[-2:])  # self.get_observation()
        self._dis_ee = np.linalg.norm(observations[-2:])
        self._dis_base = np.linalg.norm(observations[0:2])
        if self._terminated:
            return True
        if self.stepsCounter >= self._maxSteps - 1:
            self._terminated = True
            self.r_penalty = 0
            return True
        if self._dis_ee < THRESHOLD:
            print('Reaching : ', self._dis_ee, self._goal, self.robot_state)
            self._terminated = True
            self.r_penalty = 100
            self.successCount += 1
            return True
        return False

    def reward(self):
        tau = 1-self._dis_ee/self._dis_init
        tau = np.cbrt(tau)
        ree = -np.log(self._dis_ee)
        rbase = -np.log(self._dis_base)
        if self._dis_ee < 0.5:
            reward = ree
        else:
            if tau < 1:
                reward = (1 - tau) * ree + tau * rbase
                # penalty = 0
            else:
                reward = rbase
                #penalty = 2*(tau-1)*self._dis_ee
        #reward = -(1-tau)*self._dis_ee - tau*self._dis_base + self.r_penalty - self.stepsCounter/self._maxSteps - penalty
        reward = ree + self.r_penalty - self.stepsCounter/self._maxSteps# + np.random.normal(ree, self._dis_ee/100)#+ self.r_penalty - self.stepsCounter/self._maxSteps #- np.linalg.norm(self._action)+ np.random.normal(0, 1/self.stepsCounter/5)
        #reward = reward + self.r_penalty - self.stepsCounter/self._maxSteps
        return reward

    # check if robot reaches the boundary
    def check_boundary(self, p):
        for i in range(len(p)):
            if np.abs(p[i]) > self.spacemax:
                return True
        return False
    # check if robot angular or joint angles in the range
    def check_w(self, w, dw):
        w0 = w
        assert len(w) == len(dw)
        for i in range(len(w)):
            if np.abs(w[i]+dw[i]) > np.pi:
                w[i] = w[i]
            else:
                w[i] += dw[i]
        return w, w0
    # check robot vel
    def check_v(self, v, dv):
        assert len(v) == len(dv)
        v0 = v
        v += dv
        if np.linalg.norm(v) > 1:
            v -= dv
        return v, v0

    def transform_joint_position(self, a, l, pbase, wbase):
        assert len(a) == 3
        p1 = np.mat([l[0]*np.cos(a[0]), l[0]*np.sin(a[0]), 1])
        p2 = np.mat([l[0]*np.cos(a[0]) + l[1]*np.cos(a[0] + a[1]), l[0]*np.sin(a[0]) + l[1]*np.sin(a[0] + a[1]), 1])
        p3 = np.mat([l[0]*np.cos(a[0]) + l[1]*np.cos(a[0] + a[1]) + l[2]*np.cos(a[0] + a[1] + a[2]),
                     l[0]*np.sin(a[0]) + l[1]*np.sin(a[0] + a[1]) + l[2]*np.sin(a[0] + a[1] + a[2]), 1])

        m0 = np.mat([[np.cos(wbase), -np.sin(wbase), pbase[0]], [np.sin(wbase), np.cos(wbase), pbase[1]], [0, 0, 1]])

        t1 = np.matmul(m0, p1[0].transpose()).transpose()
        t2 = np.matmul(m0, p2[0].transpose()).transpose()
        t3 = np.matmul(m0, p3[0].transpose()).transpose()
        t = np.array([t1[0, 0:2], t2[0, 0:2], t3[0, 0:2]])
        t = t.reshape(1, 6)[0, 0:6]

        return t

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
            rcircle = self.viewer.draw_circle(THRESHOLD, filled=False)
            rcircle.set_color(0.1, 0.1, 0.1)
            self.viewer.add_geom(rcircle)
            self.circletrans = rendering.Transform(translation=self._goal)
            rcircle.add_attr(self.circletrans)
            # draw mobile robot
            l, w = CARLENGTH / 2, CARWIDTH / 2
            rpolygon = ((-l, w), (l, w), (l, -w), (-l, -w))
            rrobot = rendering.FilledPolygon(rpolygon)
            rrobot.set_color(0.2, 0.2, 0.7)
            self.viewer.add_geom(rrobot)
            self.cartrans = rendering.Transform()
            rrobot.add_attr(self.cartrans)
            # build arm
            theta = [self.joint_state[0], self.joint_state[0] + self.joint_state[1],
                     self.joint_state[0] + self.joint_state[1] + self.joint_state[2]]
            # draw arm link0
            link0 = rendering.make_capsule(LINKLENGTH, LINKWIDTH)
            link0.set_color(0.3, 0.5, 0.6)
            self.viewer.add_geom(link0)
            self.link0trans = rendering.Transform(translation=(0, 0.0), rotation=theta[0])
            link0.add_attr(self.link0trans)
            link0.add_attr(self.cartrans)
            # draw arm link1
            link1 = rendering.make_capsule(LINKLENGTH, LINKWIDTH)
            link1.set_color(0.8, 0.4, 0.4)
            self.viewer.add_geom(link1)
            self.link1trans = rendering.Transform(translation=(LINKLENGTH, 0), rotation=theta[1])
            link1.add_attr(self.link1trans)
            link1.add_attr(self.link0trans)
            link1.add_attr(self.cartrans)
            # draw arm link2
            link2 = rendering.make_capsule(LINKLENGTH, LINKWIDTH)
            link2.set_color(0.3, 0.5, 0.6)
            self.viewer.add_geom(link2)
            self.link2trans = rendering.Transform(translation=(LINKLENGTH, 0.0), rotation=theta[2])
            link2.add_attr(self.link2trans)
            link2.add_attr(self.link1trans)
            link2.add_attr(self.link0trans)
            link2.add_attr(self.cartrans)
        # self.goaltrans.set_translation(self._goal[0], self._goal[1])
        # self.circletrans.set_translation(self._goal[0], self._goal[1])
        self.link0trans.set_rotation(self.robot_state[5])
        self.link1trans.set_rotation(self.robot_state[6])
        self.link2trans.set_rotation(self.robot_state[7])
        self.cartrans.set_translation(self.robot_state[0], self.robot_state[1])
        self.cartrans.set_rotation(self.robot_state[4])


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def sample_action(self):
        if not self._isDiscrete:
            a = np.random.uniform(-1, 1, size=self._action_dim)
        return a

    #def set_fps(self, fps=30):
        #pyglet.clock.set_fps_limit(fps)


if __name__ == '__main__':
    # np.random.seed(11)
    env = MMRobotGymEnv(maxstep=2e2, space=2)
    rec = VideoRecorder(env, path='/home/zheng/fromremote/video/a.mp4')
    #mon = Monitor(env, '/home/zheng/fromremote/video/video',video_callable=lambda episode_id: True,force = True)
    #env.set_fps(30)
    for ep in range(1):
        s = env.reset()

        # print(env.transform_joint_position([0,0,0],[1,2,1]))
        # print('reset')
        for i in range(200):
            env.render()
            env.unwrapped.render()
            rec.capture_frame()
            assert not rec.empty
            assert not rec.broken
            assert os.path.exists(rec.path)
            f = open(rec.path)
            #time.sleep(0.02)
            a = env.sample_action()
            #a = [0, 0, 0, 1, 1]
            s, r, done, info = env.step(a)
            print(i, s, r, done, info)
            if done:
                break
        rec.close()
        env.close()