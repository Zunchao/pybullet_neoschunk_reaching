"""
2D mobile manipulation reaching
mobile box with 3-link arm
built on 11.07.2019
based on mmRobotGymEnv.py
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
from ddpg.ddpg_noise import OUNoise, OrnsteinUhlenbeckProcess
from env import reachingRewards

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 720
#pyglet.clock.set_fps_limit(10000)
THRESHOLD = 0.05#*1e1
VIEWBOUND = 1.2
VIEWLINE = 1.06
CARLENGTH = 0.16
CARWIDTH = 0.1
LINKLENGTH = 0.12
LINKWIDTH = 0.03


class MM2DRobotCGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, maxstep=1e2, space=1, ifobs=False):
        self._isDiscrete = False
        self._maxSteps = maxstep
        self.stepsCounter = 0
        self.episodeCount = 0
        self.successCount = 0
        self._action_dim = 5
        self._observation_dim = 9
        self._state_dim = 8
        self.dt = 0.05
        self.spacemax = space
        self.spacemin = -space
        self._observation = []
        self._dis_ee = 0
        self._dis_base = 0
        self._dis_init = 0
        self._dis_vor = 0
        self._goal = np.zeros(2)
        self.r_penalty = 0
        self.joint_state = []
        self._action = []
        self.rboundary = 0
        self.var = 2
        self.has_obs = ifobs
        self.linklength = [LINKLENGTH, LINKLENGTH, LINKLENGTH]

        self._terminated = False
        # robot in viewer (w, l)
        self.robot_view = [20.0, 40.0]
        # robot state [pbx, pby, bv, btheta, bw, j1, j2, j3]
        # base position, base linear vel, base angular, base angular vel, joint positions
        self.robot_state = np.zeros(self._state_dim)
        self.pstart = []
        self.pee = []
        self.pobs = []
        self.plinks = []
        self._with_prio = False
        self.rfunc = reachingRewards.ReachingReward(with_priority=self._with_prio, goal=self._goal, armpos=self.pee,
                                                    basepos=self.pstart, opos=self.pobs)
        self.viewer = None
        self.seed(1)
        self.reset()
        daction = 1
        if not self._isDiscrete:
            self.action_bound = np.ones(self._action_dim) * daction
            self.action_space = spaces.Box(low=-self.action_bound, high=self.action_bound, dtype=np.float32)

        self.observation_dim = len(self.get_observation())
        #print('observation dim : ', self.observation_dim)
        observation_high = np.array([largeValObservation] * self.observation_dim)
        #self.datawriterfile = csv.writer(open(os.path.join(parentdir, 'neobotix_schunk_pybullet/results/ddpg2.csv'), "a"))
        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float32)

    def reset(self):
        self._terminated = False
        self.stepsCounter = 0
        self.r_penalty = 0
        self.pstart = []
        self.pee = []
        self.pobs = []
        self.plinks = []
        self.seed()

        self.robot_state[0:2] = np.random.uniform(self.spacemin, self.spacemax, 2)  # base position
        self.robot_state[2] = 0  # base vel
        self.robot_state[3] = np.random.uniform(-np.pi, np.pi)  # base angular
        self.robot_state[4] = 0  # base angular vel
        self.robot_state[5:8] = np.random.uniform(-np.pi, np.pi)  # joint angles
        # self.robot_state = np.zeros(self._state_dim)
        # self.robot_state[4] = -np.pi/2
        # self._goal = np.random.uniform(self.spacemin, self.spacemax, 2)
        self.joint_state = self.robot_state[5:8]
        self.pstart = self.robot_state[0:2]
        if not self.pstart[0]:
            self.pstart += 1e-9

        self.plinks = self.transform_joint_position(self.robot_state[5:8], self.linklength, self.robot_state[0:2],
                                                    self.robot_state[4])
        self.pee = self.plinks[-2:]
        if self.has_obs:
            while True:
                xopos = np.random.uniform(0, self.pee[0])
                yopos = np.random.uniform(0, self.pee[1])
                self.pobs = np.random.uniform(self.spacemin, self.spacemax, 2)#np.array([xopos, yopos])
                if np.linalg.norm(self.pobs)>0.1 and not self.check_collison():
                    break

        self._dis_ee = np.linalg.norm(self.pee)
        self._dis_vor = self._dis_ee
        self._dis_init = self._dis_ee

        dis_base = np.linalg.norm(self.robot_state[0:2])
        if dis_base < 1:
            self.rboundary = 1
        else:
            self.rboundary = dis_base + 1
        self.rboundary = 1+self.spacemax
        self._dis_base = dis_base
        self.get_observation()
        return np.array(self._observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_observation(self):
        observation = []# [^pbx, ^pby, bv, btheta, bw, j1, j2, j3, (^pbssx, ^pobsy), ^pex, ^pey]
        self.plinks = self.transform_joint_position(self.robot_state[5:8], self.linklength, self.robot_state[0:2],
                                                    self.robot_state[3])
        observation.extend(self.robot_state)
        if self.has_obs:
            observation.extend(self.pobs)
        observation.extend(self.plinks[4:6])
        '''
        # encode angles to sin and cos
        observation[0:2] = self.robot_state[0:2]/self.rboundary
        observation.append(self.robot_state[2])
        observation.append(np.sin(self.robot_state[3]))
        observation.append(np.cos(self.robot_state[3]))
        observation.append(np.sin(self.robot_state[4]))
        observation.append(np.cos(self.robot_state[4]))
        observation.append(np.sin(self.robot_state[5]))
        observation.append(np.cos(self.robot_state[5]))
        observation.append(np.sin(self.robot_state[6]))
        observation.append(np.cos(self.robot_state[6]))
        observation[11:13] = Tpos[4:6]/self.rboundary
        '''
        # print('ob', observation[10:12], Tpos[4:6])
        # observation[10:16] = Tpos
        # observation[5:7] = self._goal
        self._observation = observation #+ np.random.normal(0,0.02,size=(1,9))[0]

        return observation

    def step(self, action):
        # action = [^j1, ^j2, ^j3, ^deltav, ^deltaw]
        # joint state difference, base vel differences
        self._action = action*0.5
        self.stepsCounter += 1
        # check the base angular, base vel, base position and arm position on this base config
        wbasenew, wbaseold = self.check_w([self.robot_state[4]], [x * self.dt for x in [action[4]]])
        self.robot_state[4] = wbasenew[0]  # new angular vel
        thetabasenew, thetabaseold = self.check_theta([self.robot_state[3]], [x * self.dt for x in [self.robot_state[4]]])
        self.robot_state[3] = thetabasenew[0]  # new angular
        vbasenew, vbaseold = self.check_v([self.robot_state[2]], [x * self.dt for x in [action[3]]])
        self.robot_state[2] = vbasenew[0]  # new base linear vel
        self.robot_state[0:2] += self.robot_state[2] * self.dt * np.array([np.cos(self.robot_state[3]), np.sin(self.robot_state[3])])
        #Tpos0 = self.transform_joint_position(self.robot_state[5:8], self.linklength, self.robot_state[0:2], self.robot_state[4])
        deltadis = np.linalg.norm(self.robot_state[0:2])
        if deltadis > self.rboundary:# or self.check_boundary(Tpos0):
            self.r_penalty = -5
            self._terminated = True
            #self.robot_state[0:2] -= self.robot_state[2] * self.dt * np.array([np.cos(self.robot_state[3]), np.sin(self.robot_state[3])])
            #self.r_penalty = -10*deltadis#*np.linalg.norm(self.robot_state[0:2])
            #self.robot_state[2] = 0 #-vbasenew[0]#np.zeros(2)#vbaseold
        #else:
            #self.r_penalty = 0
            #self._terminated = True
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
            #self.datawriterfile.writerow([self.successCount, self.episodeCount, self.successCount/self.episodeCount])
            print('success rate : ', self.successCount, self.episodeCount, self.successCount/self.episodeCount)
        # observation = self.get_observation()
        reward = self.reward()
        return np.array(self._observation), reward, self._terminated, {}

    def terminator(self):
        observations = self.get_observation()
        self.pbase = observations[0:2]
        self.pee = observations[-2:]
        self.rfunc = reachingRewards.ReachingReward(with_priority=self._with_prio, goal=self._goal, armpos=self.pee,
                                                    basepos=self.pbase, opos=self.pobs)
        # relative_dis = np.subtract(self._goal, observations[-2:])  # self.get_observation()
        self._dis_ee = self.rfunc.distance(self._goal, self.pee)/self.rboundary
        self._dis_base = self.rfunc.distance(self._goal, self.pbase)/self.rboundary
        # Tpos0 = self.transform_joint_position(self.robot_state[5:8], self.linklength, self.robot_state[0:2], self.robot_state[3])
        if self.has_obs:
            if self.check_collison():
                self._terminated = True
                self.r_penalty = -5
        # print('dis', self._dis_ee, np.linalg.norm(observations[-2:]))
        if self.stepsCounter >= self._maxSteps - 1:
            self._terminated = True
            self.r_penalty = -5

        if self._dis_ee < THRESHOLD:
            print('Reaching : ', self._dis_ee, self._goal, self.robot_state)
            self._terminated = True
            self.r_penalty = 10
            self.successCount += 1

        if self._terminated:
            return True
        return False

    def path_guided_reward(self):
        a = self.pstart[1] / self.pstart[0]
        b = 0
        a1 = -1 / a
        b1 = self.pee[1] - self.pee[0] * a1
        xinter = b1 / (a - a1)
        yinter = a * xinter
        discurrent = np.sqrt(xinter ** 2 + yinter ** 2)

        if discurrent<0.1:
            discurrent = 0.1

        xigma2 = discurrent ** 2 / (2 * np.pi)
        dxiyi = np.abs(self.pee[1] - a * self.pee[0]) / np.sqrt(a ** 2 + b**2)
        r = 1 / discurrent * np.exp(-dxiyi ** 2 / 2 / xigma2) - 10 / 1 - 1
        return r

    def reward(self):
        self.var = self.var*0.9
        tau = self._dis_ee/self._dis_init
        tau = np.cbrt(tau)
        ree = self._dis_ee**2 #- np.log(self._dis_ee) - self._dis_eeself.rfunc.r_divid(self._goal, self.pee)#
        rbase = np.log(self._dis_base)#self.rfunc.r_divid(self._goal, self.pbase)#
        if self._dis_base < 0.3:
            rp = ree
        else:
            if tau < 1:
                rp = (1 - tau) * ree + tau * rbase
                # penalty = 0
            else:
                rp = rbase

        delta_dis = self._dis_ee - self._dis_vor
        self._dis_vor = self._dis_ee
        if delta_dis < 0:
            f_reward = -self.rboundary*delta_dis*10
            f_noise = 0
        else:
            f_reward = self.rboundary*delta_dis*10
            f_noise = np.random.normal(0, self._dis_ee)
                #penalty = 2*(tau-1)*self._dis_ee
        #reward = -(1-tau)*self._dis_ee - tau*self._dis_base + self.r_penalty - self.stepsCounter/self._maxSteps - penalty
        # noise = OrnsteinUhlenbeckProcess(1, n_steps_annealing=self._maxSteps).generate(self.stepsCounter)[0]
        # noise = OUNoise(1).noise()[0]
        noise = np.random.normal(ree, self._dis_ee**2)
        r = self.path_guided_reward()
        reward_vector = [-10*ree, self.r_penalty, - self.stepsCounter/self._maxSteps, -2*np.linalg.norm(self._action)**2]#self.path_guided_reward()#-np.log((1+np.exp(-self.rfunc.reward_normal(1,0))))self.spacemax *
        reward = np.sum(reward_vector)
        # #+ np.random.normal(np.linalg.norm(self._action), self._dis_ee/50)#/np.linalg.norm(reward_vector)
        # #+ np.random.normal(ree, 1/(self.episodeCount+1))#self._dis_ee/20)
        # #+ self.r_penalty  + np.random.normal(ree, self._dis_ee/100)
        # #+ self.r_penalty - self.stepsCounter/self._maxSteps
        # #- np.linalg.norm(self._action)+ np.random.normal(0, 1/self.stepsCounter/5)
        return reward

    # check if robot reaches the boundary
    def check_boundary(self, p):
        for i in range(len(p)):
            if np.abs(p[i]) > self.spacemax:
                return True
        return False
    # check if robot angular vel in the range
    def check_w(self, w, dw):
        assert len(w) == len(dw)
        w0 = np.zeros(len(w))
        for i in range(len(w)):
            w0[i] = w[i]
            if np.abs(w[i]+dw[i]) > 1:
                w[i] = w[i]
            else:
                w[i] += dw[i]
        return w, w0
    # check robot vel
    def check_v(self, v, dv):
        assert len(v) == len(dv)
        v0 = np.zeros(len(v))
        for i in range(len(v)):
            v0[i] = v[i]
            if np.abs(v[i]+dv[i]) > 0.5:
                v[i] = v[i]
            else:
                v[i] += dv[i]
        return v, v0

    # check if robot angular or joint angles in the range
    def check_theta(self, theta, dtheta):
        assert len(dtheta) == len(dtheta)
        theta0 = np.zeros(len(theta))
        for i in range(len(theta)):
            theta0[i] = theta[i]
            if np.abs(theta[i]+dtheta[i]) > np.pi:
                theta[i] = theta[i]
            else:
                theta[i] += dtheta[i]
        return theta, theta0

    def check_collison(self):
        for i in [0, 2, 4]:
            dis = np.linalg.norm(self.pobs-self.plinks[i:i+2])
            if dis < 0.1:
                return True
        return False

    def transform_joint_position(self, a, l, pbase, thetabase):
        assert len(a) == 3
        p1 = np.mat([l[0]*np.cos(a[0]), l[0]*np.sin(a[0]), 1])
        p2 = np.mat([l[0]*np.cos(a[0]) + l[1]*np.cos(a[0] + a[1]), l[0]*np.sin(a[0]) + l[1]*np.sin(a[0] + a[1]), 1])
        p3 = np.mat([l[0]*np.cos(a[0]) + l[1]*np.cos(a[0] + a[1]) + l[2]*np.cos(a[0] + a[1] + a[2]),
                     l[0]*np.sin(a[0]) + l[1]*np.sin(a[0] + a[1]) + l[2]*np.sin(a[0] + a[1] + a[2]), 1])

        m0 = np.mat([[np.cos(thetabase), -np.sin(thetabase), pbase[0]], [np.sin(thetabase), np.cos(thetabase), pbase[1]], [0, 0, 1]])

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
            viewboundary = np.sqrt(2)*self.spacemax+1
            self.viewer.set_bounds(-viewboundary, viewboundary, -viewboundary, viewboundary)
            #self.viewer.add_geom(rendering.Line((-VIEWLINE, VIEWLINE), (-VIEWLINE, -VIEWLINE)))
            #self.viewer.add_geom(rendering.Line((VIEWLINE, VIEWLINE), (VIEWLINE, -VIEWLINE)))
            #self.viewer.add_geom(rendering.Line((VIEWLINE, -VIEWLINE), (-VIEWLINE, -VIEWLINE)))
            #self.viewer.add_geom(rendering.Line((VIEWLINE, VIEWLINE), (-VIEWLINE, VIEWLINE)))
            # draw obs point
            if self.has_obs:
                obs = self.viewer.draw_circle(0.1)
                obs.set_color(0.4, 0.8, 0.6)
                self.viewer.add_geom(obs)
                self.obstrans = rendering.Transform(translation=self.pobs)
                obs.add_attr(self.obstrans)
            # draw goal point
            rgoal = self.viewer.draw_circle(0.025)
            rgoal.set_color(1, 0.0, 0.0)
            self.viewer.add_geom(rgoal)
            self.goaltrans = rendering.Transform(translation=self._goal)
            rgoal.add_attr(self.goaltrans)
            # draw reaching bounding box
            rcircle = self.viewer.draw_circle(THRESHOLD, filled=False)
            rcircle.set_color(0.1, 0.1, 0.1)
            self.viewer.add_geom(rcircle)
            self.circletrans = rendering.Transform(translation=self._goal)
            rcircle.add_attr(self.circletrans)
            # draw moving area as a circle boundary
            rcircle = self.viewer.draw_circle(self.rboundary, filled=False)
            rcircle.set_color(0.3, 0.5, 0.6)
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
        self.link0trans.set_rotation(self.robot_state[4])
        self.link1trans.set_rotation(self.robot_state[5])
        self.link2trans.set_rotation(self.robot_state[6])
        self.cartrans.set_translation(self.robot_state[0], self.robot_state[1])
        self.cartrans.set_rotation(self.robot_state[3])

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
    env = MM2DRobotCGymEnv(maxstep=2e2, space=0.5, ifobs=True)
    rec = VideoRecorder(env, path='/home/zheng/fromremote/video/ab.mp4')
    #mon = Monitor(env, '/home/zheng/fromremote/video/video',video_callable=lambda episode_id: True,force = True)
    #env.set_fps(30)
    for ep in range(500):
        s = env.reset()

        # print(env.transform_joint_position([0,0,0],[1,2,1]))
        # print('reset')
        for i in range(500):
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
                print(i)
                break
        rec.close()
        env.close()