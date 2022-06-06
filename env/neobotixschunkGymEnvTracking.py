"""
DRL env in pybullet for mobile manipulation of neobotix mp500 + schunk lwa4d + schunk gripper pg70
original built by X. Wang & Z. Zheng
developed and maintained by Z. Zheng, @KIT-IPR
schunk model meshes source : https://github.com/ipa320/schunk_modular_robotics
neobotix model meshed source : https://github.com/neobotix/neo_mp_500
model modified by Y. Zhang and J. Su.
model updated by Z. Zheng
"""

import os
import inspect
import datetime

import gym
from gym import spaces
from gym.utils import seeding
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
import pybullet as p
import time
from pkg_resources import parse_version
import csv

from env import neobotixschunk
from env import neobotixschunk_goal
from env import neobotixschunk_obstacle
#from env import reachingRewards
#from env import heuristicReward
#from env import fieldDirection
#from ddpg.ddpg_noise import NormalActionNoise, AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
os.sys.path.insert(0, PARENT_DIR)

SUCCESS_STEPS_UPDATE = 1000  # parameter for update success rate every SUCCESS_STEPS_UPDATE during the training
largeValObservation = 100
RENDER_HEIGHT = 720
RENDER_WIDTH = 960
PATH_POINT_RADIUS = 0.01
COLLISION_THRESHOLD = 0.5


class NeobotixSchunkGymEnvTracking(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdf_root=PARENT_DIR,
                 action_repeat=1,
                 time_step=1. / 240.,
                 enable_self_collision_flag=True,
                 is_discrete=False,
                 renders=False,
                 max_steps=1e3,
                 reward_type='rdense',
                 if_prioritized=False,
                 action_dim=9,
                 random_initial=True,
                 ws_boundary=1,
                 if_obstacle=False,
                 if_obstacle_moving=False,
                 if_goal_moving_type='static'):
        super(NeobotixSchunkGymEnvTracking, self).__init__()
        self.urdf_root = urdf_root
        self.action_repeat = action_repeat
        self.enable_self_collision_flag = enable_self_collision_flag
        self.is_discrete = is_discrete
        self.if_rendering = renders
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.with_prioritized_reward = if_prioritized
        self.action_dim = action_dim
        self.is_random_init = random_initial
        self.ws_boundary = ws_boundary
        self.if_obstacle = if_obstacle
        self.if_obstacle_moving = if_obstacle_moving
        self.if_goal_moving_type = if_goal_moving_type
        self.observation = []
        self.step_counter_per_episode = 0
        self.sound_reaching_counter_per_episode = 0
        self.time_step = time_step
        self.r_termination = 0
        self.r_penalty_collision = 0
        self.terminated = 0
        self.dis_vor = 100
        self.success_update_counter = 0
        self.episode_counter = 0
        self.update_step_counter = 0
        self.total_success_counter = 0
        self.dis_ee_init = 0
        self.dis_collision = 1
        self.dis_ee = 0
        self.dis_base = 0
        self.ee_position = []
        self.base_position = []
        self.collision_relative_position = []
        self.flag_collide = 0
        self.collision_probability = 0
        self.actions = []
        self.u = 0
        self.sound_reaching_number = 1
        self.seed_number = 0
        self.np_random = None
        self._pb = p
        self.former_ee_pos = []
        self.former_base_pos = []
        self.init_ee = []
        self.init_goal = []

        self.path_point_base = None  # rm
        self.path_point_ee = None  # rm
        self.former_goal_pos = []

        self.ground_uid = None
        self.robot = None
        self.goal = None
        self.obstacle = None

        self.heu_reward_function = None
        self.URDF_GROUND = os.path.join(self.urdf_root, "pybullet_neoschunk_reaching/data/plane.urdf")

        self.DATA_SUCCESS_RATE = None
        self.DATA_ACTION = None
        self.DATA_STEPS = None

        self.cam_dist = 4
        self.cam_yaw = 0
        self.cam_pitch = -89.9

        self.disc_total_reward = 0

        if self.if_rendering:
            cid = self._pb.connect(self._pb.SHARED_MEMORY)
            if cid < 0:
                cid = self._pb.connect(self._pb.GUI)
            #self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_RENDERING, 0)
            #self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_GUI, 0)
            # disable tinyrenderer, software (CPU) renderer, we don't use it here
            #self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_TINY_RENDERER, 0)
            #self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            #self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_GUI, 1)
            self._pb.resetDebugVisualizerCamera(self.cam_dist, self.cam_yaw, self.cam_pitch, [0.52, -0.2, -0.33])
        else:
            self._pb.connect(self._pb.DIRECT)

        self.seed_number = self.seed()
        self.set_data_csv_path()
        if self.if_obstacle_moving:
            self.if_obstacle = True
        self.reset()

        self.observation_dim = len(self.getExtendedObservation())
        observation_high = np.array([largeValObservation] * self.observation_dim)
        action_boundary = 1
        if self.is_discrete:
            self.action_space = spaces.MultiDiscrete(np.ones(self.action_dim) * 3)
        else:
            self.action_bound = np.ones(self.action_dim) * action_boundary
            self.action_space = spaces.Box(low=-self.action_bound, high=self.action_bound, dtype=np.float32)

        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float32)
        self.viewer = None
        # self.calculateField = fieldDirection.FieldDirection()
        # help(neobotixschunk)

    def set_data_csv_path(self):
        #self.DATA_SUCCESS_RATE = os.path.join(self.urdf_root,'pybullet_neoschunk_reaching/results/success_rate_update_noobs_'+str(self.seed_number[0])+'.csv')
        t_date = datetime.datetime.now()
        # print(t_date) 2020-10-08 16:18:21.814188
        path_date = "{:%Y%m%d%H%M%S}".format(t_date)
        path_end = 'free'
        if self.with_prioritized_reward:
            path_end = 'prio'
        if self.if_obstacle:
            path_end ='obs'
        if self.if_obstacle_moving:
            path_end = 'obsmoving'
        if self.if_goal_moving_type == 'random':
            path_end = 'random'
        if self.if_goal_moving_type == 'line':
            path_end = 'line'
        if self.if_goal_moving_type == 'circle':
            path_end = 'circle'
        path_csv = '_' + path_date + '_' + path_end + '.csv'
        #self.DATA_SUCCESS_RATE = os.path.join(self.urdf_root, 'pybullet_neoschunk_reaching/results/success_rate'+path_csv)
        #self.DATA_ACTION = os.path.join(self.urdf_root, 'pybullet_neoschunk_reaching/results/action'+path_csv)
        #self.DATA_STEPS = os.path.join(self.urdf_root, 'pybullet_neoschunk_reaching/results/steps'+path_csv)

    def reset_params(self):
        self.r_penalty_collision = 0
        self.r_termination = 0
        self.terminated = 0
        self.collision_probability = 0
        self.step_counter_per_episode = 0
        self.sound_reaching_counter_per_episode = 0
        self.sound_reaching_number = 1
        self.u = 0
        self.dis_collision = 1

    def reset(self):
        """
        reset env
        :return:
        """
        self.reset_params()
        self._pb.resetSimulation()
        self._pb.setPhysicsEngineParameter(numSolverIterations=200, enableFileCaching=0)
        self._pb.setPhysicsEngineParameter(solverResidualThreshold=1e-30)
        self._pb.setTimeStep(self.time_step)
        self._pb.setGravity(0, 0, -9.81)
        self._pb.setRealTimeSimulation(False)

        #video_path = os.path.join(self.urdf_root,'pybullet_neoschunk_reaching/results/videos/video_'+str(self.episode_counter)+'.mp4')
        #open(video_path, 'a')
        #self.logvideo = self._pb.startStateLogging(self._pb.STATE_LOGGING_VIDEO_MP4, video_path)
        if self.if_rendering:
            #self.path_point_base = self._pb.createVisualShape(shapeType=self._pb.GEOM_SPHERE, radius=PATH_POINT_RADIUS, rgbaColor=[0, 0, 1, 0.9])
            #self.path_point_ee = self._pb.createVisualShape(shapeType=self._pb.GEOM_SPHERE, radius=PATH_POINT_RADIUS, rgbaColor=[1, 0, 0, 0.9])
            time.sleep(self.time_step)

        self.ground_uid = self._pb.loadURDF(self.URDF_GROUND, [0, 0, -0.001], useFixedBase=True, flags=self._pb.URDF_ENABLE_SLEEPING)

        self.robot = neobotixschunk.NeobotixSchunk(urdf_root_path=self.urdf_root, ws_boundary=self.ws_boundary, rseed=self.np_random)
        if self.is_random_init:
            self.robot.resetRandomRobotState()

        self.goal = neobotixschunk_goal.NeobotixSchunkGoal(urdf_root_path=self.urdf_root, ws_boundary=self.ws_boundary, rseed=self.np_random, if_goal_moving_type=self.if_goal_moving_type, p_goal_start=np.zeros(3))
        self.goal.resetGoal()

        if self.if_obstacle:
            self.obstacle = neobotixschunk_obstacle.NeobotixSchunkObstacle(urdf_root_path=self.urdf_root, ws_boundary=self.ws_boundary, rseed=self.np_random, if_obstacle_moving=self.if_obstacle_moving)
            self.obstacle.resetObstacle()

        while True:
            flag1 = self.robot.check_collision_self() #or (np.linalg.norm(self.goal_position-[-0.38, 0.0, 0.76])>0.6)  # distance between goal and 2nd link
            if self.if_obstacle:
                self.check_collision_obs()
                flag2 = (self.dis_collision < 0.1)
                flag3 = (np.linalg.norm(self.goal.goal_position[0:2] - self.obstacle.obstacle_position[0:2]) < 0.1)
                flag1 = (flag1 or flag2 or flag3)
            if flag1:
                self.goal.resetGoal()
                if self.is_random_init:
                    self.robot.resetRandomRobotState()
                if self.if_obstacle:
                    self.obstacle.resetObstacle()
            else:
                break
        self._pb.stepSimulation()
        self.observation = self.getExtendedObservation()
        self.goal.goal_position = self.ee_position
        self.goal.goal_orientation = self._pb.getQuaternionFromEuler(self.observation[3:6])
        self.goal.resetGoal()
        self.former_ee_pos = self.ee_position
        self.former_base_pos = self.base_position
        self.former_goal_pos = self.goal.goal_position
        self.init_ee = self.ee_position
        self.init_goal = self.goal.goal_position
        self.dis_ee_init = np.linalg.norm(np.subtract(self.ee_position, self.goal.goal_position))
        self.dis_base = np.linalg.norm(np.subtract(self.base_position, self.goal.goal_position))
        self.dis_ee = self.dis_ee_init
        self.dis_vor = self.dis_ee_init

        print('Episode :', self.episode_counter, ' Total Reward :', self.disc_total_reward)
        self.disc_total_reward = 0
        return self.observation

    def calculate_point2line(self, point, line_endpoint1, line_endpoint2):
        """
        calculate distance from current ee position to the straight line path between ee to goal at initial state
        :return: distance to the line
        """
        line_vector = np.subtract(line_endpoint1, line_endpoint2)
        point_line_vector = np.subtract(point, line_endpoint2)
        delta_line_vector = np.dot(line_vector, line_vector)
        if delta_line_vector == 0:
            delta_line_vector = 1e-16
        t = np.dot(line_vector, point_line_vector) / delta_line_vector
        point_on_line = line_vector * t + line_endpoint2
        d = np.linalg.norm(np.subtract(point, point_on_line))
        return d

    def close(self):
        self._pb.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        """
        get observation
        :return:
        """
        observation = self.robot.getObservation()
        self.goal.getGoalState()
        self.ee_position = observation[0:3]
        self.base_position = observation[12:15]
        relative_pos_ee = np.subtract(self.goal.goal_position, self.ee_position)
        observation[0:3] = relative_pos_ee
        relative_pos_base = np.subtract(self.goal.goal_position, self.base_position)
        observation[12:15] = relative_pos_base
        self.dis_ee = np.linalg.norm(relative_pos_ee)
        self.dis_base = np.linalg.norm(relative_pos_base)

        ee_vell = observation[6:9]
        ee_vela = observation[9:12]
        base_vell = observation[18:21]
        base_vela = observation[21:24]
        #if self.if_goal_moving_type is 'static':
            #ee_vell = [0, 0, 0]
            #base_vell = [0, 0, 0]
        if self.if_goal_moving_type is not 'static':
            relative_vell_goal_ee = np.subtract(self.goal.goal_linear_velocity, ee_vell)
            relative_vela_goal_ee = np.subtract(self.goal.goal_angular_velocity, ee_vela)
            relative_vell_goal_base = np.subtract(self.goal.goal_linear_velocity, base_vell)
            relative_vela_goal_base = np.subtract(self.goal.goal_angular_velocity, base_vela)
        else:
            relative_vell_goal_ee = np.zeros(3)
            relative_vela_goal_ee = np.zeros(3)
            relative_vell_goal_base = np.zeros(3)
            relative_vela_goal_base = np.zeros(3)

        observation[6:9] = relative_vell_goal_ee
        observation[9:12] = relative_vela_goal_ee
        observation[18:21] = relative_vell_goal_base
        observation[21:24] = relative_vela_goal_base

        self.collision_relative_position = np.ones(3)
        # add dims for goal relative obs and base pos 3, relative collisions 3
        observation.extend(np.zeros(9))  # 44,45,46,47,48,49,50,51,52
        if self.if_obstacle:
            self.obstacle.getObstacleState()
            observation[-9:-6] = np.subtract(self.obstacle.obstacle_linear_velocity, base_vell)  # 44,45,46
            observation[-6:-3] = np.subtract(self.obstacle.obstacle_position, self.base_position)  # 47,48,49
            observation[-3:] = self.collision_relative_position  # 50,51,52
        observation = np.array(observation)
        # observation = self.np_random.normal(observation, 0.001, size=len(observation))
        if self.if_goal_moving_type is not 'static':
            # remove states : ee orn(3,4,5), ee vel(6,7,8,9,10,11), base vel(18,19,20,21,22,23), joint vels 7(31,32,33,34,35,36,37)
            # remove 0, 1, or static elements : base orn xy(15,16,17), obs v(44,45,46), relative base obs z(49), relative collision z(52)
            rm_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35, 36, 37, 44, 45, 46, 47, 48, 49, 52]
        else:
            # remove states : ee orn(3,4,5), ee vel(6,7,8,9,10,11), base vel(18,19,20,21,22,23), joint vels 7(31,32,33,34,35,36,37)
            # remove 0, 1, or static elements : base orn xy(15,16), relative base obs z(49), relative collision z(52)
            rm_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35, 36, 37, 44, 45, 46, 47, 48, 49, 52]
        self.observation = np.delete(observation, rm_indices)
        # self.observation = observation
        return self.observation

    def step(self, input_action):
        """
        scale actions
        :param input_action:
        :return:
        """
        '''
        if self.dis_ee < 0.5:
            p_scale = 0.01
            # input_action[0:2] = input_action[0:2]*np.ones(2)*p_scale
        else:
            p_scale = 1
            # input_action[2:9] = input_action[2:9]*np.ones(7)*p_scale
        scaled_action = np.multiply(input_action, self.action_bound*p_scale)
        '''
        scaled_action = np.zeros(self.action_dim)
        if self.action_dim == 10:
            accjoint = 1
            scaled_action[0] = input_action[0] * accjoint
            scaled_action[1] = input_action[1] * accjoint
            scaled_action[2] = input_action[2] * accjoint
            scaled_action[3] = input_action[3] * accjoint
            scaled_action[4] = input_action[4] * accjoint
            scaled_action[5] = input_action[5] * accjoint
            scaled_action[6] = input_action[6] * accjoint
            scaled_action[7] = input_action[7]
            scaled_action[8] = input_action[8]
            scaled_action[9] = input_action[9]
        elif self.action_dim == 6:
            if self.is_discrete:
                dx = 0.02
                dy = 0.02
                dz = 0.02
                dv = 0.03
                dw = 0.04
                actions = np.array([[-dx, 0, dx],
                                    [-dy, 0, dy],
                                    [-dz, 0, dz],
                                    [-dv, 0, dv],
                                    [-dv, 0, dv],
                                    [-dw, 0, dw]])
                scaled_action = np.choose(np.array(input_action), actions.T)
            else:
                accjoint = 1#0.5
                scaled_action[0] = input_action[0] * accjoint
                scaled_action[1] = input_action[1] * accjoint
                scaled_action[2] = input_action[2] * accjoint
                scaled_action[3] = input_action[3] #* 0.3
                scaled_action[4] = input_action[4] #* 0.3
                scaled_action[5] = input_action[5] #* 0.4
        elif self.action_dim == 9:
            if self.is_discrete:
                dx = 0.02
                dy = 0.02
                dz = 0.02
                dwx = 0.05
                dwy = 0.05
                dwz = 0.05
                dv = 0.03
                dw = 0.04
                actions = np.array([[-dx, 0, dx],
                                    [-dy, 0, dy],
                                    [-dz, 0, dz],
                                    [-dwx, 0, dwx],
                                    [-dwy, 0, dwy],
                                    [-dwz, 0, dwz],
                                    [-dv, 0, dv],
                                    [-dv, 0, dv],
                                    [-dw, 0, dw]])
                scaled_action = np.choose(np.array(input_action), actions.T)
            else:
                accjoint = 0.03
                scaled_action[0] = input_action[0] * accjoint
                scaled_action[1] = input_action[1] * accjoint
                scaled_action[2] = input_action[2] * accjoint
                scaled_action[3] = input_action[3] * accjoint/2
                scaled_action[4] = input_action[4] * accjoint/2
                scaled_action[5] = input_action[5] * accjoint/2
                scaled_action[3] = input_action[6] * 0.3
                scaled_action[4] = input_action[7] * 0.3
                scaled_action[5] = input_action[8] * 0.4
        return self.step_shaped(scaled_action)

    def step_shaped(self, action_scaled):
        """
        step scaled actions
        :param action_scaled:
        :return:
        """
        self.former_ee_pos = self.ee_position
        self.former_base_pos = self.base_position
        self.former_goal_pos = self.goal.goal_position
        for i in range(self.action_repeat):
            self.goal.setGoalState()
            if self.if_obstacle:
                self.obstacle.setObstacleState()
            self.robot.applyAction(action_scaled)
            self._pb.stepSimulation()
            done = self._termination()
            if done:
                self.episode_counter += 1
                self.update_step_counter += 1
                #self._pb.stopStateLogging(self.logvideo)
                break
            self.step_counter_per_episode += 1

        if self.if_rendering:
            #self._pb.createMultiBody(baseMass=0, basePosition=self.ee_position, baseVisualShapeIndex=self.path_point_ee)
            #self._pb.createMultiBody(baseMass=0, basePosition=self.base_position, baseVisualShapeIndex=self.path_point_base)
            time.sleep(self.time_step)
        #print(self.dis_collision, self.collision_relative_position)
        self.u = np.linalg.norm(action_scaled)
        self.actions = action_scaled

        nobs = np.linalg.norm(self.observation)
        if nobs == 0:
            nobs += 1e-16
        #self.observation = self.observation / nobs
        reward = self._reward()
        return self.observation, reward, done, {}

    def check_collision_obs(self):
        """
        check collisions with obstacle
        :return:
        """
        closest_points = []
        closest_distances = []
        if self.if_obstacle:
            closest_points = self._pb.getClosestPoints(self.robot.neobotix_schunk_uid, self.obstacle.obstacle_uid, COLLISION_THRESHOLD)
        self.flag_collide = len(closest_points)
        if self.flag_collide:
            for i in range(self.flag_collide):
                closest_distances.append(closest_points[i][8])
            self.dis_collision = np.min(closest_distances)
            #self.collision_probability = np.exp(100*(0.1-self.dis_collision))
            j = np.argmin(closest_distances)
            self.collision_relative_position = np.array(closest_points[j][5])-np.array(closest_points[j][6])
            self.collision_relative_position[2] = 0
            return True
        return False

    def _termination(self):
        self.observation = self.getExtendedObservation()
        #if self.if_obstacle:
            #self.r_func = reachingRewards.ReachingReward(with_priority=self.with_prioritized_reward, goal=self.goal.goal_position, armpos=self.ee_position, basepos=self.base_position, opos=self.obstacle.obstacle_position)
        '''
        if self.observation[8]>1e2:
            print('ACHTUNG : unstable status!')
            self.terminated = -1
            self.r_termination = -100
            return True
        '''
        if self.check_collision_obs():
            # force, d_force = self.calculateField.compute_sum_force()
            self.observation[-2:] = self.collision_relative_position[0:2]
            if self.dis_collision > COLLISION_THRESHOLD:
                self.r_penalty_collision = -1
            elif self.dis_collision > 0.1:
                self.r_penalty_collision = -np.log(self.dis_collision)/np.log(COLLISION_THRESHOLD)
            else:
                self.r_penalty_collision = -np.log(0.1)/np.log(COLLISION_THRESHOLD)
                self.r_termination = -100
                self.terminated = 2
                print('ACHTUNG : collision with obs!')
                return True

        if self.robot.check_collision_self():
            self.terminated = 3
            self.r_termination = -100
            print('ACHTUNG : self-collision!')
            return True

        if self.step_counter_per_episode >= self.max_steps-1:
            print('Episode Done!')
            self.terminated = 4
            self.r_termination = -50
            return True

        if self.terminated:
            return True

        return False

    def _reward(self):
        reward = 0
        #rd = self.heu_reward_function.fun_gaussian(self.observation[0:2])
        #r = self.r_func.reward_field3d()
        dline = self.calculate_point2line(self.ee_position, self.init_ee, self.init_goal)
        delta_dis = self.dis_ee - self.dis_vor
        self.dis_vor = self.dis_ee
        if self.dis_ee_init:
            tau = self.dis_ee/self.dis_ee_init
            tau = tau**2#np.cbrt(tau)

        ree = -1.5*self.dis_ee + np.exp(-10*self.dis_ee**2)
        rbase = -self.dis_base**2
        rline = ree-dline**2

        if self.reward_type == 'rdense':
            # noise = AdaptiveParamNoiseSpec(mu=0, sigma=0.1)
            reward = ree #+ self.r_termination  + 80*self.r_penalty_collision - 40*self.u**2 - 10*(self.step_counter_per_episode/self.max_steps)**2
            if self.if_goal_moving_type == 'line':
                # dline_of_line^2 = dee^2 - dline^2, dline^2 = dz^2 + dy^2
                dis_line2 = self.dis_ee**2-((self.ee_position[2]-self.goal.goal_position[2])**2+(self.ee_position[1]-self.goal.goal_position[1])**2)
                reward = ree  #+ self.r_termination + 50 * self.r_penalty_collision - 40 * self.u ** 2 - 10 * (self.step_counter_per_episode / self.max_steps) ** 2
            if self.if_goal_moving_type == 'circle':
                dis_circle2 = self.dis_ee**2-(self.ee_position[2]-self.goal.goal_position[2])**2
                reward = ree  #+ self.r_termination + 50 * self.r_penalty_collision - 40 * self.u ** 2 - 10 * (self.step_counter_per_episode / self.max_steps) ** 2

        elif self.reward_type == 'rsparse':
            if delta_dis > 0:
                reward = 0
            else:
                reward = 1

        self.disc_total_reward += reward * 0.998 ** self.update_step_counter
        return reward#/1000

    def render(self, mode='rgb_array', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = self._pb.getBasePositionAndOrientation(self.robot.neobotix_schunk_uid)
        #text_goal = 'goal position : (' + str(round(self.goal_position[0], 4)) + ', ' + str(round(self.goal_position[1], 4)) + ', ' + str(round(self.goal_position[2], 4)) + ')'
        #text_ee = 'ee position : (' + str(round(self.ee_position[0], 4)) + ', ' + str(round(self.ee_position[1], 4)) + ', ' + str(round(self.ee_position[2], 4)) + ')'
        #self._pb.addUserDebugText(text_goal, [-3, -1, 3.4], textSize=1.5)
        #self._pb.addUserDebugText(text_ee, [-3, -1, 3], textSize=1.5, lifeTime=0.5)
        self._pb.addUserDebugLine(self.former_ee_pos, self.ee_position, [0, 0, 1], 3)
        self._pb.addUserDebugLine(self.former_base_pos, self.base_position, [0, 1, 0], 3)
        if self.if_goal_moving_type is not 'static':
            self._pb.addUserDebugLine(self.former_goal_pos, self.goal.goal_position, [1, 0, 0], 2)
        view_matrix = self._pb.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos, distance=self.cam_dist, yaw=self.cam_yaw, pitch=self.cam_pitch, roll=0, upAxisIndex=2)
        proj_matrix = self._pb.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT, nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = self._pb.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=self._pb.ER_BULLET_HARDWARE_OPENGL)
        #renderer=self._pb.ER_TINY_RENDERER
        #self._pb.ER_BULLET_HARDWARE_OPENGL
        #self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_RENDERING, 1)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
