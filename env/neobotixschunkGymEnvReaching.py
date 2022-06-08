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
import pybullet as pb
import time
from pkg_resources import parse_version
import csv

from env import neobotixschunk
from env import neobotixschunk_goal
from env import neobotixschunk_obstacle
from env import neobotixschunk_scenario
from env import reachingRewards
from env import heuristicReward
from env import fieldDirection
#from ddpg.ddpg_noise import NormalActionNoise, AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
os.sys.path.insert(0, PARENT_DIR)

SUCCESS_STEPS_UPDATE = 100  # parameter to update success rate every SUCCESS_STEPS_UPDATE during the training
largeValObservation = 1.0
RENDER_HEIGHT = 720
RENDER_WIDTH = 960
PATH_POINT_RADIUS = 0.01
COLLISION_THRESHOLD = 0.5


class NeobotixSchunkGymEnvReaching(gym.Env):
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
                 if_goal_moving_type='static',
                 if_scenario=False,
                 result_dir=PARENT_DIR):
        self.urdf_root = urdf_root
        self.action_repeat = action_repeat
        self.enable_self_collision_flag = enable_self_collision_flag
        self.is_discrete = is_discrete
        self.if_rendering = renders
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.if_prioritized = if_prioritized
        self.action_dim = action_dim
        self.if_random_init = random_initial
        self.ws_boundary = ws_boundary
        self.if_obstacle = if_obstacle
        self.if_obstacle_moving = if_obstacle_moving
        self.if_goal_moving_type = if_goal_moving_type
        self.if_scenario = if_scenario
        self.observation = []
        self.step_counter_per_episode = 0
        self.sound_reaching_counter_per_episode = 0
        self.time_step = time_step
        self.r_termination = 0
        self.r_penalty_collision = 0
        self.terminated = 0
        self.dis_ee_vor = 100
        self.dis_base_vor = 100
        self.success_update_counter = 0
        self.episode_counter = 0
        self.update_step_counter = 0
        self.total_success_counter = 0
        self.dis_ee_init = 0
        self.dis_base_init = 0
        self.dis_collision = 1
        self.dis_ee = 0
        self.dis_base = 0
        self.ee_position = []
        self.base_position = []
        self.collision_relative_position = np.zeros(3)
        self.flag_collide = 0
        self.collision_probability = 0
        self.actions = np.zeros(7)
        self.input_u = 0
        self.sound_reaching_number = 1
        self.seed_number = 0
        self.np_random = None
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
        self.scenario = None

        self.heu_reward_function = None
        self.urdf_ground = os.path.join(self.urdf_root, "pybullet_neoschunk_reaching/data/plane.urdf")

        self.data_path_success_rate = None
        self.data_path_action = None
        self.data_path_steps = None
        if result_dir is not None:
            self.result_dir = result_dir
        else:
            self.result_dir = '/home/zzc/results/logs/videos/'

        self.cam_dist = 4
        self.cam_yaw = 0
        self.cam_pitch = -89.99

        self.flag_r1 = 1
        self.flag_r2 = 1

        self.r_function = None
        self.dis_action = 0

        gym.logger.set_level(40)

        if self.if_rendering:
            cid = pb.connect(pb.SHARED_MEMORY)
            if cid < 0:
                cid = pb.connect(pb.GUI)
            #pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
            #pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
            # disable tinyrenderer, software (CPU) renderer, we don't use it here
            #pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER, 0)
            #pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            #pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
            pb.resetDebugVisualizerCamera(self.cam_dist, self.cam_yaw, self.cam_pitch, [0.0, -0.5, -0.0])
        else:
            pb.connect(pb.DIRECT)

        self.seed_number = self.seed()
        self.__set_data_csv_path()
        if self.if_obstacle_moving:
            self.if_obstacle = True

        self.reset()
        self.observation_dim = len(self.observation)
        # largeValObservation >= max(observation[])
        observation_high = np.array([largeValObservation] * self.observation_dim)

        action_boundary = 1.0
        if self.is_discrete:
            self.action_space = spaces.MultiDiscrete(np.ones(self.action_dim) * 3)
        else:
            self.action_bound = np.ones(self.action_dim) * action_boundary
            self.action_space = spaces.Box(low=-self.action_bound, high=self.action_bound, dtype=np.float32)

        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float32)
        self.viewer = None
        # self.calculateField = fieldDirection.FieldDirection()
        # help(neobotixschunk)

    def __set_data_csv_path(self):
        #self.data_path_success_rate = os.path.join(self.urdf_root,'pybullet_neoschunk_reaching/results/success_rate_update_noobs_'+str(self.seed_number[0])+'.csv')
        t_date = datetime.datetime.now()
        # print(t_date) 2020-10-08 16:18:21.814188
        path_date = "{:%Y%m%d%H%M%S}".format(t_date)
        path_end = 'free'
        if self.if_prioritized:
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
        print('log results path ', self.result_dir)
        self.data_path_success_rate = os.path.join(self.result_dir, 'success_rate' + path_csv)
        self.data_path_action = os.path.join(self.result_dir, 'action' + path_csv)
        self.data_path_steps = os.path.join(self.result_dir, 'steps' + path_csv)

    def __reset_params(self):
        self.r_penalty_collision = 0
        self.r_termination = 0
        self.terminated = 0
        self.collision_probability = 0
        self.step_counter_per_episode = 0
        self.sound_reaching_counter_per_episode = 0
        self.sound_reaching_number = 1
        self.input_u = 0
        self.dis_collision = 1
        self.flag_r1 = 1
        self.flag_r2 = 1

    def reset(self):
        """
        reset env
        :return:
        """
        self.__reset_params()
        pb.resetSimulation()
        pb.setPhysicsEngineParameter(numSolverIterations=200, enableFileCaching=0)
        pb.setPhysicsEngineParameter(solverResidualThreshold=1e-30)
        pb.setTimeStep(self.time_step)
        pb.setGravity(0, 0, -9.81)
        pb.setRealTimeSimulation(False)

        #video_path = os.path.join(self.urdf_root,'pybullet_neoschunk_reaching/results/videos/video_'+str(self.episode_counter)+'.mp4')
        #open(video_path, 'a')
        #self.logvideo = pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, video_path)
        if self.if_rendering:
            #self.path_point_base = pb.createVisualShape(shapeType=pb.GEOM_SPHERE, radius=PATH_POINT_RADIUS, rgbaColor=[0, 0, 1, 0.9])
            #self.path_point_ee = pb.createVisualShape(shapeType=pb.GEOM_SPHERE, radius=PATH_POINT_RADIUS, rgbaColor=[1, 0, 0, 0.9])
            time.sleep(self.time_step)

        self.ground_uid = pb.loadURDF(self.urdf_ground, [0, 0, -0.001], useFixedBase=True, flags=pb.URDF_ENABLE_SLEEPING)

        self.robot = neobotixschunk.NeobotixSchunk(urdf_root_path=self.urdf_root, ws_boundary=self.ws_boundary, rseed=self.np_random)
        if self.if_random_init:
            self.robot.resetRandomRobotState()

        self.goal = neobotixschunk_goal.NeobotixSchunkGoal(urdf_root_path=self.urdf_root, ws_boundary=self.ws_boundary, rseed=self.np_random, if_goal_moving_type=self.if_goal_moving_type, p_goal_start=np.zeros(3))
        self.goal.resetGoal()

        if self.if_scenario:
            self.scenario = neobotixschunk_scenario.NeobotixSchunkScenario()
            self.goal.resetGoalScenario()
            #self.goal.resetGoalScenarioRandom()

        if self.if_obstacle:
            self.obstacle = neobotixschunk_obstacle.NeobotixSchunkObstacle(urdf_root_path=self.urdf_root, ws_boundary=self.ws_boundary, rseed=self.np_random, if_obstacle_moving=self.if_obstacle_moving)
            self.obstacle.resetObstacle()

        while True:
            flag1 = self.robot.check_collision_self() #or (np.linalg.norm(self.goal_position-[-0.38, 0.0, 0.76])>0.6)  # distance between goal and 2nd link
            if self.if_obstacle:
                self.__check_collision_obstacle()
                flag2 = (self.dis_collision < 0.1)
                flag3 = (np.linalg.norm(self.goal.goal_position[0:2] - self.obstacle.obstacle_position[0:2]) < 0.1)
                flag1 = (flag1 or flag2 or flag3)
            if self.if_scenario:
                flag4 = self.__check_collision_wall()
                flag1 = (flag1 or flag4)
            if flag1:
                #self.goal.resetGoal()
                if self.if_random_init:
                    self.robot.resetRandomRobotState()
                if self.if_obstacle:
                    self.obstacle.resetObstacle()
            else:
                break

        pb.stepSimulation()
        self.__get_observation()
        #self.goal.goal_position = self.ee_position
        #self.goal.resetGoal()
        self.former_ee_pos = self.ee_position
        self.former_base_pos = self.base_position
        self.former_goal_pos = self.goal.goal_position
        self.init_ee = self.ee_position
        self.init_goal = self.goal.goal_position
        self.dis_ee_init = np.linalg.norm(np.subtract(self.ee_position, self.goal.goal_position))
        self.dis_base_init = np.linalg.norm(np.subtract(self.base_position[0:2], self.goal.goal_position[0:2]))
        self.dis_base = np.linalg.norm(np.subtract(self.base_position[0:2], self.goal.goal_position[0:2]))

        self.dis_ee = self.dis_ee_init
        self.dis_ee_vor = self.dis_ee_init
        self.dis_base_vor = self.dis_base_init
        #self.heu_reward_function = heuristicReward.HeuristicReward(self.ee_position[0:2], self.goal_position[0:2], self.obstacle_position[0:2])
        data_writer_file = csv.writer(open(self.data_path_success_rate, "a"))

        if self.episode_counter:
            success_rate_update = 0
            if self.update_step_counter:
                success_rate_update = self.success_update_counter / self.update_step_counter
            success_rate_total = self.total_success_counter / self.episode_counter

            if self.update_step_counter == SUCCESS_STEPS_UPDATE:
                data_writer_file.writerow([self.episode_counter, self.total_success_counter, success_rate_total, self.update_step_counter, self.success_update_counter, success_rate_update, self.ws_boundary])
            '''
            if success_rate_update > 0.5 and self.update_step_counter == SUCCESS_STEPS_UPDATE:
                self.ws_boundary += 0.25
                self.max_steps += 100

            if self.ws_boundary > 0.5:
                self.ws_boundary = 0.5

            if self.max_steps > 300:
                self.max_steps = 300
            '''
            if not self.episode_counter%SUCCESS_STEPS_UPDATE:
                self.success_update_counter = 0
                self.update_step_counter = 0
            print('Episode :', self.episode_counter,
                  ' SuccessTotal :', self.total_success_counter,
                  ' success_rate_total :', success_rate_total,
                  f' StepUpdate : {self.update_step_counter}/{SUCCESS_STEPS_UPDATE}',
                  f' SuccessUpdate : {self.success_update_counter}/{SUCCESS_STEPS_UPDATE}',
                  ' success_rate_updateUpdate :', success_rate_update,
                  self.ws_boundary)

        return self.observation

    def __calculate_point2line(self, point, line_endpoint1, line_endpoint2):
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
        pb.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __get_observation(self):
        """
        get observation
        :return:
        """
        observation = self.robot.getObservation()
        # add dims for goal relative obs and base pos 3, relative collisions 3
        observation.extend(np.zeros(6))  # 56,57,58,59,60,61
        self.goal.getGoalState()
        self.ee_position = observation[0:3]
        self.base_position = observation[16:19]
        relative_pos_ee = np.subtract(self.goal.goal_position, self.ee_position)
        relative_pos_base = np.subtract(self.goal.goal_position, self.base_position)
        self.dis_ee = np.linalg.norm(relative_pos_ee)
        self.dis_base = np.linalg.norm(relative_pos_base[0:2])

        ee_vell = observation[10:13]
        ee_vela = observation[13:16]
        base_vell = observation[26:29]
        base_vela = observation[29:32]
        relative_vell_goal_ee = np.subtract(ee_vell, self.goal.goal_linear_velocity)
        relative_vela_goal_ee = np.subtract(ee_vela, self.goal.goal_angular_velocity)
        relative_vell_goal_base = np.subtract(base_vell, self.goal.goal_linear_velocity)
        relative_vela_goal_base = np.subtract(base_vela, self.goal.goal_angular_velocity)

        observation_mod = observation
        observation_mod[0:3] = relative_pos_ee
        observation_mod[16:19] = relative_pos_base
        observation_mod[18] = 0
        observation_mod[10:13] = relative_vell_goal_ee
        observation_mod[13:16] = relative_vela_goal_ee
        observation_mod[26:29] = relative_vell_goal_base
        observation_mod[29:32] = relative_vela_goal_base

        if self.if_obstacle:
            self.obstacle.getObstacleState()
            self.__check_collision_obstacle()
            observation_mod[-6:-3] = np.subtract(self.obstacle.obstacle_linear_velocity, base_vell)  # 56,57,58
            observation_mod[-3:] = self.collision_relative_position  # 59,60,61
        observation_array = np.array(observation_mod)
        # observation = self.np_random.normal(observation, 0.001, size=len(observation))
        # remove states : ee vel(10,11,12,13,14,15), joint vels 7(39,40,41,42,43,44,45)
        rm_indices = [10, 11, 12, 13, 14, 15, 39, 40, 41, 42, 43, 44, 45]
        simple_observation = np.delete(observation_array, rm_indices)
        # self.observation = simple_observation
        nobs = np.linalg.norm(simple_observation)
        if nobs == 0:
            nobs += 1e-16
        self.observation = simple_observation / nobs
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
            # tau = self.dis_ee / self.dis_ee_init
            tau = self.dis_base / self.dis_base_init if self.dis_base_init else 0
            #tau = np.cbrt(tau)#tau  # **2#
            accjoint = 0.1
            accbase = 1
            #accbase = tau
            #accjoint = 1-tau
            scaled_action[0] = input_action[0] * accjoint
            scaled_action[1] = input_action[1] * accjoint
            scaled_action[2] = input_action[2] * accjoint
            scaled_action[3] = input_action[3] * accjoint
            scaled_action[4] = input_action[4] * accjoint
            scaled_action[5] = input_action[5] * accjoint
            scaled_action[6] = input_action[6] * accjoint
            scaled_action[7] = input_action[7] * accbase
            scaled_action[8] = input_action[8] * accbase
            scaled_action[9] = input_action[9] * accbase
            #scaled_action=np.random.normal(scaled_action,0.1,10)
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
        return self.__step_shaped(scaled_action)

    def __step_shaped(self, action_scaled):
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
            pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)
            pb.stepSimulation()
            done = self.__termination()
            if done:
                self.episode_counter += 1
                self.update_step_counter += 1
                #pb.stopStateLogging(self.logvideo)
                break
            self.step_counter_per_episode += 1
        if self.terminated == 1:
            action_data_writer_file = csv.writer(open(self.data_path_action, "a"))
            action_data_writer_file.writerow(
                [self.step_counter_per_episode, action_scaled[0], action_scaled[1], action_scaled[2], action_scaled[3],
                 action_scaled[4], action_scaled[5], action_scaled[6]])

        if self.if_rendering:
            #pb.createMultiBody(baseMass=0, basePosition=self.ee_position, baseVisualShapeIndex=self.path_point_ee)
            #pb.createMultiBody(baseMass=0, basePosition=self.base_position, baseVisualShapeIndex=self.path_point_base)
            time.sleep(self.time_step)
        #print(self.dis_collision, self.collision_relative_position)
        self.input_u = np.linalg.norm(action_scaled)
        self.dis_action = np.linalg.norm(self.actions[0:7]-action_scaled[0:7], ord=np.inf)
        #print("action old : ", self.actions, action_scaled, self.actions[0:7]-action_scaled[0:7], self.dis_action)
        self.actions = action_scaled
        reward = self.__reward()
        self.__get_observation()
        return self.observation, reward, done, {}

    def __check_collision_wall(self):
        if_wall_collide = 0
        if self.if_scenario:
            closest_points1 = pb.getClosestPoints(self.robot.neobotix_schunk_uid, self.scenario.scenario_uid1, COLLISION_THRESHOLD/25)
            closest_points2 = pb.getClosestPoints(self.robot.neobotix_schunk_uid, self.scenario.scenario_uid2, COLLISION_THRESHOLD/25)
            closest_points3 = pb.getClosestPoints(self.robot.neobotix_schunk_uid, self.scenario.scenario_uid3, COLLISION_THRESHOLD/25)
            closest_points4 = pb.getClosestPoints(self.robot.neobotix_schunk_uid, self.scenario.scenario_uid4, COLLISION_THRESHOLD/25)
            closest_points5 = pb.getClosestPoints(self.robot.neobotix_schunk_uid, self.scenario.scenario_uid5, COLLISION_THRESHOLD/25)
            closest_points6 = pb.getClosestPoints(self.robot.neobotix_schunk_uid, self.scenario.scenario_uid6, COLLISION_THRESHOLD/25)
            closest_points7 = pb.getClosestPoints(self.robot.neobotix_schunk_uid, self.scenario.scenario_uid7, COLLISION_THRESHOLD/25)
            if_wall_collide = len(closest_points1)+len(closest_points2)+len(closest_points3)+len(closest_points4)+len(closest_points5)+len(closest_points6)+len(closest_points7)
        #print(if_wall_collide)
        if if_wall_collide:
            return True
        return False

    def __check_collision_obstacle(self):
        """
        check collisions with obstacle
        :return:
        """
        closest_points = []
        closest_distances = []
        self.collision_relative_position = np.zeros(3)
        if self.if_obstacle:
            closest_points = pb.getClosestPoints(self.robot.neobotix_schunk_uid, self.obstacle.obstacle_uid, COLLISION_THRESHOLD)
        self.flag_collide = len(closest_points)
        if self.flag_collide:
            for i in range(self.flag_collide):
                closest_distances.append(closest_points[i][8])
            self.dis_collision = np.min(closest_distances)
            #self.collision_probability = np.exp(100*(0.1-self.dis_collision))
            j = np.argmin(closest_distances)
            self.collision_relative_position = np.array(closest_points[j][5])-np.array(closest_points[j][6])
            #self.collision_relative_position[2] = 0
            return True
        return False

    def __termination(self):
        self.__get_observation()
        if self.__check_collision_obstacle():
            # force, d_force = self.calculateField.compute_sum_force()
            self.observation[-3:] = self.collision_relative_position#[0:2]
            if self.dis_collision > COLLISION_THRESHOLD:
                self.r_penalty_collision = -1/self.dis_collision**2#np.log(COLLISION_THRESHOLD)#-1
            elif self.dis_collision > 0.1:
                self.r_penalty_collision = -1/self.dis_collision**2#np.log(self.dis_collision)#-np.log(self.dis_collision)/np.log(COLLISION_THRESHOLD)
            else:
                self.r_penalty_collision = -1/self.dis_collision**2#np.log(0.1)#-np.log(0.1)/np.log(COLLISION_THRESHOLD)
                self.r_termination = -10000
                self.terminated = 2
                print('ACHTUNG : collision with obs!')
                return True
        if self.if_obstacle:
            self.r_function = reachingRewards.ReachingReward(with_priority=self.if_prioritized, goal=self.goal.goal_position, armpos=self.ee_position, basepos=self.base_position, opos=self.obstacle.obstacle_position)
        '''
        if self.observation[8]>1e2:
            print('ACHTUNG : unstable status!')
            self.terminated = -1
            self.r_termination = -100
            return True
        '''
        if self.__check_collision_wall():
            self.r_termination = -100
            self.terminated = 5
            print('ACHTUNG : collision with walls!')
            return True
        if self.robot.check_collision_self():
            self.terminated = 3
            self.r_termination = -10000#self.step_counter_per_episode
            print('ACHTUNG : self-collision!')
            return True
        if self.dis_ee < 0.05:#0.2/self.sound_reaching_number:#self.dis_base < 0.1
            self.r_termination = 10*self.sound_reaching_number
            self.sound_reaching_counter_per_episode += 1
            self.sound_reaching_number += 1
            steps_data_writer_file = csv.writer(open(self.data_path_steps, "a"))
            steps_data_writer_file.writerow([self.episode_counter, self.step_counter_per_episode])
            if self.sound_reaching_number == 2:
                self.terminated = 1
                self.r_termination = 1e5
                self.success_update_counter += 1
                self.total_success_counter += 1
                print('Terminate reaching at step ', self.step_counter_per_episode, ' in episode ', self.episode_counter)
                print('observation : ', self.observation, ' goal : ', self.goal.goal_position, ' distance : ', self.dis_ee)
                return True
        if self.step_counter_per_episode >= self.max_steps-1:
            print('ACHTUNG : greater than maxstep!')
            self.terminated = 4
            self.r_termination = -self.step_counter_per_episode/self.max_steps
            return True
        """
        if self.dis_base > self.ws_boundary*1.5:
            self.terminated = 5
            return True
        """
        if self.terminated:
            return True
        return False

    def __reward_delta_p_dis(self):
        """
        reward : distance between position t and position t+1
        """
        delta_dis_p_ee = np.linalg.norm(self.former_ee_pos, self.ee_position)
        delta_dis_p_base = np.linalg.norm(self.former_base_pos, self.base_position)
        delta_dis_ee = self.dis_ee - self.dis_ee_vor
        self.dis_ee_vor = self.dis_ee
        delta_dis_base = self.dis_base - self.dis_base_vor
        self.dis_base_vor = self.dis_base
        rdpde = - delta_dis_ee - delta_dis_p_ee
        rdpd = rdpde - delta_dis_base - delta_dis_p_base
        return rdpde, rdpd

    def __reward_delta_dis(self):
        """
        reward : distance between former distance and current distance
        """
        delta_dis_ee = self.dis_ee - self.dis_ee_vor
        self.dis_ee_vor = self.dis_ee
        delta_dis_base = self.dis_base - self.dis_base_vor
        self.dis_base_vor = self.dis_base
        return - delta_dis_ee - delta_dis_base

    def __reward_prioritized(self):
        """
        reward : prioritized
        """
        tau = self.dis_base / self.dis_base_init if self.dis_base_init else 0
        #tau = np.cbrt(tau)#tau#**2#
        ree = -self.dis_ee ** 2
        rbase = -self.dis_base ** 2
        if self.dis_base < 0.3:
            rp = 20*ree
        else:
            if tau < 1:
                rp = (1 - 1*tau) * ree + tau * rbase
                rp = 5*rp
            else:
                rp = rbase
                rp = rp
        return rp

    def __reward_normal(self):
        """
        reward : normal
        """
        ree = -self.dis_ee ** 2
        rbase = -self.dis_base ** 2
        return ree + rbase

    def __reward_stage_scale(self):
        """
        reward : stage scale value
        """
        rbase_scale = 10 * (round(self.dis_base * 10) + 1)
        if self.dis_ee_init-self.dis_ee>0.2*self.flag_r1 and self.dis_ee_init>0.2*self.flag_r1:
            r_stage = 10*self.flag_r1
            self.flag_r1 = self.flag_r1+1
        elif self.dis_ee-self.dis_ee_init>0.2*self.flag_r2:
            r_stage = -20*self.flag_r2
            self.flag_r2 = self.flag_r2+1
        else:
            r_stage = 0
        return r_stage

    def __reward_step(self):
        """
        reward : from step
        """
        if (self.step_counter_per_episode + 1) % 50:
            r_step = - 2 * (self.step_counter_per_episode / self.max_steps) ** 2
        else:
            r_step = - 2 * (self.step_counter_per_episode / self.max_steps) ** 2
        return r_step

    def __reward(self):
        reward = 0
        rdpde, rdpd = self.__reward_delta_p_dis()
        #rd = self.heu_reward_function.fun_gaussian(self.observation[0:2])
        #r = self.r_function.reward_field3d()
        #dline = self.__calculate_point2line(self.ee_position, self.init_ee, self.init_goal)
        delta_dis_ee = self.dis_ee - self.dis_ee_vor
        self.dis_ee_vor = self.dis_ee
        delta_dis_base = self.dis_base - self.dis_base_vor
        self.dis_base_vor = self.dis_base
        #ree = self.r_function.reward_divid(0)
        ree = -self.dis_ee**2 #np.exp(-100*self.dis_ee**2)
        rbase = -self.dis_base**2
        '''
        rline = ree-dline**2
        '''
        r_stage = 0
        k1 = 10
        #k1 = rbase_scale
        r_step = 0

        if self.reward_type == 'rdense':
            # noise = AdaptiveParamNoiseSpec(mu=0, sigma=0.1) - self.input_u**2
            reward = k1 * rdpde + self.r_termination + self.r_penalty_collision*10 + r_step + r_stage - self.dis_action
            if self.if_prioritized:
                reward = k1 * self.__reward_prioritized() + self.r_termination + self.r_penalty_collision*10 + r_step + r_stage - self.dis_action#- self.input_u**2/50
            if self.if_goal_moving_type == 'line':
                # dline_of_line^2 = dee^2 - dline^2, dline^2 = dz^2 + dy^2
                dis_line2 = self.dis_ee**2-((self.ee_position[2]-self.goal.goal_position[2])**2+(self.ee_position[1]-self.goal.goal_position[1])**2)
                reward = k1 * ree + self.r_termination + 50 * self.r_penalty_collision - 40 * self.input_u ** 2 + r_step
            if self.if_goal_moving_type == 'circle':
                dis_circle2 = self.dis_ee**2-(self.ee_position[2]-self.goal.goal_position[2])**2
                reward = k1 * ree + self.r_termination + 50 * self.r_penalty_collision - 40 * self.input_u ** 2 + r_step

        elif self.reward_type == 'rsparse':
            if delta_dis_ee > 0:
                reward = 0
            else:
                reward = 1
        #reward = np.exp(reward)/10000
        #print('--------------', self.step_counter_per_episode, self.max_steps, self.dis_ee, self.dis_base, rp, r_step, r_stage, ree, self.input_u**2, self.dis_action, reward)
        #reward = np.random.normal(reward, np.cbrt(1/self.step_counter_per_episode)-0.1)
        #reward = reward + self.base_position[1]
        return reward

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = pb.getBasePositionAndOrientation(self.robot.neobotix_schunk_uid)
        #text_goal = 'goal position : (' + str(round(self.goal_position[0], 4)) + ', ' + str(round(self.goal_position[1], 4)) + ', ' + str(round(self.goal_position[2], 4)) + ')'
        #text_ee = 'ee position : (' + str(round(self.ee_position[0], 4)) + ', ' + str(round(self.ee_position[1], 4)) + ', ' + str(round(self.ee_position[2], 4)) + ')'
        #pb.addUserDebugText(text_goal, [-3, -1, 3.4], textSize=1.5)
        #pb.addUserDebugText(text_ee, [-3, -1, 3], textSize=1.5, lifeTime=0.5)
        pb.addUserDebugLine(self.former_ee_pos, self.ee_position, [1, 0, 1], 3)
        pb.addUserDebugLine(self.former_base_pos, self.base_position, [0, 0, 0], 3)
        if self.if_goal_moving_type is not 'static':
            pb.addUserDebugLine(self.former_goal_pos, self.goal.goal_position, [1, 0, 0], 3)
        view_matrix = pb.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos, distance=self.cam_dist, yaw=self.cam_yaw, pitch=self.cam_pitch, roll=0, upAxisIndex=2)
        proj_matrix = pb.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT, nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = pb.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
        #renderer=pb.ER_TINY_RENDERER
        #pb.ER_BULLET_HARDWARE_OPENGL
        #pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step


if __name__ == "__main__":
    #如果你安装了tensorflow
    from stable_baselines.common.env_checker import check_env
    # 如果你安装了pytorch
    # from stable_baselines3.common.env_checker import check_env
    environment = NeobotixSchunkGymEnvReaching(urdf_root=PARENT_DIR,
                                               renders=True,
                                               is_discrete=False,
                                               action_repeat=1,
                                               max_steps=50,
                                               action_dim=10,
                                               ws_boundary=1,
                                               random_initial=True,
                                               if_prioritized=False,
                                               if_obstacle=True,
                                               if_obstacle_moving=True,
                                               if_goal_moving_type='static',
                                               if_scenario=False)
    check_env(environment)
    # environment = NeobotixGymEnv(renders=1, isDiscrete=False, maxSteps=2e3, actionDim=2, colliObj=0, wsBoundary=1, randomInitial=0)environment = NeobotixSchunkGymEnv(renders=1, isDiscrete=False, maxSteps=3e3, actionDim=10, colliObj=0, wsBoundary=1, randomInitial=1)
    # environment._p.startStateLogging(environment._p.STATE_LOGGING_VIDEO_MP4, "TEST_GUI.mp4")
    dv = 1
    actionIds = []
    dvalue = 0
    actionIds.append(pb.addUserDebugParameter("arm_1_joint", -dv, dv, dvalue))
    actionIds.append(pb.addUserDebugParameter("arm_2_joint", -dv, dv, dvalue))
    actionIds.append(pb.addUserDebugParameter("arm_3_joint", -dv, dv, dvalue))
    actionIds.append(pb.addUserDebugParameter("arm_4_joint", -dv, dv, dvalue))
    actionIds.append(pb.addUserDebugParameter("arm_5_joint", -dv, dv, dvalue))
    actionIds.append(pb.addUserDebugParameter("arm_6_joint", -dv, dv, dvalue))
    actionIds.append(pb.addUserDebugParameter("arm_7_joint", -dv, dv, dvalue))
    actionIds.append(pb.addUserDebugParameter("basevelocityx", -dv, dv, dvalue))
    actionIds.append(pb.addUserDebugParameter("basevelocityy", -dv, dv, dvalue))
    actionIds.append(pb.addUserDebugParameter("baseangularvelocity", -dv, dv, dvalue))

    done = 0
    n_steps = 100000

    while not False:
        environment.reset()
        disc_total_rew = 0
        t = 0
        for i in range(n_steps):
            action = []
            for actionId in actionIds:
                action.append(pb.readUserDebugParameter(actionId))
            #action = environment.action_space.sample()
            state, reward, done, info = environment.step(action)
            # print('len', i, len(state), state, info, reward)
            # state, reward, done, info = environment.step(environment._sample_action())
            # print('step', state, reward, done, info)
            # obs = environment.getExtendedObservation()
            # print(environment._p.getPhysicsEngineParameters)
            environment.render()
            disc_total_rew += reward * 0.998 ** t
            t += 1
            #massCenterLineId = pb.addUserDebugLine([environment.ee_position[0], environment.ee_position[1], 0], environment.ee_position[0:3], [1, 0, 0])
            if done:
                break
        print(disc_total_rew, t)