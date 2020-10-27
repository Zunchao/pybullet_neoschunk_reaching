'''
original built by X. Wang & Z. Zheng, @KIT-IPR
developed by Z. Zheng
schunk model meshes source : https://github.com/ipa320/schunk_modular_robotics
neobotix model meshed source : https://github.com/neobotix/neo_mp_500
model modified by Y. Zhang and J. Su.
'''
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

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
from env import reachingRewards
from env import heuristicReward
#from env import fieldDirection
#from ddpg.ddpg_noise import NormalActionNoise, AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise

SUCCESS_STEPS_UPDATE = 1000  # parameter for update success rate every SUCCESS_STEPS_UPDATE during the training
largeValObservation = 100
RENDER_HEIGHT = 720
RENDER_WIDTH = 960
OBS_VEL_LIMITS = [1.5, 1.5, 0]
GOAL_VEL_LIMITS = [1, 1, 1.2]
OBSTACLE_RADIUS = 0.1
OBSTACLE_HEIGHT = 2
GOAL_RADIUS = 0.05
PATH_POINT_RADIUS = 0.01


class NeobotixSchunkGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdf_root=parentdir,
                 action_repeat=1,
                 time_step=1. / 240.,
                 enable_self_collision_flag=True,
                 is_discrete=False,
                 renders=False,
                 max_steps=1e3,
                 reward_type='rdense',
                 action_dim=9,
                 random_initial=True,
                 ws_boundary=1,
                 if_obstacle=False,
                 if_obstacle_moving=False,
                 if_goal_moving=False):
        self.urdf_root = urdf_root
        self.action_repeat = action_repeat
        self.enable_self_collision_flag = enable_self_collision_flag
        self.is_discrete = is_discrete
        self.if_rendering = renders
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.action_dim = action_dim
        self.is_random_init = random_initial
        self.ws_boundary = ws_boundary
        self.if_obstacle = if_obstacle
        self.if_obstacle_moving = if_obstacle_moving
        self.if_goal_moving = if_goal_moving
        self.observation = []
        self.step_counter_per_episode = 0
        self.sound_reaching_counter_per_episode = 0
        self.time_step = time_step
        self.r_termination = 0
        self.r_penalty_collision = 0
        self.terminated = 0
        self.cam_dist = 4
        self.cam_yaw = 180
        self.cam_pitch = -40
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
        self.flag_collide = 0
        self.collision_probability = 0
        self.with_prioritized_reward = False
        self.actions = []
        self.u = 0
        self.goal_position = np.zeros(3)
        self.goal_orientation = np.array([0, 0, 0, 1])
        self.obstacle_position = np.ones(3)
        self.sound_reaching_number = 1
        self.obstacle_linear_velocity = np.zeros(3)
        self.obstacle_angular_velocity = np.zeros(3)
        self.goal_linear_velocity = np.zeros(3)
        self.goal_angular_velocity = np.zeros(3)
        self.seed_number = 0
        self.np_random = None
        self._p = p
        self.former_ee_pos = []
        self.former_base_pos = []
        self.obstacle_state = []
        self.goal_state = []

        self.path_point_base = None
        self.path_point_ee = None

        self.obstacle_uid = None
        self.goal_uid = None
        self.neobotix_schunk = None
        self.ground_uid = None

        self.heu_reward_function = None
        self.URDF_GROUND = os.path.join(self.urdf_root, "pybullet_neoschunk_reaching/data/plane.urdf")
        self.URDF_GOAL = os.path.join(self.urdf_root, "pybullet_neoschunk_reaching/data/spheregoal.urdf")
        self.URDF_OBSTACLE = os.path.join(self.urdf_root, "pybullet_neoschunk_reaching/data/cylinder_verticle.urdf")
        if self.if_rendering:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # disable tinyrenderer, software (CPU) renderer, we don't use it here
            #p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
            #p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            #p.resetDebugVisualizerCamera(self.cam_dist, self.cam_yaw, self.cam_pitch, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        self.seed_number = self.seed()
        #self.DATA_SUCCESS_RATE = os.path.join(self.urdf_root,'pybullet_neoschunk_reaching/results/success_rate_update_noobs_'+str(self.seed_number[0])+'.csv')
        self.DATA_SUCCESS_RATE = os.path.join(self.urdf_root, 'pybullet_neoschunk_reaching/results/success_rate_update_obs29d_server_0928no.csv')
        self.DATA_ACTION = os.path.join(self.urdf_root, 'pybullet_neoschunk_reaching/results/success_rate_update_obs29d_actions_server_0928no.csv')
        self.DATA_STEPS = os.path.join(self.urdf_root, 'pybullet_neoschunk_reaching/results/success_rate_update_obs29d_steps_server_0928no.csv')

        if self.if_obstacle_moving:
            self.if_obstacle = True

        self.reset()

        self.collision_relative_position = np.zeros(3)
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
        self.obstacle_linear_velocity = np.zeros(3)
        self.obstacle_angular_velocity = np.zeros(3)
        self.goal_linear_velocity = np.zeros(3)
        self.goal_angular_velocity = np.zeros(3)

    def reset(self):
        """
        reset env
        :return:
        """
        self.reset_params()
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=200, enableFileCaching=0)
        p.setPhysicsEngineParameter(solverResidualThreshold=1e-30)
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(False)

        #video_path = os.path.join(self.urdf_root,'pybullet_neoschunk_reaching/results/videos/video_'+str(self.episode_counter)+'.mp4')
        #open(video_path, 'a')
        #self.logvideo = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
        if self.if_rendering:
            self.path_point_base = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=PATH_POINT_RADIUS, rgbaColor=[0, 0, 1, 0.9])
            self.path_point_ee = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=PATH_POINT_RADIUS, rgbaColor=[1, 0, 0, 0.9])

        self.ground_uid = p.loadURDF(self.URDF_GROUND, [0, 0, -0.001], useFixedBase=True, flags=p.URDF_ENABLE_SLEEPING)

        self.goal_position = np.array([self.np_random.uniform(-self.ws_boundary, self.ws_boundary), self.np_random.uniform(-self.ws_boundary, self.ws_boundary), self.np_random.uniform(0.5, 1.5)])
        #self.goal_uid = p.loadURDF(self.URDF_GOAL, basePosition=self.goal_position)
        goal_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=GOAL_RADIUS,  rgbaColor=[1, 0, 0, 0.5])
        self.goal_uid = p.createMultiBody(baseMass=0, basePosition=self.goal_position, baseVisualShapeIndex=goal_id)
        #goal_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=GOAL_RADIUS)
        #self.goal_uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=goal_id, basePosition=[0, 0, 0])
        self.resetGoal()

        self.neobotix_schunk = neobotixschunk.NeobotixSchunk(urdf_root_path=self.urdf_root, time_step=self.time_step, ws_boundary=self.ws_boundary)
        if self.is_random_init:
            self.resetRandomRobot()

        if self.if_obstacle:
            #self.obstacle_uid = p.loadURDF(self.URDF_OBSTACLE)
            #obstacle_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=OBSTACLE_RADIUS)
            #self.obstacle_uid = p.createMultiBody(baseMass=0, basePosition=[0, 0, 0], baseCollisionShapeIndex=obstacle_id)
            #p.changeDynamics(self.obstacle_uid, -1, spinningFriction=0.001, rollingFriction=0.001, linearDamping=0.0)
            obstacle_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=OBSTACLE_RADIUS, height=OBSTACLE_HEIGHT)
            self.obstacle_position = np.array([self.np_random.uniform(-self.ws_boundary, self.ws_boundary), self.np_random.uniform(-self.ws_boundary, self.ws_boundary), 1])
            self.obstacle_uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_id, basePosition=self.obstacle_position)
            self.resetObstacle()

        while True:
            flag1 = self.check_collision_self() #or (np.linalg.norm(self.goal_position-[-0.38, 0.0, 0.76])>0.6)  # distance between goal and 2nd link
            if self.if_obstacle:
                self.check_collision_obs()
                flag2 = (self.dis_collision < 0.1)
                flag3 = (np.linalg.norm(self.goal_position[0:2] - self.obstacle_position[0:2]) < OBSTACLE_RADIUS)
                flag1 = (flag1 or flag2 or flag3)
            if flag1:
                self.resetGoal()
                if self.is_random_init:
                    self.resetRandomRobot()
                if self.if_obstacle:
                    self.resetObstacle()
            else:
                break

        p.stepSimulation()
        self.observation = self.getExtendedObservation()
        self.former_ee_pos = self.ee_position
        self.former_base_pos = self.base_position

        self.dis_ee_init = np.linalg.norm(np.subtract(self.ee_position, self.goal_position))
        self.dis_base = np.linalg.norm(np.subtract(self.base_position, self.goal_position))
        self.dis_ee = self.dis_ee_init
        self.dis_vor = self.dis_ee_init
        #self.heu_reward_function = heuristicReward.HeuristicReward(self.ee_position[0:2], self.goal_position[0:2], self.obstacle_position[0:2])
        data_writer_file = csv.writer(open(self.DATA_SUCCESS_RATE, "a"))

        if self.episode_counter:
            success_rate_update = 0
            if self.update_step_counter:
                success_rate_update = self.success_update_counter / self.update_step_counter
            success_rate_total = self.total_success_counter / self.episode_counter
            data_writer_file.writerow([self.episode_counter, self.total_success_counter, success_rate_total, self.update_step_counter, self.success_update_counter, success_rate_update])
            if not self.episode_counter%SUCCESS_STEPS_UPDATE:
                self.success_update_counter = 0
                self.update_step_counter = 0
            print('Episode :', self.episode_counter,
                  ' SuccessTotal :', self.total_success_counter,
                  ' success_rate_total :', success_rate_total,
                  f' StepUpdate : {self.update_step_counter}/{SUCCESS_STEPS_UPDATE}',
                  f' SuccessUpdate : {self.success_update_counter}/{SUCCESS_STEPS_UPDATE}',
                  ' success_rate_updateUpdate :', success_rate_update)

        return self.observation

    def resetRandomRobot(self):
        """
        reset random arm joint positions
        :return:
        """
        # reset arm joint positions and controllers
        j1 = self.np_random.uniform(-self.neobotix_schunk.j1_limit, self.neobotix_schunk.j1_limit)
        j2 = self.np_random.uniform(-self.neobotix_schunk.j2_limit, self.neobotix_schunk.j2_limit)
        j3 = self.np_random.uniform(-self.neobotix_schunk.j1_limit, self.neobotix_schunk.j1_limit)
        j4 = self.np_random.uniform(-self.neobotix_schunk.j4_limit, self.neobotix_schunk.j4_limit)
        j5 = self.np_random.uniform(-self.neobotix_schunk.j1_limit, self.neobotix_schunk.j1_limit)
        j6 = self.np_random.uniform(-self.neobotix_schunk.j4_limit, self.neobotix_schunk.j4_limit)
        j7 = self.np_random.uniform(-self.neobotix_schunk.j7_limit, self.neobotix_schunk.j7_limit)
        initial_joint_positions = np.array([j1, j2, j3, j4, j5, j6, j7])
        for j in range(len(self.neobotix_schunk.active_arm_index)):
            p.resetJointState(self.neobotix_schunk.neobotix_schunk_uid,
                              jointIndex=self.neobotix_schunk.active_arm_index[j],
                              targetValue=initial_joint_positions[j],
                              targetVelocity=0)
        # do not need random set base if relative positions are used
        '''
        # initial_joint_positions = np.zeros(len(self.neobotix_schunk.active_arm_index))
        bpos, born = p.getBasePositionAndOrientation(self.neobotix_schunk.neobotix_schunk_uid)
        initial_basep = np.array([self.np_random.uniform(-self.ws_boundary, self.ws_boundary),
                                  self.np_random.uniform(-self.ws_boundary, self.ws_boundary),
                                  bpos[2]])
        initial_basea = np.array([0, 0, self.np_random.uniform(-np.pi, np.pi)])
        initial_baseo = p.getQuaternionFromEuler(initial_basea)
        
        p.resetBasePositionAndOrientation(self.neobotix_schunk.neobotix_schunk_uid, initial_basep, initial_baseo)
        p.resetBaseVelocity(self.neobotix_schunk.neobotix_schunk_uid, np.zeros(3), np.zeros(3))
        '''
        return initial_joint_positions

    def resetGoal(self):
        """
        reset goal position
        :return:
        """
        self.goal_position = np.array(
            [self.np_random.uniform(-self.ws_boundary, self.ws_boundary), self.np_random.uniform(-self.ws_boundary, self.ws_boundary),
             self.np_random.uniform(0.5, 1.5)])
        p.resetBasePositionAndOrientation(self.goal_uid, self.goal_position, np.array([0, 0, 0, 1]))

    def setGoalState(self):
        """
        get goal state : goal velocity
        :return:
        """
        if self.if_goal_moving:
            self.goal_linear_velocity += np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5)])
            self.goal_linear_velocity = np.clip(self.goal_linear_velocity, -np.array(GOAL_VEL_LIMITS), np.array(GOAL_VEL_LIMITS))
            p.resetBaseVelocity(objectUniqueId=self.goal_uid,
                                linearVelocity=self.goal_linear_velocity,
                                angularVelocity=self.goal_angular_velocity)

    def getGoalState(self):
        """
        get goal state : goal velocity
        :return:
        """
        basev = p.getBaseVelocity(self.goal_uid)
        pos, orn = p.getBasePositionAndOrientation(self.goal_uid)
        self.goal_position = np.array(pos)
        #print(self.goal_uid, self.goal_linear_velocity, pos, self.goal_position)
        if self.goal_position[2] > 1.5:
            self.goal_position[2] = 1.5
        if self.goal_position[2] < 0.5:
            self.goal_position[2] = 0.5
        self.goal_state = np.array(basev[0])

    def resetObstacle(self):
        """
        reset obstacle position, default for cylinder, option for sphere(commented)
        :return:
        """
        '''
        gxy = self.np_random.normal(self.goal_position[0:2], 0.3, size=2)
        xopos = self.np_random.uniform(0, gxy[0])  # rx, self.goal_position[0]) #+ 0.20
        yopos = self.np_random.uniform(0, gxy[1])  # self.np_random.uniform(ry, self.goal_position[1])
        zopos = 0.8  # self.np_random.uniform(0, 1.5)
        self.obstacle_position = np.array([xopos, yopos, zopos])
        '''
        self.obstacle_position = np.array(
            [self.np_random.uniform(-self.ws_boundary, self.ws_boundary), 
             self.np_random.uniform(-self.ws_boundary, self.ws_boundary),
             OBSTACLE_HEIGHT/2])
        p.resetBasePositionAndOrientation(self.obstacle_uid, self.obstacle_position, np.array([0, 0, 0, 1]))

    def setObstacleState(self):
        """
        get obstacle state : obstacle velocity and position relative to base
        :return:
        """
        if self.if_obstacle_moving:
            #v = p.getBaseVelocity(self.obstacle_uid)
            #self.obstacle_linear_velocity[0:2] = np.array(v[0][0:2])
            #self.obstacle_linear_velocity[2] = v[1][2]
            self.obstacle_linear_velocity += np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5), 0])
            self.obstacle_linear_velocity = np.clip(self.obstacle_linear_velocity, -np.array(OBS_VEL_LIMITS), np.array(OBS_VEL_LIMITS))
            p.resetBaseVelocity(objectUniqueId=self.obstacle_uid,
                                linearVelocity=self.obstacle_linear_velocity,
                                angularVelocity=self.obstacle_angular_velocity)


    def getObstacleState(self):
        """
        get obstacle state : obstacle velocity and position relative to base
        :return:
        """
        if self.if_obstacle_moving:
            obsvel = p.getBaseVelocity(self.obstacle_uid)
            self.obstacle_linear_velocity = np.array(obsvel[0])
            obspos, obsorn = p.getBasePositionAndOrientation(self.obstacle_uid)
            self.obstacle_position = np.array(obspos)
            #print(self.obstacle_uid, self.obstacle_linear_velocity, obspos, self.obstacle_position)
            #if self.obstacle_position[2]<OBSTACLE_RADIUS:
                #self.obstacle_position[2] = OBSTACLE_RADIUS
                #p.resetBasePositionAndOrientation(self.obstacle_uid, self.obstacle_position, np.array([0, 0, 0, 1]))
            obs_state = np.array([self.obstacle_linear_velocity, self.obstacle_position-self.base_position])
            #obs_state = self.obstacle_linear_velocity[0:2]
        else:
            obs_state = np.array([np.zeros(3), self.obstacle_position-self.base_position])
            #obs_state = np.zeros(2)

        self.obstacle_state = obs_state.flatten()
        self.obstacle_state[5] = 0
        #return obs_state.flatten()

    def close(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        """
        get observation
        :return:
        """
        observation = self.neobotix_schunk.getObservation()
        self.getGoalState()
        self.ee_position = observation[0:3]
        self.base_position = observation[6:9]

        relative_pos_ee = np.subtract(self.goal_position, self.ee_position)
        observation[0:3] = relative_pos_ee

        relative_pos_base = np.subtract(self.goal_position, self.base_position)
        observation[6:9] = relative_pos_base

        self.dis_ee = np.linalg.norm(relative_pos_ee)
        self.dis_base = np.linalg.norm(relative_pos_base)
        self.collision_relative_position = np.zeros(3)

        observation.extend(np.zeros(12))
        observation = np.array(observation)
        # observation = self.np_random.normal(observation, 0.001, size=len(observation))
        # remove some states
        observation = np.delete(observation, [3, 4, 5, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37])
        observation[-12:-9] = self.goal_state

        if self.if_obstacle:
            self.getObstacleState()
            observation[-9:-3] = self.obstacle_state
            observation[-3:] = self.collision_relative_position

        # remove zero elements
        zero_index = [6, 7, 27, 30, 33]
        self.observation = np.delete(observation, zero_index)
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
            accjoint = 0.5
            scaled_action[0] = input_action[0] * accjoint
            scaled_action[1] = input_action[1] * accjoint
            scaled_action[2] = input_action[2] * accjoint
            scaled_action[3] = input_action[3] * accjoint
            scaled_action[4] = input_action[4] * accjoint
            scaled_action[5] = input_action[5] * accjoint
            scaled_action[6] = input_action[6] * accjoint

            scaled_action[7] = input_action[7]*0.3#/0.13
            scaled_action[8] = input_action[8]*0.3#/0.13
            scaled_action[9] = input_action[9]*0.4#
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
        for i in range(self.action_repeat):
            self.setGoalState()
            self.setObstacleState()
            self.neobotix_schunk.applyAction(action_scaled)
            p.stepSimulation()
            done = self._termination()
            if done:
                self.episode_counter += 1
                self.update_step_counter += 1
                #p.stopStateLogging(self.logvideo)
                break
            self.step_counter_per_episode += 1

        if self.terminated==1:
            action_data_writer_file = csv.writer(open(self.DATA_ACTION, "a"))
            action_data_writer_file.writerow(
                [self.step_counter_per_episode, action_scaled[0], action_scaled[1], action_scaled[2], action_scaled[3],
                 action_scaled[4], action_scaled[5]])

        if self.if_rendering:
            p.createMultiBody(baseMass=0, basePosition=self.ee_position, baseVisualShapeIndex=self.path_point_ee)
            p.createMultiBody(baseMass=0, basePosition=self.base_position, baseVisualShapeIndex=self.path_point_base)
            time.sleep(self.time_step)
        #print('state after step ', len(self.observation), self.observation)
        self.u = np.linalg.norm(action_scaled)
        self.actions = action_scaled

        nobs = np.linalg.norm(self.observation)
        if nobs == 0:
            nobs += 1e-16
        #self.observation = self.observation / nobs
        reward = self._reward()
        return self.observation, reward, done, {}

    def render(self, mode='rgb_array', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = p.getBasePositionAndOrientation(self.neobotix_schunk.neobotix_schunk_uid)
        #text_goal = 'goal position : (' + str(round(self.goal_position[0], 4)) + ', ' + str(round(self.goal_position[1], 4)) + ', ' + str(round(self.goal_position[2], 4)) + ')'
        #text_ee = 'ee position : (' + str(round(self.ee_position[0], 4)) + ', ' + str(round(self.ee_position[1], 4)) + ', ' + str(round(self.ee_position[2], 4)) + ')'
        #p.addUserDebugText(text_goal, [-3, -1, 3.4], textSize=1.5)
        #p.addUserDebugText(text_ee, [-3, -1, 3], textSize=1.5, lifeTime=0.5)
        #p.addUserDebugLine(self.former_ee_pos, self.ee_position, [1, 0, 0], 3)
        #p.addUserDebugLine(self.former_base_pos, self.base_position, [0, 0, 0], 3)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos, distance=self.cam_dist, yaw=self.cam_yaw, pitch=self.cam_pitch, roll=0, upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT, nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = p.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        #renderer=p.ER_TINY_RENDERER
        #p.ER_BULLET_HARDWARE_OPENGL
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def check_collision_obs(self):
        """
        check collisions with obstacle
        :return:
        """
        closest_points = []
        closest_distances = []
        if self.if_obstacle:
            closest_points = p.getClosestPoints(self.neobotix_schunk.neobotix_schunk_uid, self.obstacle_uid, 2000)
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

    def check_collision_self(self):
        """
        check self-collisions
        :return:
        """
        '''
        for i in self.neobotix_schunk.checkCollisonIndex:
            dcontact = p.getContactPoints(self.neobotix_schunk.neobotix_schunk_uid, self.neobotix_schunk.neobotix_schunk_uid, i)
            if len(dcontact):
                #print('self collision!', dcontact)
                return True
        '''
        dcontact = p.getContactPoints(self.neobotix_schunk.neobotix_schunk_uid, self.neobotix_schunk.neobotix_schunk_uid)
        if len(dcontact):
            #print('c ', dcontact)
            return True
        return False

    def _termination(self):
        self.observation = self.getExtendedObservation()
        self.r_func = reachingRewards.ReachingReward(with_priority=self.with_prioritized_reward, goal=self.goal_position, armpos=self.ee_position, basepos=self.base_position, opos=self.obstacle_position)
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
            if self.dis_collision > 0.1:
                self.r_penalty_collision = np.log(self.dis_collision/(self.dis_collision+1))
            else:
                self.r_termination = -100
                self.terminated = 2
                print('ACHTUNG : collision with obs!')
                return True

        if self.check_collision_self():
            self.terminated = 3
            self.r_termination = -100
            print('ACHTUNG : self-collision!')
            return True

        if self.dis_ee < 0.05:#0.2/self.sound_reaching_number:
            self.r_termination = 10*self.sound_reaching_number
            self.sound_reaching_counter_per_episode += 1
            self.sound_reaching_number += 1
            steps_data_writer_file = csv.writer(open(self.DATA_STEPS, "a"))
            steps_data_writer_file.writerow([self.episode_counter, self.step_counter_per_episode])
            if self.sound_reaching_number == 2:
                self.terminated = 1
                self.r_termination = 200
                self.success_update_counter += 1
                self.total_success_counter += 1
                print('Terminate reaching at step ', self.step_counter_per_episode, ' in episode ', self.episode_counter)
                print('observation : ', self.observation, ' goal : ', self.goal_position, ' distance : ', self.dis_ee)
                return True

        if self.step_counter_per_episode >= self.max_steps-1:
            print('ACHTUNG : greater than maxstep!')
            self.terminated = 4
            self.r_termination = -10
            return True

        if self.terminated:
            return True

        return False

    def _reward(self):
        #rd = self.heu_reward_function.fun_gaussian(self.observation[0:2])
        #r = self.r_func.reward_field3d()

        delta_dis = self.dis_ee - self.dis_vor
        self.dis_vor = self.dis_ee

        tau = self.dis_ee/self.dis_ee_init
        tau = tau**2#np.cbrt(tau)
        #ree = self.r_func.reward_divid(0)
        ree = -self.dis_ee**2
        rbase = -self.dis_base**2

        if self.dis_base < 0.3:
            rp = ree
        else:
            if tau < 1:
                rp = (1 - tau) * ree + tau * rbase
                rp = 10*rp
            else:
                rp = rbase
                rp = 20*rp

        if self.reward_type == 'rdense':
            if delta_dis < 0:
                f_reward = 1
            else:
                f_reward = 0
            # noise = AdaptiveParamNoiseSpec(mu=0, sigma=0.1)
            reward = 100*ree + self.r_termination + 50*self.r_penalty_collision - 40*self.u**2 - 5*(self.step_counter_per_episode/self.max_steps)**2
            reward = reward/1000
        elif self.reward_type == 'rsparse':
            if delta_dis > 0:
                reward = 0
            else:
                reward = 1
        return reward

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
