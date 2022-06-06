"""
env of sphere goal, static and moving(random, line, circle).
developed by Z. Zheng, @KIT-IPR
"""

import os
import pybullet as p
import numpy as np

GOAL_VEL_LIMITS = [2.5, 2.5, 2.5]
GOAL_VEL = 0.8
GOAL_RADIUS = 0.05
GOAL_HEIGHT_LOW = 0.5
GOAL_HEIGHT_HIGH = 1.5
TRACKING_STEP = 0.005
LINE_VEL = 1.0
CIRCLE_R = 0.5
CIRCLE_H = 1
CIRCLE_VEL = 2.0

class NeobotixSchunkGoal:
    def __init__(self,
                 urdf_root_path=None,
                 ws_boundary=1,
                 rseed=None,
                 if_goal_moving_type='static',
                 p_goal_start=np.zeros(3)):
        self.urdf_root = urdf_root_path
        self.ws_boundary = ws_boundary
        self.np_random = rseed
        self.if_goal_moving_type = if_goal_moving_type
        self.goal_position = p_goal_start
        self.goal_orientation = np.array([0, 0, 0, 1])
        self.goal_linear_velocity = np.zeros(3)
        self.goal_angular_velocity = np.zeros(3)
        self.goal_start = p_goal_start
        self.goal_circle_r = 0
        self._pb = p
        self.URDF_GOAL = os.path.join(self.urdf_root, "pybullet_neoschunk_reaching/data/spheregoal.urdf")  # unused
        # self.goal_uid = self._pb.loadURDF(self.URDF_GOAL, basePosition=self.goal_position)
        goal_id = self._pb.createVisualShape(shapeType=self._pb.GEOM_SPHERE, radius=GOAL_RADIUS, rgbaColor=[1, 0, 0, 0.5])
        self.goal_uid = self._pb.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_id, basePosition=self.goal_position)
        self.goal_orientation = np.array([0, 0, 0, 1])
        self.resetGoal()

    def resetGoal(self):
        """
        reset goal position
        :return:
        """
        if self.if_goal_moving_type == 'static':
            self.goal_position = np.array([self.np_random.uniform(-self.ws_boundary, self.ws_boundary), self.np_random.uniform(-self.ws_boundary, self.ws_boundary), self.np_random.uniform(GOAL_HEIGHT_LOW, GOAL_HEIGHT_HIGH)])
            #self.goal_position = np.array([0, 0, 1])
        '''
        if self.if_goal_moving_type == 'line':
            self.goal_position = np.array([0, 0, 1])  # initial position for line tracking
        if self.if_goal_moving_type == 'circle':
            self.goal_position = np.array([CIRCLE_R, 0, CIRCLE_H])  # initial position for circle tracking
        'random' or 
        '''
        self._pb.resetBasePositionAndOrientation(self.goal_uid, self.goal_position, np.array([0, 0, 0, 1]))
        self.goal_circle_r = np.sqrt(self.goal_position[0]**2 + self.goal_position[1]**2)

    def resetGoalScenario(self):
        """
        reset goal position
        :return:
        """
        #self.goal_position = np.array([self.np_random.uniform(-self.ws_boundary, self.ws_boundary), self.np_random.uniform(-self.ws_boundary, self.ws_boundary), self.np_random.uniform(GOAL_HEIGHT_LOW, GOAL_HEIGHT_HIGH)])
        self.goal_position = np.array([0, 2.3, 1])
        '''
        if self.if_goal_moving_type == 'line':
            self.goal_position = np.array([0, 0, 1])  # initial position for line tracking
        if self.if_goal_moving_type == 'circle':
            self.goal_position = np.array([CIRCLE_R, 0, CIRCLE_H])  # initial position for circle tracking
        'random' or 
        '''
        self._pb.resetBasePositionAndOrientation(self.goal_uid, self.goal_position, np.array([0, 0, 0, 1]))
        self.goal_circle_r = np.sqrt(self.goal_position[0]**2 + self.goal_position[1]**2)

    def resetGoalScenarioRandom(self):
        """
        reset goal position
        :return:
        """
        self.goal_position = np.array([self.np_random.uniform(-self.ws_boundary, self.ws_boundary), self.np_random.uniform(1.5, 2.5), self.np_random.uniform(GOAL_HEIGHT_LOW, GOAL_HEIGHT_HIGH)])
        self._pb.resetBasePositionAndOrientation(self.goal_uid, self.goal_position, np.array([0, 0, 0, 1]))

    def setGoalState(self):
        """
        set goal state by type
        :return:
        """
        if self.if_goal_moving_type == 'random' or 'static':
            self.setGoalStateRandom()
        if self.if_goal_moving_type == 'line':
            self.setGoalStateInLine()
        if self.if_goal_moving_type == 'circle':
            self.setGoalStateInCircle()

    def getGoalState(self):
        """
        get goal state : goal velocity
        :return:
        """
        basev = self._pb.getBaseVelocity(self.goal_uid)
        pos, orn = self._pb.getBasePositionAndOrientation(self.goal_uid)
        self.goal_position = np.array(pos)
        if self.goal_position[2] > GOAL_HEIGHT_HIGH:
            self.goal_position[2] = GOAL_HEIGHT_HIGH
        if self.goal_position[2] < GOAL_HEIGHT_LOW:
            self.goal_position[2] = GOAL_HEIGHT_LOW
        self.goal_linear_velocity = np.array(basev[0])
        self.goal_angular_velocity = np.array(basev[1])
        self._pb.resetBasePositionAndOrientation(self.goal_uid, self.goal_position, orn)

    def setGoalStateRandom(self):
        """
        set goal moving randomly direction
        :return:
        """
        if self.if_goal_moving_type == 'random':
            self.goal_linear_velocity += np.array([self.np_random.uniform(-GOAL_VEL, GOAL_VEL), self.np_random.uniform(-GOAL_VEL, GOAL_VEL), self.np_random.uniform(-GOAL_VEL, GOAL_VEL)])
            self.goal_linear_velocity = np.clip(self.goal_linear_velocity, -np.array(GOAL_VEL_LIMITS), np.array(GOAL_VEL_LIMITS))
        self._pb.resetBaseVelocity(objectUniqueId=self.goal_uid,
                                   linearVelocity=self.goal_linear_velocity,
                                   angularVelocity=self.goal_angular_velocity)

    def setGoalStateInLine(self):
        """
        set goal moving in direction of a line, line start endpoint = (0,0,1)
        initial self.goal_position=(0,0,1) required
        :return:
        """
        self.goal_linear_velocity[0] = LINE_VEL
        self._pb.resetBaseVelocity(objectUniqueId=self.goal_uid,
                                   linearVelocity=self.goal_linear_velocity,
                                   angularVelocity=self.goal_angular_velocity)

    def setGoalStateInCircle(self):
        """
        set goal moving along a circle start from (CIRCLE_R, 0, CIRCLE_H)
        initial self.goal_start required
        center of the circle is [0, 0, 0] in world
        :return:
        """
        pos, orn = self._pb.getBasePositionAndOrientation(self.goal_uid)
        arm_base_ang = self._pb.getEulerFromQuaternion(orn)
        self.goal_linear_velocity[0] = CIRCLE_VEL*np.sin(arm_base_ang[2])
        self.goal_linear_velocity[1] = -CIRCLE_VEL*np.cos(arm_base_ang[2])
        self.goal_angular_velocity[2] = CIRCLE_VEL/CIRCLE_R
        self._pb.resetBaseVelocity(objectUniqueId=self.goal_uid,
                                   linearVelocity=self.goal_linear_velocity,
                                   angularVelocity=self.goal_angular_velocity)
