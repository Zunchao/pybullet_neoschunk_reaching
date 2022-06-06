"""
env of cylinder obstacle, static and randomly move
developed by Z. Zheng, @KIT-IPR
"""

import os
import pybullet as p
import numpy as np

OBS_VEL_LIMITS = [1.5, 1.5, 0]
OBSTACLE_RADIUS = 0.1
OBSTACLE_HEIGHT = 2


class NeobotixSchunkObstacle:
    def __init__(self,
                 urdf_root_path=None,
                 ws_boundary=1,
                 rseed=None,
                 if_obstacle_moving=False):
        self.urdf_root = urdf_root_path
        self.ws_boundary = ws_boundary
        self.np_random = rseed
        self.if_obstacle_moving = if_obstacle_moving
        self._pb = p
        self.obstacle_position = np.ones(3)
        self.obstacle_linear_velocity = np.zeros(3)
        self.obstacle_angular_velocity = np.zeros(3)
        self.URDF_OBSTACLE = os.path.join(self.urdf_root, "pybullet_neoschunk_reaching/data/cylinder_verticle.urdf")  # unused
        obstacle_id = self._pb.createCollisionShape(shapeType=self._pb.GEOM_CYLINDER, radius=OBSTACLE_RADIUS, height=OBSTACLE_HEIGHT)
        #self.obstacle_uid = self._pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_id, basePosition=self.obstacle_position)
        #self.obstacle_uid = self._pb.loadURDF(self.URDF_OBSTACLE, self.obstacle_position, useFixedBase=True)
        obs_idc = self._pb.createCollisionShape(shapeType=self._pb.GEOM_SPHERE, radius=0.15)
        obs_idv = self._pb.createVisualShape(shapeType=self._pb.GEOM_SPHERE, radius=0.15, rgbaColor=[0.0, 0.0, 1, 1])
        self.obstacle_uid = self._pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=obs_idc,
                                                      baseVisualShapeIndex=obs_idv,
                                                      basePosition=[0, 0, 1])

        self.resetObstacle()

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
        self._pb.resetBasePositionAndOrientation(self.obstacle_uid, self.obstacle_position, np.array([0, 0, 0, 1]))

    def setObstacleState(self):
        """
        get obstacle state : obstacle velocity and position relative to base
        :return:
        """
        if self.if_obstacle_moving:
            #v = self._pb.getBaseVelocity(self.obstacle_uid)
            #self.obstacle_linear_velocity[0:2] = np.array(v[0][0:2])
            #self.obstacle_linear_velocity[2] = v[1][2]
            self.obstacle_linear_velocity += np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5), 0])
            self.obstacle_linear_velocity = np.clip(self.obstacle_linear_velocity, -np.array(OBS_VEL_LIMITS), np.array(OBS_VEL_LIMITS))
            self._pb.resetBaseVelocity(objectUniqueId=self.obstacle_uid,
                                linearVelocity=self.obstacle_linear_velocity,
                                angularVelocity=self.obstacle_angular_velocity)

    def getObstacleState(self):
        """
        get obstacle state : obstacle velocity and position relative to base
        :return:
        """
        if self.if_obstacle_moving:
            obsvel = self._pb.getBaseVelocity(self.obstacle_uid)
            self.obstacle_linear_velocity = np.array(obsvel[0])
            obspos, obsorn = self._pb.getBasePositionAndOrientation(self.obstacle_uid)
            self.obstacle_position = np.array(obspos)

