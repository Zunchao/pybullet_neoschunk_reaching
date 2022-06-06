"""
env of cylinder scenario, static and randomly move
developed by Z. Zheng, @KIT-IPR
"""

import os
import pybullet as p
import numpy as np

OBS_VEL_LIMITS = [1.5, 1.5, 0]
scenario_RADIUS = 0.1
scenario_HEIGHT = 2


class NeobotixSchunkScenario:
    def __init__(self,
                 urdf_root_path=None):
        self.urdf_root = urdf_root_path
        self._pb = p
        #self.URDF_scenario = os.path.join(self.urdf_root, "pybullet_neoschunk_reaching/data/cylinder_verticle.urdf")  # unused
        scenario_id_wall_o = self._pb.createCollisionShape(shapeType=self._pb.GEOM_BOX, halfExtents=[2, 0.05, 1])
        scenario_id_wall_v = self._pb.createVisualShape(shapeType=self._pb.GEOM_BOX, halfExtents=[2, 0.05, 1], rgbaColor=[0.1, 0.2, 0.3, 0.8])
        self.scenario_uid1 = self._pb.createMultiBody(baseMass=10000, baseCollisionShapeIndex=scenario_id_wall_o, baseVisualShapeIndex = scenario_id_wall_v, basePosition=[0, 3.05, 1])
        self.scenario_uid2 = self._pb.createMultiBody(baseMass=10000, baseCollisionShapeIndex=scenario_id_wall_o, baseVisualShapeIndex = scenario_id_wall_v, basePosition=[0, -1.05, 1])
        self.scenario_uid3 = self._pb.createMultiBody(baseMass=10000, baseCollisionShapeIndex=scenario_id_wall_o, baseVisualShapeIndex = scenario_id_wall_v, basePosition=[2.05, 1, 1], baseOrientation=[0, 0, 0.7071068, 0.7071068])
        self.scenario_uid4 = self._pb.createMultiBody(baseMass=10000, baseCollisionShapeIndex=scenario_id_wall_o, baseVisualShapeIndex = scenario_id_wall_v, basePosition=[-2.05, 1, 1], baseOrientation=[0, 0, 0.7071068, 0.7071068])
        scenario_id_wall_o2 = self._pb.createCollisionShape(shapeType=self._pb.GEOM_BOX, halfExtents=[0.75, 0.05, 0.7])
        scenario_id_wall_v2 = self._pb.createVisualShape(shapeType=self._pb.GEOM_BOX, halfExtents=[0.75, 0.05, 0.7], rgbaColor=[0.1, 0.2, 0.3, 0.8])
        self.scenario_uid5 = self._pb.createMultiBody(baseMass=10000, baseCollisionShapeIndex=scenario_id_wall_o2, baseVisualShapeIndex = scenario_id_wall_v2, basePosition=[-1.25, 1, 0.7])
        self.scenario_uid6 = self._pb.createMultiBody(baseMass=10000, baseCollisionShapeIndex=scenario_id_wall_o2, baseVisualShapeIndex = scenario_id_wall_v2, basePosition=[1.25, 1, 0.7])
        scenario_id_wall_o3 = self._pb.createCollisionShape(shapeType=self._pb.GEOM_BOX, halfExtents=[2, 0.05, 0.3])
        scenario_id_wall_v3 = self._pb.createVisualShape(shapeType=self._pb.GEOM_BOX, halfExtents=[2, 0.05, 0.3], rgbaColor=[0.1, 0.2, 0.3, 0.8])
        self.scenario_uid7 = self._pb.createMultiBody(baseMass=10000, baseCollisionShapeIndex=scenario_id_wall_o3, baseVisualShapeIndex = scenario_id_wall_v3, basePosition=[0, 1, 1.7])


        #self.scenario_uid = self._pb.loadURDF(self.URDF_scenario, self.scenario_position, useFixedBase=True)
        #self.resetScenario()

    def resetScenario(self):
        """
        reset scenario position, default for cylinder, option for sphere(commented)
        :return:
        """
        '''
        gxy = self.np_random.normal(self.goal_position[0:2], 0.3, size=2)
        xopos = self.np_random.uniform(0, gxy[0])  # rx, self.goal_position[0]) #+ 0.20
        yopos = self.np_random.uniform(0, gxy[1])  # self.np_random.uniform(ry, self.goal_position[1])
        zopos = 0.8  # self.np_random.uniform(0, 1.5)
        self.scenario_position = np.array([xopos, yopos, zopos])

        self.scenario_position = np.array(
            [self.np_random.uniform(-self.ws_boundary, self.ws_boundary),
             self.np_random.uniform(-self.ws_boundary, self.ws_boundary),
             scenario_HEIGHT/2])
        self._pb.resetBasePositionAndOrientation(self.scenario_uid, self.scenario_position, np.array([0, 0, 0, 1]))
        '''

    def setScenarioState(self):
        """
        get scenario state : scenario velocity and position relative to base
        :return:
        if self.if_scenario_moving:
            #v = self._pb.getBaseVelocity(self.scenario_uid)
            #self.scenario_linear_velocity[0:2] = np.array(v[0][0:2])
            #self.scenario_linear_velocity[2] = v[1][2]
            self.scenario_linear_velocity += np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5), 0])
            self.scenario_linear_velocity = np.clip(self.scenario_linear_velocity, -np.array(OBS_VEL_LIMITS), np.array(OBS_VEL_LIMITS))
            self._pb.resetBaseVelocity(objectUniqueId=self.scenario_uid,
                                linearVelocity=self.scenario_linear_velocity,
                                angularVelocity=self.scenario_angular_velocity)
        """

    def getScenarioState(self):
        """
        get scenario state : scenario velocity and position relative to base
        :return:
        obsvel = self._pb.getBaseVelocity(self.scenario_uid)
        self.scenario_linear_velocity = np.array(obsvel[0])
        obspos, obsorn = self._pb.getBasePositionAndOrientation(self.scenario_uid)
        self.scenario_position = np.array(obspos)
        """

