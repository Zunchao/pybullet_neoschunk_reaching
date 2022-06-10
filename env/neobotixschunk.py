"""
original built by X. Wang & Z. Zheng, @KIT-IPR
developed and maintained by Z. Zheng
schunk model meshes source : https://github.com/ipa320/schunk_modular_robotics
neobotix model meshed source : https://github.com/neobotix/neo_mp_500
model modified by Y. Zhang and J. Su.
"""
"""
(0, b'base_footprint_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'base_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
(1, b'wheel_left_joint', 0, 7, 6, 1, 0.0, 0.0, -1e+16, 1e+16, 1000.0, 3.5, b'wheel_left_link', (-1.0, 0.0, 0.0), (0.0, 0.25, 0.13), (0.0, 0.0, 0.7071066656470943, 0.7071068967259818), 0)
(2, b'wheel_right_joint', 0, 8, 7, 1, 0.0, 0.0, -1e+16, 1e+16, 1000.0, 3.5, b'wheel_right_link', (-1.0, 0.0, 0.0), (0.0, -0.25, 0.13), (0.0, 0.0, 0.7071066656470943, 0.7071068967259818), 0)
(3, b'hanger_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'hanger', (0.0, 0.0, 0.0), (-0.43, 0.0, 0.176), (0.49999999999997324, 0.49999983660255165, -0.49999999999997324, 0.5000001633974483), 0)
(4, b'arm_podest_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'arm_podest_link', (0.0, 0.0, 0.0), (-0.38, 0.0, 0.36), (0.0, 0.0, 0.0, 1.0), 0)
(5, b'arm_base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'arm_base_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.12000000000000001), (0.0, 0.0, 0.0, 1.0), 4)
(6, b'arm_1_joint', 0, 9, 8, 1, 0.5, 0.0, -3.12159265359, 3.12159265359, 216.0, 0.43633, b'arm_1_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.02), (0.0, 0.0, 0.0, 1.0), 5)
(7, b'arm_2_joint', 0, 10, 9, 1, 0.5, 0.0, -2.04, 2.04, 216.0, 0.43633, b'arm_2_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.22), (0.5, 0.5, 0.4999999999999999, 0.5000000000000001), 6)
(8, b'arm_3_joint', 0, 11, 10, 1, 0.5, 0.0, -3.12159265359, 3.12159265359, 81.5, 0.4189, b'arm_3_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (-0.5, -0.4999999999999999, -0.5, 0.5000000000000001), 7)
(9, b'arm_4_joint', 0, 12, 11, 1, 0.5, 0.0, -2.08, 2.08, 81.5, 0.4189, b'arm_4_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.528), (0.5, 0.5, 0.4999999999999999, 0.5000000000000001), 8)
(10, b'arm_5_joint', 0, 13, 12, 1, 0.5, 0.0, -3.12159265359, 3.12159265359, 20.7, 0.43633, b'arm_5_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (-0.5, -0.4999999999999999, -0.5, 0.5000000000000001), 9)
(11, b'arm_6_joint', 0, 14, 13, 1, 0.5, 0.0, -2.94, 2.94, 15.0, 1.2566, b'arm_6_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.503), (0.5, 0.5, 0.4999999999999999, 0.5000000000000001), 10)
(12, b'arm_7_joint', 0, 15, 14, 1, 0.5, 0.0, -2.94, 2.94, 15.0, 1.2566, b'arm_7_link', (0.0, 0.0, 1.0), (0.0025, 0.0, 0.0), (-0.5, -0.4999999999999999, -0.5, 0.5000000000000001), 11)
(13, b'gripper_podest_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'gripper_podest_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.07339999999999999), (0.0, 0.0, 0.0, 1.0), 12)
(14, b'gripper_palm_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'gripper_palm_link', (0.0, 0.0, 0.0), (0.0, 0.0, -0.008), (0.0, 0.0, 0.0, 1.0), 13)
(15, b'gripper_finger_left_joint', 4, -1, -1, 0, 0.08, 0.08, -0.0301, -0.01, 10.0, 0.041, b'gripper_finger_left_link', (0.0, 0.0, 0.0), (0.0, -0.025, -0.026100000000000012), (0.0, 0.0, 0.0, 1.0), 14)
(16, b'gripper_finger_right_joint', 4, -1, -1, 0, 0.08, 0.08, 0.01, 0.0301, 10.0, 0.041, b'gripper_finger_right_link', (0.0, 0.0, 0.0), (0.0, 0.025, -0.026100000000000012), (0.0, 0.0, 0.0, 1.0), 14)
(17, b'grasping_frame_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'grasping_frame', (0.0, 0.0, 0.0), (0.0, 0.0, -0.026100000000000012), (0.0, 0.0, 0.0, 1.0), 14)
(18, b'laserscanner_front_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'laserscanner_front_link', (0.0, 0.0, 0.0), (0.244, 0.0, 0.141), (0.9999996829318346, 0.0, 0.0, -0.0007963267107332632), 0)
"""
import os
import pybullet as p
import numpy as np

URDF_USE_SELF_COLLISION = 1
URDF_USE_SELF_COLLISION_EXCLUDE_PARENT = 1  # unused
URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS = 1  # unused

WHEELDIAMETER = 0.2675
BASELENGTH = 0.676
BASEWIDTH = 0.507
BASE_VEL_LIMITS = [1, 1, 1.2]
ARM_JOINT_VEL_LIMITS = [0.43633, 0.43633, 0.4189, 0.4189, 0.43633, 1.2566, 1.2566]


class NeobotixSchunk:
    def __init__(self, urdf_root_path=None, ws_boundary=1, rseed=None):
        self.urdf_root_path = urdf_root_path
        self.np_random = rseed
        self.max_force = 1000
        self.max_velocity = 1.5  # unused yet
        self.use_simulation = 1  # unused yet
        self.use_null_space = 0  # unused yet
        self.use_orientation = 1  # unused yet
        self.j1_limit = np.pi  # limits for arm link 1, 3, 5
        self.j2_limit = 116 / 180 * np.pi  # limits for arm link 2
        self.j4_limit = 119 / 180 * np.pi  # limits for arm link 4
        self.j6_limit = 118 / 180 * np.pi  # limits for arm link 6
        self.j7_limit = 170 / 180 * np.pi  # limits for arm link 7
        self.joint_position = []
        self.joint_velocity = []
        self.base_velocity = []
        self.wheel_index = []
        self.wheel_velocity = []
        self.active_arm_index = []
        self.end_effector_index = []
        self.collision_check_index = []
        self.ws_range = ws_boundary
        self.neobotix_schunk_uid = None
        # joint damping coefficents
        self.jd = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
        self.URDF_FILE_ROBOT = os.path.join(
            self.urdf_root_path,
            "pybullet_neoschunk_reaching/data/neobotixschunk/neobotixschunk.urdf",
        )
        self.pb = p
        self.reset()

    def reset(self):
        # load robot model
        self.neobotix_schunk_uid = self.pb.loadURDF(
            self.URDF_FILE_ROBOT,
            useFixedBase=True,
            flags=self.pb.URDF_USE_SELF_COLLISION,
        )  # mp500lwa4d.urdf or neobotixschunk.urdf
        joint_names = {}
        for i in range(self.pb.getNumJoints(self.neobotix_schunk_uid)):
            joint_info = self.pb.getJointInfo(self.neobotix_schunk_uid, i)
            joint_names[joint_info[1].decode("UTF-8")] = joint_info[0]
            # print(joint_info)

        self.j1_limit = self.pb.getJointInfo(self.neobotix_schunk_uid, 6)[9] - 0.01  # limits for arm link 1, 3, 5
        self.j2_limit = self.pb.getJointInfo(self.neobotix_schunk_uid, 7)[9] - 0.01  # limits for arm link 2
        self.j4_limit = self.pb.getJointInfo(self.neobotix_schunk_uid, 9)[9] - 0.01  # limits for arm link 4, 6
        self.j7_limit = self.pb.getJointInfo(self.neobotix_schunk_uid, 12)[9] - 0.01  # limits for arm link 7
        id_wheel_left_joint = joint_names["wheel_left_joint"]
        id_wheel_right_joint = joint_names["wheel_right_joint"]

        id_armpodest_joint = joint_names["arm_podest_joint"]
        id_arm_base_joint = joint_names["arm_base_joint"]
        id_arm_1_joint = joint_names["arm_1_joint"]
        id_arm_2_joint = joint_names["arm_2_joint"]
        id_arm_3_joint = joint_names["arm_3_joint"]
        id_arm_4_joint = joint_names["arm_4_joint"]
        id_arm_5_joint = joint_names["arm_5_joint"]
        id_arm_6_joint = joint_names["arm_6_joint"]
        id_arm_7_joint = joint_names["arm_7_joint"]

        id_gripper_joint = joint_names["grasping_frame_joint"]
        id_grasping_frame_joint = joint_names["gripper_palm_joint"]
        id_gripper_l_joint = joint_names["gripper_finger_left_joint"]
        id_gripper_r_joint = joint_names["gripper_finger_right_joint"]

        id_base_joint = joint_names["base_footprint_joint"]
        id_laser_joint = joint_names["laserscanner_front_joint"]

        self.wheel_index = [id_wheel_left_joint, id_wheel_right_joint]
        self.active_arm_index = [
            id_arm_1_joint,
            id_arm_2_joint,
            id_arm_3_joint,
            id_arm_4_joint,
            id_arm_5_joint,
            id_arm_6_joint,
            id_arm_7_joint,
        ]
        self.end_effector_index = id_grasping_frame_joint
        self.collision_check_index = [
            id_gripper_joint,
            id_gripper_l_joint,
            id_gripper_r_joint,
            id_arm_base_joint,
            id_arm_1_joint,
            id_arm_2_joint,
            id_arm_3_joint,
            id_arm_4_joint,
            id_arm_5_joint,
            id_arm_6_joint,
            id_arm_7_joint,
            id_base_joint,
            id_laser_joint,
            id_armpodest_joint,
        ]
        # disable collision between link 10 and 12 : arm link 5 and 7
        self.pb.setCollisionFilterPair(
            self.neobotix_schunk_uid,
            self.neobotix_schunk_uid,
            self.active_arm_index[4],
            self.active_arm_index[-1],
            enableCollision=0,
        )
        # self.pb.createConstraint(self.neobotix_schunk_uid, -1, self.neobotix_schunk_uid, 5, self.pb.JOINT_FIXED, [0, 0, 0], [0.19, 0, 0.5], [0., 0., 0])

        initial_wheel_vel = np.zeros(len(self.wheel_index))
        self.base_velocity = np.zeros(3)
        self.wheel_velocity = np.zeros(2)
        initial_basep = np.zeros(3)
        initial_basep[0] = 0.0
        initial_basep[1] = -0.0
        initial_baseo = np.array([0, 0, 0, 1])
        self.pb.resetBasePositionAndOrientation(
            self.neobotix_schunk_uid, initial_basep, initial_baseo
        )
        self.pb.resetBaseVelocity(
            objectUniqueId=self.neobotix_schunk_uid,
            linearVelocity=self.base_velocity,
            angularVelocity=self.base_velocity,
        )

        self.joint_position = np.zeros(len(self.active_arm_index))
        self.joint_velocity = np.zeros(len(self.active_arm_index))
        """
        self.pb.setJointMotorControlArray(bodyIndex=self.neobotix_schunk_uid,
                                    jointIndices=self.active_arm_index,
                                    controlMode=self.pb.VELOCITY_CONTROL,
                                    targetVelocities=self.joint_velocity,
                                    forces=len(self.active_arm_index)*[self.max_force])
        
        for i in range(len(self.wheel_index)):
            self.pb.resetJointState(self.neobotix_schunk_uid,
                              jointIndex=self.wheel_index[i],
                              targetValue=initial_wheel_vel[i],
                              targetVelocity=initial_wheel_vel[i])
            self.pb.setJointMotorControl2(bodyUniqueId=self.neobotix_schunk_uid,
                                    jointIndex=self.wheel_index[i],
                                    controlMode=self.pb.VELOCITY_CONTROL,
                                    targetVelocity=initial_wheel_vel[i],
                                    force=self.max_force)
        """

        for j in range(len(self.active_arm_index)):
            self.pb.resetJointState(
                self.neobotix_schunk_uid,
                jointIndex=self.active_arm_index[j],
                targetValue=self.joint_position[j],
                targetVelocity=0,
            )
        """
        self.pb.setJointMotorControl2(bodyUniqueId=self.neobotix_schunk_uid,
                                jointIndex=self.active_arm_index[j],
                                controlMode=self.pb.POSITION_CONTROL,
                                targetPosition=self.joint_position[j],
                                maxVelocity=0)
        
        self.pb.enableJointForceTorqueSensor(bodyUniqueId=self.neobotix_schunk_uid,
                                       jointIndex=self.active_arm_index[j],
                                       enableSensor=True)
        """

    def resetRandomRobotState(self):
        """
        reset random arm joint positions
        :return:
        """
        # reset arm joint positions and controllers
        j1 = self.np_random.uniform(-self.j1_limit, self.j1_limit)
        j2 = self.np_random.uniform(-self.j2_limit, self.j2_limit)
        j3 = self.np_random.uniform(-self.j1_limit, self.j1_limit)
        j4 = self.np_random.uniform(-self.j4_limit, self.j4_limit)
        j5 = self.np_random.uniform(-self.j1_limit, self.j1_limit)
        j6 = self.np_random.uniform(-self.j4_limit, self.j4_limit)
        j7 = self.np_random.uniform(-self.j7_limit, self.j7_limit)
        self.joint_position = np.array([j1, j2, j3, j4, j5, j6, j7])
        for j in range(len(self.active_arm_index)):
            self.pb.resetJointState(
                self.neobotix_schunk_uid,
                jointIndex=self.active_arm_index[j],
                targetValue=self.joint_position[j],
                targetVelocity=0,
            )
        # do not need random set base if relative positions are used
        """
        # initial_joint_positions = np.zeros(len(self.active_arm_index))
        bpos, born = self.pb.getBasePositionAndOrientation(self.neobotix_schunk_uid)
        initial_basep = np.array([self.np_random.uniform(-self.ws_range, self.ws_range),
                                  self.np_random.uniform(-self.ws_range, self.ws_range),
                                  bpos[2]])
        initial_basea = np.array([0, 0, self.np_random.uniform(-np.pi, np.pi)])
        initial_baseo = self.pb.getQuaternionFromEuler(initial_basea)        
        self.pb.resetBasePositionAndOrientation(self.neobotix_schunk_uid, initial_basep, initial_baseo)        
        """
        self.pb.resetBaseVelocity(self.neobotix_schunk_uid, np.zeros(3), np.zeros(3))

    def getActionDimension(self):
        return len(self.active_arm_index) + len(self.wheel_index)

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        """
        read state of robot
        :return: ee position, base pose, joint angles, ee pose in base frame
        """
        observation = []
        # get ee pose and vel
        ee_link_state = self.pb.getLinkState(
            self.neobotix_schunk_uid,
            linkIndex=self.end_effector_index,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
        )
        pos = ee_link_state[0]  # position x y z
        quat = ee_link_state[1]  # quaternion x y z w
        euler = self.pb.getEulerFromQuaternion(quat)  # euler r p y
        vell = ee_link_state[6]
        vela = ee_link_state[7]
        observation.extend(list(pos))  # 0,1,2
        observation.extend(list(quat))  # 3,4,5,6
        observation.extend(list(euler))  # 7,8,9
        observation.extend(list(vell))  # 10,11,12
        observation.extend(list(vela))  # 13,14,15
        # get base pose
        basepos, basequat = self.pb.getBasePositionAndOrientation(self.neobotix_schunk_uid)
        baseeul = self.pb.getEulerFromQuaternion(basequat)
        observation.extend(list(basepos))  # 16,17,18
        observation.extend(list(basequat))  # 19,20,21,22
        observation.extend(list(baseeul))  # 23,24,25
        # get base linear and angular vel
        basev = self.pb.getBaseVelocity(self.neobotix_schunk_uid)
        observation.extend(list(basev[0]))  # 26,27,28
        observation.extend(list(basev[1]))  # 29,30,31
        # get joint positions and velocities
        joint_s = self.pb.getJointStates(bodyUniqueId=self.neobotix_schunk_uid, jointIndices=self.active_arm_index)
        joints = [x[0] for x in joint_s]
        jointv = [x[1] for x in joint_s]
        observation.extend(list(joints))  # 32,33,34,35,36,37,38
        observation.extend(list(jointv))  # 39,40,41,42,43,44,45
        # get ee pose in base frame
        ee_base_pos, ee_base_quat, ee_base_euler = self.calculate_in_base_frame(pos, quat)
        observation.extend(ee_base_pos)  # 46,47,48
        observation.extend(ee_base_quat)  # 49,50,51,52
        observation.extend(ee_base_euler)  # 53,54,55
        """
        for i in self.active_arm_index:
            links = self.pb.getLinkState(self.neobotix_schunk_uid, linkIndex=i)
            arm_base_pos, arm_base_ori = self.pb.multiplyTransforms(base_world_pos_transform_invert,
                                                            base_world_ori_transform_invert, links[4], links[5])
            arm_base_ang = self.pb.getEulerFromQuaternion(arm_base_ori)
            observation.extend(arm_base_pos)
            observation.extend(arm_base_ang)
            #print(joints)
            
        ee_rotation = self.pb.getMatrixFromQuaternion(self.pb.getQuaternionFromEuler(euler))
        ee_transform = np.array([ee_rotation[0], ee_rotation[1], ee_rotation[2], pos[0],
                                 ee_rotation[3], ee_rotation[4], ee_rotation[5], pos[1],
                                 ee_rotation[6], ee_rotation[7], ee_rotation[8], pos[2],
                                 0, 0, 0, 1])
        ee_transforms = ee_transform.reshape(4, 4)
        ee_base_transform = bm_transform_inv.dot(ee_transforms)
        print('transform of ee in base frame : ', ee_base_transform)
        #ee_base_rotation =

        bm_pose_world = np.array([basepos[0], basepos[1], basepos[2], 1])
        print('base position in world : ', bm_pose_world)
        bm_pose = bm_transform_inv.dot(bm_pose_world)
        #print('base position in base frame : ', bm_pose)
        bm_ori_world = np.array([baseeul[0], baseeul[1], baseeul[2], 1])
        print('base orientation in world : ', bm_ori_world)
        bm_ori = bm_transform_inv.dot(bm_ori_world)
        print('base orientation in base frame : ', bm_ori)
        ee_pose_world = np.array([pos[0], pos[1], pos[2], 1])
        #print('ee position in world frame : ', ee_pose_world)
        ee_base_pos = bm_transform_inv.dot(ee_pose_world)[0:3]
        print('ee position in base frame : ', ee_base_pos)
        ee_ori_world = np.array([euler[0], euler[1], euler[2], 1])
        print('ee orientation in world frame : ', ee_ori_world)
        ee_ori_base = bm_transform_inv.dot(ee_ori_world)[0:3]
        print('ee orientation in base frame : ', ee_ori_base)
        
        # print(self.pb.getLinkState(self.neobotix_schunk_uid, 7))
        # for i in self.active_arm_index:
        # joints = self.pb.getJointState(bodyUniqueId=self.neobotix_schunk_uid, jointIndex=i)
        # observation.append((joints[0]))
        # observation.append((joints[1]))
        # print(joints)
        # for i in self.wheel_index:
        # vt = self.pb.getJointState(self.neobotix_schunk_uid, jointIndex=i)
        # observation.append((vt[1]))
        """
        return observation

    def calculate_in_base_frame(self, pos, quat):
        """
        calculate a pose in base frame : T(ee,base)=T(ee,world)*T(base,world).invert
        :param pos:
        :param orn:
        :return:
        """
        basepos, basequat = self.pb.getBasePositionAndOrientation(self.neobotix_schunk_uid)
        base_world_pos_transform_invert, base_world_quat_transform_invert = self.pb.invertTransform(basepos, basequat)
        in_base_pos, in_base_quat = self.pb.multiplyTransforms(pos, quat, base_world_pos_transform_invert, base_world_quat_transform_invert)
        in_base_euler = self.pb.getEulerFromQuaternion(in_base_quat)
        return in_base_pos, in_base_quat, in_base_euler

    def calculate_in_non_world_frame(self, i, pos, orn):
        """
        calculate a pose in link i frame
        :param i:
        :param pos:
        :param orn:
        :return:
        """
        links = self.pb.getLinkState(self.neobotix_schunk_uid, linkIndex=i)
        i_world_pos_transform_invert, i_world_ori_transform_invert = self.pb.invertTransform(links[4], links[5])
        i_pos, i_ori = self.pb.multiplyTransforms(i_world_pos_transform_invert, i_world_ori_transform_invert, pos, orn)
        i_ang = self.pb.getEulerFromQuaternion(i_ori)
        return i_pos, i_ang

    def check_base_velocity(self, base_vel):
        """
        BASE_VEL_LIMITS = [0.4, 0.4, 0.5]
        :param base_vel:
        :param delta_bv:
        :return:
        """
        """
        if np.abs(base_vel[0]) > 0.4:
            base_vel[0] = np.sign(base_vel[0])*0.4# - delta_bv[0]
        if np.abs(base_vel[1]) > 0.4:
            base_vel[1] = np.sign(base_vel[1])*0.4# - delta_bv[1]
        if np.abs(base_vel[2]) > 0.5:
            base_vel[2] = np.sign(base_vel[2])*0.5# - delta_bv[1]
        """
        base_vel = np.clip(base_vel, -np.array(BASE_VEL_LIMITS), np.array(BASE_VEL_LIMITS))
        return base_vel

    def check_joint_states(self, joint_state, vels):
        # joint limits from lwa4d data sheet and modified based on rviz visual
        flag = 0
        if np.abs(joint_state[0]) > self.j1_limit:
            joint_state[0] = np.sign(joint_state[0]) * self.j1_limit
            if np.sign(joint_state[0]) == np.sign(vels[0]):
                vels[0] = 0
            flag += 1
        if np.abs(joint_state[1]) > self.j2_limit:
            joint_state[1] = np.sign(joint_state[1]) * self.j2_limit
            if np.sign(joint_state[1]) == np.sign(vels[1]):
                vels[1] = 0
            flag += 1
        if np.abs(joint_state[2]) > self.j1_limit:
            joint_state[2] = np.sign(joint_state[2]) * self.j1_limit
            if np.sign(joint_state[2]) == np.sign(vels[2]):
                vels[2] = 0
            flag += 1
        if np.abs(joint_state[3]) > self.j4_limit:
            joint_state[3] = np.sign(joint_state[3]) * self.j4_limit
            if np.sign(joint_state[3]) == np.sign(vels[3]):
                vels[3] = 0
            flag += 1
        if np.abs(joint_state[4]) > self.j1_limit:
            joint_state[4] = np.sign(joint_state[4]) * self.j1_limit
            if np.sign(joint_state[4]) == np.sign(vels[4]):
                vels[4] = 0
            flag += 1
        if np.abs(joint_state[5]) > self.j4_limit:
            joint_state[5] = np.sign(joint_state[5]) * self.j4_limit
            if np.sign(joint_state[5]) == np.sign(vels[5]):
                vels[5] = 0
            flag += 1
        if np.abs(joint_state[6]) > self.j7_limit:
            joint_state[6] = np.sign(joint_state[6]) * self.j7_limit
            if np.sign(joint_state[6]) == np.sign(vels[6]):
                vels[6] = 0
            flag += 1
        return joint_state, vels, flag

    def check_joint_vels(self, vels):
        """
        vel_limits = [0.43633, 0.43633, 0.4189, 0.4189, 0.43633,1.2566, 1.2566]
        joint vel limits from lwa4d data sheet and modified based on rviz visual
        :param vels: joint vels
        :return: clipped joint vels
        """
        vels = np.clip(vels, -np.array(ARM_JOINT_VEL_LIMITS), np.array(ARM_JOINT_VEL_LIMITS))
        return vels

    def rotation_matrix(self, ang):
        """
        Calculates a rotation matrix around z-axis.
        :param ang: Angle to rotate.
        :return: 3x3 rotation matrix
        """
        return np.array(
            [[np.cos(ang), -np.sin(ang), 0],
             [np.sin(ang), +np.cos(ang), 0],
             [0, 0, 1]]
        )

    def check_collision_self(self):
        """
        check self-collisions
        :return: true or false
        """
        """
        for i in self.robot.checkCollisonIndex:
            dcontact = self.pb.getContactPoints(self.robot.neobotix_schunk_uid, self.robot.neobotix_schunk_uid, i)
            if len(dcontact):
                #print('self collision!', dcontact)
                return True
        """
        if len(self.pb.getContactPoints(self.neobotix_schunk_uid, self.neobotix_schunk_uid)):
            return True
        return False

    def applyAction(self, action):
        """
        execute
        :param action:
        :return:
        """
        if len(action) == 10:
            self.applyAction_joint(action)
        elif len(action) == 6:
            self.applyAction_ee(action)
        elif len(action) == 9:
            self.applyAction_ee_ori(action)

    def applyAction_joint(self, action):
        """
        apply action of joint angles difference, and base vel difference
        :param action:
        :return:
        """
        """
        gripper_l_state = self.pb.getLinkState(bodyUniqueId=self.neobotix_schunk_uid, linkIndex=self.collision_check_index[1])
        gripper_l_pos = gripper_l_state
        gripper_r_state = self.pb.getLinkState(bodyUniqueId=self.neobotix_schunk_uid, linkIndex=self.collision_check_index[2])
        gripper_r_pos = gripper_r_state
        """

        basev = self.pb.getBaseVelocity(self.neobotix_schunk_uid)
        self.base_velocity[0] = basev[0][0]
        self.base_velocity[1] = basev[0][1]
        self.base_velocity[2] = basev[1][2]
        self.base_velocity += action[7:10]
        self.base_velocity = self.check_base_velocity(self.base_velocity)
        base_pos, base_orn = self.pb.getBasePositionAndOrientation(self.neobotix_schunk_uid)
        b_rotation = self.pb.getMatrixFromQuaternion(base_orn)
        bm_rotation = np.array(b_rotation).reshape(3, 3)
        #self.base_velocity = bm_rotation.dot(self.base_velocity)
        vel = np.array([self.base_velocity[0], self.base_velocity[1], 0])
        ang = np.array([0, 0, self.base_velocity[2]])

        """
        mb_link_state1 = self.pb.getLinkState(self.neobotix_schunk_uid, 5, False, False)
        mb_ang_w1 = self.pb.getEulerFromQuaternion(mb_link_state1[5])[2]
        mb_link_state = self.pb.getLinkState(self.neobotix_schunk_uid, 0, False, False)
        mb_ang_w = self.pb.getEulerFromQuaternion(mb_link_state[5])[2]
        self.base_velocity = self.rotation_matrix(mb_ang_w).dot(self.base_velocity)
        """
        jointstates = self.pb.getJointStates(self.neobotix_schunk_uid, jointIndices=self.active_arm_index)
        jpos = [x[0] for x in jointstates]
        jvel = [x[1] for x in jointstates]
        self.joint_position = jpos
        self.joint_velocity = jvel
        self.joint_position += action[0:7]
        #self.joint_velocity += action[0:7]
        #self.joint_velocity = self.check_joint_vels(self.joint_velocity)
        self.joint_position, self.joint_velocity, flag = self.check_joint_states(self.joint_position, self.joint_velocity)

        self.pb.resetBaseVelocity(
            objectUniqueId=self.neobotix_schunk_uid,
            linearVelocity=vel,
            angularVelocity=ang,
        )
        for j in range(len(self.active_arm_index)):
            self.pb.resetJointState(
                self.neobotix_schunk_uid,
                jointIndex=self.active_arm_index[j],
                targetValue=self.joint_position[j],
                targetVelocity=0,
            )
        """
        self.pb.setJointMotorControlArray(bodyIndex=self.neobotix_schunk_uid,
                                          jointIndices=self.active_arm_index,
                                          controlMode=self.pb.VELOCITY_CONTROL,
                                          targetVelocities=self.joint_velocity,
                                          forces=len(self.active_arm_index)*[self.max_force])
        
        self.pb.setJointMotorControlArray(
            bodyIndex=self.neobotix_schunk_uid,
            jointIndices=self.active_arm_index,
            controlMode=self.pb.POSITION_CONTROL,
            targetPositions=self.joint_position,
            forces=len(self.active_arm_index) * [self.max_force],
        )  # , maxVelocity=0.43633 , positionGain=1e-5, velocityGain=1e-5)
    """

    def applyAction_ee(self, action):
        """
        apply action of joint ee position difference, and base vel difference
        :param action:
        :return:
        """
        """
        basev = self.pb.getBaseVelocity(self.neobotix_schunk_uid)
        self.base_velocity[0:2] = np.array(basev[0][0:2])
        self.base_velocity[2] = basev[1][2]
        print('0', basev)
        """
        self.base_velocity += action[3:6]
        self.base_velocity = self.check_base_velocity(self.base_velocity)
        base_pos, base_orn = self.pb.getBasePositionAndOrientation(self.neobotix_schunk_uid)
        b_rotation = self.pb.getMatrixFromQuaternion(base_orn)
        bm_rotation = np.array(b_rotation).reshape(3, 3)
        self.base_velocity = bm_rotation.dot(self.base_velocity)
        vel = np.array([self.base_velocity[0], self.base_velocity[1], 0])
        ang = np.array([0, 0, self.base_velocity[2]])
        """
        bm_transform = np.array([b_rotation[0], b_rotation[1], b_rotation[2], base_pos[0],
                                 b_rotation[3], b_rotation[4], b_rotation[5], base_pos[1],
                                 b_rotation[6], b_rotation[7], b_rotation[8], base_pos[2],
                                 0, 0, 0, 1])
        bm_transforms = bm_transform.reshape(4, 4)
        bm_transform_inv = np.linalg.inv(bm_transforms)
        """
        ee_link_state = self.pb.getLinkState(
            self.neobotix_schunk_uid, linkIndex=self.end_effector_index
        )
        actual_ee_position = ee_link_state[0]
        """
        ee_pose_world = np.array([actual_ee_position[0], actual_ee_position[1], actual_ee_position[2], 1])
        actual_ee_position = bm_transform_inv.dot(ee_pose_world)[0:3]
        """
        actual_ee_position += action[0:3]
        actual_ee_position[2] = np.clip(actual_ee_position[2], 0.45, 1.65)
        joint_position = self.pb.calculateInverseKinematics(
            self.neobotix_schunk_uid, self.end_effector_index, actual_ee_position
        )
        self.joint_position = joint_position[2:9]

        self.pb.resetBaseVelocity(
            objectUniqueId=self.neobotix_schunk_uid,
            linearVelocity=vel,
            angularVelocity=ang,
        )

        self.pb.setJointMotorControlArray(
            bodyIndex=self.neobotix_schunk_uid,
            jointIndices=self.active_arm_index,
            controlMode=self.pb.POSITION_CONTROL,
            targetPositions=self.joint_position,
            forces=len(self.active_arm_index) * [self.max_force],
        )  # , maxVelocity=0.43633 , positionGain=1e-5, velocityGain=1e-5)

    def applyAction_ee_ori(self, action):
        """
        apply action of joint ee pose difference, and base vel difference
        :param action:
        :return:
        """
        self.base_velocity += action[6:9]
        self.base_velocity = self.check_base_velocity(self.base_velocity)
        base_pos, base_orn = self.pb.getBasePositionAndOrientation(
            self.neobotix_schunk_uid
        )
        b_rotation = self.pb.getMatrixFromQuaternion(base_orn)
        bm_rotation = np.array(b_rotation).reshape(3, 3)
        self.base_velocity = bm_rotation.dot(self.base_velocity)
        vel = np.array([self.base_velocity[0], self.base_velocity[1], 0])
        ang = np.array([0, 0, self.base_velocity[2]])
        """
        bm_transform = np.array([b_rotation[0], b_rotation[1], b_rotation[2], base_pos[0],
                                 b_rotation[3], b_rotation[4], b_rotation[5], base_pos[1],
                                 b_rotation[6], b_rotation[7], b_rotation[8], base_pos[2],
                                 0, 0, 0, 1])
        bm_transforms = bm_transform.reshape(4, 4)
        bm_transform_inv = np.linalg.inv(bm_transforms)
        """
        ee_link_state = self.pb.getLinkState(
            self.neobotix_schunk_uid, linkIndex=self.end_effector_index
        )
        actual_ee_position = ee_link_state[0]
        actual_ee_orientation = self.pb.getEulerFromQuaternion(ee_link_state[1])
        """
        ee_pose_world = np.array([actual_ee_position[0], actual_ee_position[1], actual_ee_position[2], 1])
        actual_ee_position = bm_transform_inv.dot(ee_pose_world)[0:3]
        """
        actual_ee_position += action[0:3]
        actual_ee_orientation += action[3:6]
        # actual_ee_position[0] = np.clip(actual_ee_position[0], -1.2, 0.5)
        # actual_ee_position[1] = np.clip(actual_ee_position[1], -0.8, 0.8)
        actual_ee_position[2] = np.clip(actual_ee_position[2], 0.45, 1.65)
        actual_ee_orientation = np.clip(actual_ee_orientation, -np.ones(3) * np.pi, np.ones(3) * np.pi)
        actual_ee_orientation = self.pb.getQuaternionFromEuler(actual_ee_orientation)
        joint_position = self.pb.calculateInverseKinematics(
            self.neobotix_schunk_uid,
            self.end_effector_index,
            actual_ee_position,
            actual_ee_orientation,
            jointDamping=self.jd,
        )
        self.joint_position = joint_position[2:9]

        self.pb.resetBaseVelocity(
            objectUniqueId=self.neobotix_schunk_uid,
            linearVelocity=vel,
            angularVelocity=ang,
        )

        self.pb.setJointMotorControlArray(
            bodyIndex=self.neobotix_schunk_uid,
            jointIndices=self.active_arm_index,
            controlMode=self.pb.POSITION_CONTROL,
            targetPositions=self.joint_position,
            forces=len(self.active_arm_index) * [self.max_force],
        )  # , maxVelocity=0.43633 , positionGain=1e-5, velocityGain=1e-5)
