import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import pybullet as p

import numpy as np
import argparse
from termcolor import cprint

from pybullet_planning import BASE_LINK, RED, BLUE, GREEN
from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, has_gui, set_camera, wait_for_duration, wait_if_gui, apply_alpha
from pybullet_planning import Pose, Point, Euler
from pybullet_planning import multiply, invert, get_distance
from pybullet_planning import create_obj, create_attachment, Attachment
from pybullet_planning import link_from_name, get_link_pose, get_moving_links, get_link_name, get_disabled_collisions, \
    get_body_body_disabled_collisions, has_link, are_links_adjacent
from pybullet_planning import get_num_joints, get_joint_names, get_movable_joints, set_joint_positions, joint_from_name, \
    joints_from_names, get_sample_fn, plan_joint_motion
from pybullet_planning import dump_world, set_pose
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn, expand_links, create_box
from pybullet_planning import pairwise_collision, pairwise_collision_info, draw_collision_diagnosis, body_collision_info


NEOSCHUNK_URDF = os.path.join(parentdir, 'pybullet_neoschunk_reaching/data/neobotixschunk/neobotixschunk.urdf')
PLANE_URDF = os.path.join(parentdir, 'pybullet_neoschunk_reaching/data', "plane.urdf")
GOAL_URDF = os.path.join(parentdir, "pybullet_neoschunk_reaching/data/spheregoal.urdf")
OBS_URDF = os.path.join(parentdir, "pybullet_neoschunk_reaching/data/cylinder_verticle.urdf")

'''
(0, b'base_footprint_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'base_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
(1, b'wheel_left_joint', 0, 7, 6, 1, 0.0, 0.0, -1e+16, 1e+16, 1000.0, 3.5, b'wheel_left_link', (-1.0, 0.0, 0.0), (0.0, 0.25, 0.13), (0.0, 0.0, 0.7071066656470943, 0.7071068967259818), 0)
(2, b'wheel_right_joint', 0, 8, 7, 1, 0.0, 0.0, -1e+16, 1e+16, 1000.0, 3.5, b'wheel_right_link', (-1.0, 0.0, 0.0), (0.0, -0.25, 0.13), (0.0, 0.0, 0.7071066656470943, 0.7071068967259818), 0)
(3, b'hanger_joint', 4, -1, -1, 0, 0.0, 0.0, -1e+16, 1e+16, 1000.0, 3.5, b'hanger', (0.0, 0.0, 0.0), (-0.43, 0.0, 0.176), (0.49999999999997324, 0.49999983660255165, -0.49999999999997324, 0.5000001633974483), 0)
(4, b'arm_podest_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'arm_podest_link', (0.0, 0.0, 0.0), (-0.38, 0.0, 0.36), (0.0, 0.0, 0.0, 1.0), 0)
(5, b'arm_base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'arm_base_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.14), (0.0, 0.0, 0.0, 1.0), 4)
(6, b'arm_1_joint', 0, 9, 8, 1, 0.0, 0.0, -3.12159265359, 3.12159265359, 216.0, 0.43633, b'arm_1_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.02), (0.0, 0.0, 0.0, 1.0), 5)
(7, b'arm_2_joint', 0, 10, 9, 1, 0.9, 0.08, -2.04, 2.04, 216.0, 0.43633, b'arm_2_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.24), (0.5, 0.5, 0.49999999999755174, 0.5000000000024483), 6)
(8, b'arm_3_joint', 0, 11, 10, 1, 0.0, 0.0, -3.12159265359, 3.12159265359, 81.5, 0.4189, b'arm_3_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (-0.5, -0.49999999999755174, -0.5, 0.5000000000024483), 7)
(9, b'arm_4_joint', 0, 12, 11, 1, 0.9, 0.08, -2.08, 2.08, 81.5, 0.4189, b'arm_4_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.328), (0.5, 0.5, 0.49999999999755174, 0.5000000000024483), 8)
(10, b'arm_5_joint', 0, 13, 12, 1, 0.0, 0.0, -3.12159265359, 3.12159265359, 20.7, 0.43633, b'arm_5_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (-0.5, -0.49999999999755174, -0.5, 0.5000000000024483), 9)
(11, b'arm_6_joint', 0, 14, 13, 1, 0.0, 0.0, -2.08, 2.08, 15.0, 1.2566, b'arm_6_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.323), (0.5, 0.5, 0.49999999999755174, 0.5000000000024483), 10)
(12, b'arm_7_joint', 0, 15, 14, 1, 0.0, 0.0, -2.94, 2.94, 15.0, 1.2566, b'arm_7_link', (0.0, 0.0, 1.0), (0.0025, 0.0, 0.0), (-0.5, -0.49999999999755174, -0.5, 0.5000000000024483), 11)
(13, b'gripper_podest_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'gripper_podest_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.1334), (0.0, 0.0, 0.0, 1.0), 12)
(14, b'gripper_palm_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'gripper_palm_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.01), (0.0, 0.0, 0.0, 1.0), 13)
(15, b'gripper_finger_left_joint', 4, -1, -1, 0, 0.08, 0.08, -0.0301, -0.01, 10.0, 0.041, b'gripper_finger_left_link', (0.0, 0.0, 0.0), (0.0, -0.025, -0.026100000000000012), (0.0, 0.0, 0.0, 1.0), 14)
(16, b'gripper_finger_right_joint', 4, -1, -1, 0, 0.08, 0.08, 0.01, 0.0301, 10.0, 0.041, b'gripper_finger_right_link', (0.0, 0.0, 0.0), (0.0, 0.025, -0.026100000000000012), (0.0, 0.0, 0.0, 1.0), 14)
(17, b'grasping_frame_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'grasping_frame', (0.0, 0.0, 0.0), (0.0, 0.0, 0.2573), (0.0, 0.0, 0.0, 1.0), 12)
(18, b'laserscanner_front_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'laserscanner_front_link', (0.0, 0.0, 0.0), (0.244, 0.0, 0.141), (0.9999996829318346, 0.0, 0.0, -0.0007963267107332632), 0)
'''

def neoschunk_demo(viewer=True):
    arm='right'
    connect(use_gui=viewer)
    plane = load_pybullet(PLANE_URDF)
    robot = load_pybullet(NEOSCHUNK_URDF, fixed_base=True)
    set_camera(yaw=0.52, pitch=-0.2, distance=-0.33, target_position=(0, 7.5, 0))
    obs = load_pybullet(OBS_URDF, fixed_base=True)
    set_pose(obs, ([0, 1, 0.8], [0, 0, 0, 1]))
    cprint('hello RFL! <ctrl+left mouse> to pan', 'green')
    #wait_for_user()

    robot_self_collision_disabled_link_names = [('arm_5_link', 'arm_7_link')]
    self_collision_links = get_disabled_collisions(robot, robot_self_collision_disabled_link_names)

    base_joint_names = ['wheel_left_joint', 'wheel_right_joint']
    base_joints = joints_from_names(robot, base_joint_names)

    arm_joint_names = ['arm_{}_joint'.format(i+1) for i in range(6)]
    arm_joints = joints_from_names(robot, arm_joint_names)
    # * if a subset of joints is used, use:
    # arm_joints = joints_from_names(robot, arm_joint_names[1:]) # this will disable the gantry-x joint
    cprint('Used base joints: {}'.format(get_joint_names(robot, base_joints)), 'yellow')
    cprint('Used arm joints: {}'.format(get_joint_names(robot, arm_joints)), 'yellow')
    collision_fn = get_collision_fn(robot, arm_joints, obstacles=[obs],
                                    self_collisions=True, disabled_collisions=self_collision_links)
    # * get a joint configuration sample function:
    # it separately sample each joint value within the feasible range
    sample_fn = get_sample_fn(robot, arm_joints)
    # * now let's plan a trajectory
    # we use y-z-arm 6 joint all together here
    cprint('Randomly sample robot start/end configuration and comptue a motion plan! (no self-collision check is performed)', 'blue')
    print('Disabled collision links needs to be given (can be parsed from a SRDF via compas_fab)')
    for _ in range(5):
        print('='*10)

        q1 = list(sample_fn())

        set_joint_positions(robot, arm_joints, q1)
        cprint('Sampled start conf: {}'.format(q1), 'cyan')
        wait_for_user()

        # let it ends at the left side
        q2 = list(sample_fn())
        cprint('Sampled end conf: {}'.format(q2), 'cyan')

        path = plan_joint_motion(robot, arm_joints, q2, obstacles=[obs], self_collisions=True
                                 )
        if path is None:
            cprint('no plan found', 'red')
            continue
        else:
            wait_for_user('a motion plan is found! Press enter to start simulating!')

        # adjusting this number will adjust the simulation speed
        time_step = 0.03
        for conf in path:
            set_joint_positions(robot, arm_joints, conf)
            wait_for_duration(time_step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nv', '--noviewer', action='store_true', help='Enables the viewer during planning, default True')
    parser.add_argument('-d', '--demo', default='NeoSchunk', help='The name of the demo')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    print('Arguments:', args)

    if  args.demo == 'NeoSchunk':
        neoschunk_demo(viewer=not args.noviewer)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
