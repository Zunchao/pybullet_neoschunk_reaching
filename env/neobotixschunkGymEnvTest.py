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

from env.neobotixschunkGymEnv import NeobotixSchunkGymEnv

def main():
    environment = NeobotixSchunkGymEnv(renders=1, is_discrete=0, max_steps=500, action_dim=6, ws_boundary=1,
                                       random_initial=1, if_prioritized=0, if_obstacle=1, if_obstacle_moving=1, if_goal_moving_type='static')
    #environment = NeobotixGymEnv(renders=1, isDiscrete=False, maxSteps=2e3, actionDim=2, colliObj=0, wsBoundary=1, randomInitial=0)environment = NeobotixSchunkGymEnv(renders=1, isDiscrete=False, maxSteps=3e3, actionDim=10, colliObj=0, wsBoundary=1, randomInitial=1)
    # environment._p.startStateLogging(environment._p.STATE_LOGGING_VIDEO_MP4, "TEST_GUI.mp4")
    dv = 1
    actionIds = []
    dvalue = 1
    actionIds.append(environment._p.addUserDebugParameter("arm_1_joint", -dv, dv, dvalue))
    actionIds.append(environment._p.addUserDebugParameter("arm_2_joint", -dv, dv, dvalue))
    actionIds.append(environment._p.addUserDebugParameter("arm_3_joint", -dv, dv, dvalue))
    actionIds.append(environment._p.addUserDebugParameter("arm_4_joint", -dv, dv, dvalue))
    actionIds.append(environment._p.addUserDebugParameter("arm_5_joint", -dv, dv, dvalue))
    actionIds.append(environment._p.addUserDebugParameter("arm_6_joint", -dv, dv, dvalue))
    actionIds.append(environment._p.addUserDebugParameter("arm_7_joint", -dv, dv, dvalue))
    actionIds.append(environment._p.addUserDebugParameter("basevelocityx", -dv, dv, dvalue))
    actionIds.append(environment._p.addUserDebugParameter("basevelocityy", -dv, dv, dvalue))
    actionIds.append(environment._p.addUserDebugParameter("baseangularvelocity", -dv, dv, dvalue))

    done = 0
    n_steps = 500

    while not False:
        environment.reset()
        disc_total_rew=0
        t=0
        for i in range(n_steps):
            action = []
            for actionId in actionIds:
                action.append(environment._p.readUserDebugParameter(actionId))
            action = environment.action_space.sample()
            state, reward, done, info = environment.step(action)
            #print('len', i, len(state), state, info, reward)
            #state, reward, done, info = environment.step(environment._sample_action())
            #print('step', state, reward, done, info)
            #obs = environment.getExtendedObservation()
            # print(environment._p.getPhysicsEngineParameters)
            environment.render()
            disc_total_rew += reward * 0.998 ** t
            t += 1
            if done:
                break
        print(disc_total_rew, t)

if __name__=="__main__":
    main()
