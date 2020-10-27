import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from env import fieldDirection
from matplotlib import cm
from ddpg.ddpg_noise import OrnsteinUhlenbeckProcess, OUNoise

D_WS = 1
HIGH_R = 1e3
K = 3
STEP = 0.2

def fun_circle(x, y, start, goal, obs):
    if start[0] == goal[0]:
        goal[0] += 1e-9

    a = (goal[1]-start[1])/(goal[0]-start[0])
    b = start[1] - a*start[0]
    disinit = np.linalg.norm(goal-start)
    print('disinit', start, goal, disinit, a, b)
    Npoint = int(np.ceil(disinit/STEP))
    xpath = np.zeros(Npoint)
    ypath = np.zeros(Npoint)
    cosa = np.cos(np.arctan2(goal[1]-start[1], goal[0]-start[0]))
    for i in range(Npoint):
        xpath[i] = start[0] + (i + 1 / 2) * STEP * cosa
        ypath[i] = a*xpath[i] + b

    d = np.ones(np.size(x))
    print(np.size(x))
    for i in range(np.size(x)):
        '''
        if start[0] < goal[0]:
            flag = (x[i] < start[0]+STEP*Npoint*cosa) and (x[i] > start[0])
        else:
            flag = (x[i] > start[0]-STEP*Npoint*cosa) and (x[i] < start[0])
        '''
        d[i] = -STEP*(Npoint+1)#disinit**2
        if np.abs(y[i]-x[i]*a-b) < np.sqrt(a*a+1)*STEP/2:
            for j in range(Npoint):
                if (x[i]-xpath[j])**2+(y[i]-ypath[j])**2 <= STEP**2/4:# and flag:
                    d[i] = -STEP*(Npoint-j)#-((goal[0]-xpath[j])**2+(goal[1]-ypath[j])**2)
                    break

        if (x[i]-goal[0])**2+(y[i]-goal[1])**2 <= 0.006:
            d[i] = 0
    #print('max ', x[np.argmax(d)], y[np.argmax(d)], np.amax(d))
    maxgoal = np.array([x[np.argmax(d)], y[np.argmax(d)], np.amax(d)])
    minobs = np.array([x[np.argmin(d)], y[np.argmin(d)], np.amin(d)])
    return d, maxgoal, minobs


def fun_plane(x, y, start, goal, obs):
    if start[0] == goal[0]:
        goal[0] += 1e-9

    a = (goal[1]-start[1])/(goal[0]-start[0])
    b = start[1] - a*start[0]
    disinit = np.linalg.norm(goal-start)
    print('disinit', start, goal, disinit, a, b)

    d = np.ones(np.size(x))
    print(np.size(x))
    for i in range(np.size(x)):
        d[i] = -1
        if start[0] < goal[0]:
            flag = (x[i] < goal[0]) and (x[i] > start[0])
        else:
            flag = (x[i] > goal[0]) and (x[i] < start[0])
        if np.abs(y[i]-x[i]*a-b) < np.sqrt(a*a+1)*STEP/2 and flag:
            d[i] = -np.sqrt((goal[0]-x[i])**2+(goal[1]-y[i])**2)/disinit

        if (x[i]-goal[0])**2+(y[i]-goal[1])**2 <= 0.006:
            d[i] = 0
    #print('max ', x[np.argmax(d)], y[np.argmax(d)], np.amax(d))
    maxgoal = np.array([x[np.argmax(d)], y[np.argmax(d)], np.amax(d)])
    minobs = np.array([x[np.argmin(d)], y[np.argmin(d)], np.amin(d)])
    return d, maxgoal, minobs

def fun_plane_squre(x, y, start, goal, obs):
    if start[0] == goal[0]:
        goal[0] += 1e-9

    a = (goal[1]-start[1])/(goal[0]-start[0])
    b = start[1] - a*start[0]
    disinit = np.linalg.norm(goal-start)
    print('disinit', start, goal, disinit, a, b)

    d = np.ones(np.size(x))
    print(np.size(x))
    for i in range(np.size(x)):
        d[i] = -1
        if start[0] < goal[0]:
            flag = (x[i] < goal[0]) and (x[i] > start[0])
        else:
            flag = (x[i] > goal[0]) and (x[i] < start[0])
        if np.abs(y[i]-x[i]*a-b) < np.sqrt(a*a+1)*STEP/2 and flag:
            d[i] = -((goal[0]-x[i])**2+(goal[1]-y[i])**2)/disinit**2

        if (x[i]-goal[0])**2+(y[i]-goal[1])**2 <= 0.006:
            d[i] = 0
    #print('max ', x[np.argmax(d)], y[np.argmax(d)], np.amax(d))
    maxgoal = np.array([x[np.argmax(d)], y[np.argmax(d)], np.amax(d)])
    minobs = np.array([x[np.argmin(d)], y[np.argmin(d)], np.amin(d)])
    return d, maxgoal, minobs

def fun_gaussian(x, y, start, goal, obs):
    if start[0] == goal[0]:
        goal[0] += 1e-9

    a = (goal[1]-start[1])/(goal[0]-start[0])
    b = start[1] - a*start[0]
    a1 = -1/a

    disinit = np.linalg.norm(goal-start)
    print('disinit', start, goal, disinit, a, b)

    d = np.ones(np.size(x))
    print(np.size(x))
    for i in range(np.size(x)):
        d[i] = -1
        b1 = y[i]-x[i]*a1
        xinter = (b1-b)/(a-a1)
        yinter = a*xinter + b

        discurrent = np.sqrt((xinter-goal[0])**2+(yinter-goal[1])**2)
        #discurrent = np.sqrt(discurrent)
        if discurrent<0.3:
            discurrent = 0.3
        xigma2 = discurrent**2/(2*np.pi)

        dxiyi = np.abs(y[i]-a*x[i]-b)/np.sqrt(a**2+1**2)
        d[i] = 1/discurrent*np.exp(-(dxiyi)**2/2/xigma2)-10/3
        '''
        if start[0] < goal[0]:
            flag = (x[i] < goal[0]) and (x[i] > start[0])
        else:
            flag = (x[i] > goal[0]) and (x[i] < start[0])
        if np.abs(y[i]-x[i]*a-b) < np.sqrt(a*a+1)*STEP/2 and flag:
            d[i] = -((goal[0]-x[i])**2+(goal[1]-y[i])**2)/disinit**2

        if (x[i]-goal[0])**2+(y[i]-goal[1])**2 <= 0.006:
            d[i] = 0
            '''
    #print('max ', x[np.argmax(d)], y[np.argmax(d)], np.amax(d))
    maxgoal = np.array([x[np.argmax(d)], y[np.argmax(d)], np.amax(d)])
    minobs = np.array([x[np.argmin(d)], y[np.argmin(d)], np.amin(d)])
    return d, maxgoal, minobs


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)
    x = y = np.arange(-D_WS, D_WS, 0.005)
    X, Y = np.meshgrid(x, y)
    lengh = np.size(X)
    goal = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])
    obs = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])
    start = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])
    print(start[0], goal[0], start[1], goal[1])
    obs = np.array([np.random.uniform(start[0], goal[0]), np.random.uniform(start[1], goal[1])])
    #goal = [0, 0] #test without obs
    #obs = [10, 10] #test without obs
    #zs, zmax, zmin = np.array(fun_circle(np.ravel(X), np.ravel(Y), start, goal, obs))
    #zs, zmax, zmin = np.array(fun_plane(np.ravel(X), np.ravel(Y), start, goal, obs))
    #zs, zmax, zmin = np.array(fun_plane_squre(np.ravel(X), np.ravel(Y), start, goal, obs))
    zs, zmax, zmin = np.array(fun_gaussian(np.ravel(X), np.ravel(Y), start, goal, obs))
    #zs, znew = np.array(fun_field(np.ravel(X), np.ravel(Y), goal, obs))
    Z = zs.reshape(X.shape)
    print(zs)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-D_WS, D_WS)
    ax.set_ylim(-D_WS, D_WS)
    ax.plot(np.array([goal[0]]), np.array([goal[1]]), np.array([0]), 'r.', markersize=10, label='goal')
    ax.plot(np.array([obs[0]]), np.array([obs[1]]), np.array([2]), 'b.', markersize=10, label='obstacle')
    ax.plot(np.array([start[0]]), np.array([start[1]]), np.array([0]), 'k.', markersize=10, label='start')
    #ax.plot(np.array([zmax[0]]), np.array([zmax[1]]), np.array([zmax[2]]), 'r*', markersize=10, label='maximum')
    #ax.plot(np.array([zmin[0]]), np.array([zmin[1]]), np.array([zmin[2]]), 'b*', markersize=10, label='minimum')
    ax.legend(numpoints=1)
    #ax.view_init(azim=0, elev=90)
    plt.show()