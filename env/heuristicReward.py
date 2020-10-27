import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from env import fieldDirection
from matplotlib import cm
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

D_WS = 1
HIGH_R = 1e3
K = 3
STEP = 0.2
THRESHOLD_G = 0.1
THRESHOLD_O = 0.1


class HeuristicReward:
    def __init__(self,
                 pstart,
                 pgoal,
                 pobs):
        self._start = pstart
        self._goal = pgoal
        self._obs = pobs

        if self._start[0] == self._goal[0]:
            self._goal[0] += 1e-9
        self._a = (self._goal[1] - self._start[1]) / (self._goal[0] - self._start[0])
        self._b = self._start[1] - self._a * self._start[0]
        self._dis_init = np.linalg.norm(self._goal - self._start)
        # print('self._dis_init', self._start, self._goal, self._dis_init, self._a, self._b)
        self._Np = int(np.ceil(self._dis_init / STEP))

    def fun_circle(self, p):
        xpath = np.zeros(self._Np)
        ypath = np.zeros(self._Np)
        cosa = np.cos(np.arctan2(self._goal[1] - self._start[1], self._goal[0] - self._start[0]))
        for i in range(self._Np):
            xpath[i] = self._start[0] + (i + 1 / 2) * STEP * cosa
            ypath[i] = self._a * xpath[i] + self._b

        d = -STEP * (self._Np + 1)  # self._dis_init**2
        if np.abs(p[1] - p[0] * self._a - self._b) < np.sqrt(self._a * self._a + 1) * STEP / 2:
            for j in range(self._Np):
                if (p[0] - xpath[j]) ** 2 + (p[1] - ypath[j]) ** 2 <= STEP ** 2 / 4:  # and flag:
                    d = -STEP * (self._Np - j)  # -((self._goal[0]-xpath[j])**2+(self._goal[1]-ypath[j])**2)
                    break

        d_p_obs = np.linalg.norm(self._obs - p)
        if d_p_obs <= THRESHOLD_O:
            d = -1.5
        if (p[0] - self._goal[0]) ** 2 + (p[1] - self._goal[1]) ** 2 <= THRESHOLD_G**2:
            d = 0
        return d

    def fun_plane(self, p):
        d = -1
        flag = self.region_flag(p)
        d_p_goal = np.linalg.norm(self._goal - p)
        if flag and np.abs(p[1] - p[0] * self._a - self._b) < np.sqrt(self._a ** 2 + 1) * STEP / 2:
            d = - d_p_goal / self._dis_init
        d_p_obs = np.linalg.norm(self._obs - p)
        if d_p_obs <= THRESHOLD_O:
            d = -1.5
        if d_p_goal <= THRESHOLD_G:
            d = 0
        return d

    def fun_plane_squre(self, p):
        d = -1
        flag = self.region_flag(p)
        d_p_goal = np.linalg.norm(self._goal - p)
        if flag and np.abs(p[1] - p[0] * self._a - self._b) < np.sqrt(self._a ** 2 + 1) * STEP / 2:
            d = - d_p_goal ** 2 / self._dis_init ** 2
        d_p_obs = np.linalg.norm(self._obs - p)
        if d_p_obs <= THRESHOLD_O:
            d = -1.5
        if d_p_goal <= THRESHOLD_G:
            d = 0
        return d

    def fun_gaussian(self, p):
        a1 = -1 / self._a
        b1 = p[1] - p[0] * a1
        xinter = (b1 - self._b) / (self._a - a1)
        yinter = self._a * xinter + self._b
        discurrent = np.sqrt((xinter - self._goal[0]) ** 2 + (yinter - self._goal[1]) ** 2)
        # discurrent = np.sqrt(discurrent)
        if discurrent < THRESHOLD_G * 2:
            discurrent = THRESHOLD_G * 2
        xigma2 = discurrent ** 2 / (2 * np.pi)
        dxiyi = np.abs(p[1] - self._a * p[0] - self._b) / np.sqrt(self._a ** 2 + 1)
        d = 1 / discurrent * np.exp(-(dxiyi) ** 2 / 2 / xigma2) - 1 / (THRESHOLD_G * 2)
        d_p_obs = np.linalg.norm(self._obs - p)
        if d_p_obs <= THRESHOLD_O:
            d = -1.5 - 1 / (THRESHOLD_G * 2)
        return d

    def region_flag(self, p):
        if self._start[0] < self._goal[0]:
            flag = (p[0] < self._goal[0]) and (p[0] > self._start[0])
        else:
            flag = (p[0] > self._goal[0]) and (p[0] < self._start[0])
        return flag


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
    test = HeuristicReward(start, goal, obs)
    # goal = [0, 0] #test without obs
    # obs = [10, 10] #test without obs
    # zs, zmax, zmin = np.array(fun_circle(np.ravel(X), np.ravel(Y), start, goal, obs))
    # zs, zmax, zmin = np.array(fun_plane(np.ravel(X), np.ravel(Y), start, goal, obs))
    # zs, zmax, zmin = np.array(fun_plane_squre(np.ravel(X), np.ravel(Y), start, goal, obs))

    xx = np.ravel(X)
    yy = np.ravel(Y)
    zs = np.ones(np.size(xx))
    for i in range(np.size(xx)):
        p = [xx[i], yy[i]]
        zs[i] = np.array(test.fun_gaussian(p))
        # zs, znew = np.array(fun_field(np.ravel(X), np.ravel(Y), goal, obs))
    Z = zs.reshape(X.shape)

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
    # ax.plot(np.array([zmax[0]]), np.array([zmax[1]]), np.array([zmax[2]]), 'r*', markersize=10, label='maximum')
    # ax.plot(np.array([zmin[0]]), np.array([zmin[1]]), np.array([zmin[2]]), 'b*', markersize=10, label='minimum')
    ax.legend(numpoints=1)
    # ax.view_init(azim=0, elev=90)
    plt.show()

