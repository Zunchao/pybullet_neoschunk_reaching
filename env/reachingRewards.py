import numpy as np
import random
from env import fieldDirection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm

class ReachingReward:
    def __init__(self, with_priority, goal, armpos, basepos, opos):
        """
        Create Different Reaching Reward Functions.
        :param goal:
        :param armpos:
        :param basepos:
        :param opos:
        """
        self._with_priority = with_priority
        self._goal = goal
        self._arm_pos = armpos
        self._base_pos = basepos
        self._obstacle_pos = opos
        self._field_fun = fieldDirection.FieldDirection()
        self._field3d_fun = fieldDirection.FieldDirection3D()

    def __len__(self):
        return len(self._obstacle_pos)

    def distance(self, p1, p2):
        """
        :param p1:
        :param p2:
        :return: return dis
        """
        return np.linalg.norm(p1-p2)

    def r_normal(self, p1, p2, n):
        """
        :param p1:
        :param p2:
        :param n:
        :return: -dis^n
        """
        r = self.distance(p1, p2)
        r = -r**n
        return r

    def r_log(self, p1, p2):
        """
        :param p1:
        :param p2:
        :return: -log(dis)
        """
        dis = self.distance(p1, p2)
        if not dis:
            dis = 1e-9
        r = -np.log(dis)
        return r

    def r_exp(self, p1, p2):
        """
        :param p1:
        :param p2:
        :return: exp(-dis)
        """
        return np.exp(-self.distance(p1, p2))

    def r_exp_neg(self, p1, p2):
        """
        :param p1:
        :param p2:
        :return: -exp(dis)
        """
        return -np.exp(self.distance(p1, p2))

    def r_divid(self, p1, p2):
        """
        :param p1:
        :param p2:
        :return: 1/dis
        """
        dis = self.distance(p1, p2)
        if not dis:
            dis = 1e-9
        r = 1/dis
        return r

    def reward_normal(self, n, tau):
        """

        :param n:
        :param tau:
        :return: r = - dis^n
        """
        if not self._with_priority:
            r = self.r_normal(self._goal, self._arm_pos, n)
        else:
            r = (1-tau)*self.r_normal(self._goal, self._arm_pos, n) + tau*self.r_normal(self._goal, self._base_pos, n)
        return r

    def reward_log(self, tau):
        """

        :param tau:
        :return: r = -log(dis)
        """
        if not self._with_priority:
            r = self.r_log(self._goal, self._arm_pos)
        else:
            r = (1-tau)*self.r_log(self._goal, self._arm_pos) + tau*self.r_log(self._goal, self._base_pos)
        return r

    def reward_exp(self, tau):
        """

        :param tau:
        :return: r = exp(-dis)
        """
        if not self._with_priority:
            r = self.r_exp(self._goal, self._arm_pos)
        else:
            r = (1 - tau) * self.r_exp(self._goal, self._arm_pos) + tau * self.r_exp(self._goal, self._base_pos)
        return r

    def reward_exp_neg(self, tau):
        """

        :param tau:
        :return: r = -exp(dis)
        """
        if not self._with_priority:
            r = self.r_exp_neg(self._goal, self._arm_pos)
        else:
            r = (1 - tau) * self.r_exp_neg(self._goal, self._arm_pos) + tau * self.r_exp_neg(self._goal, self._base_pos)
        return r

    def reward_divid(self, tau):
        """

        :param tau:
        :return: r = 1/(dis)
        """
        if not self._with_priority:
            r = self.r_divid(self._goal, self._arm_pos)
        else:
            r = (1 - tau) * self.r_divid(self._goal, self._arm_pos) + tau * self.r_divid(self._goal, self._base_pos)
        return r

    def reward_field(self):
        """
        artificial potential field method
        :return: r = sum force
        """
        da_g = self.distance(self._arm_pos, self._goal)
        da_o = self.distance(self._arm_pos, self._obstacle_pos)
        db_g = self.distance(self._base_pos, self._goal)
        db_o = self.distance(self._base_pos, self._obstacle_pos)
        force, d_force = self._field_fun.compute_sum_force(self._arm_pos, self._goal, self._obstacle_pos)
        rforce, d_rforce = self._field_fun.compute_repulse(self._arm_pos, self._obstacle_pos)
        aforce, d_aforce = self._field_fun.compute_attract(self._goal, self._arm_pos)
        return -force

    def reward_field3d(self):
        """
        artificial potential field method
        :return: r = sum force
        """
        da_g = self.distance(self._arm_pos, self._goal)
        da_o = self.distance(self._arm_pos, self._obstacle_pos)
        db_g = self.distance(self._base_pos, self._goal)
        db_o = self.distance(self._base_pos, self._obstacle_pos)
        force, d_force = self._field3d_fun.compute_sum_force(self._arm_pos, self._goal, self._obstacle_pos)
        rforce, d_rforce = self._field3d_fun.compute_repulse(self._arm_pos, self._obstacle_pos)
        aforce, d_aforce = self._field3d_fun.compute_attract(self._goal, self._arm_pos)
        return -force

if __name__ == '__main__':
    # np.random.seed(11)
    D_WS = 3
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = y = np.arange(-D_WS, D_WS, 0.1)
    X, Y = np.meshgrid(x, y)
    length = np.size(X)
    px = np.ravel(X)
    py = np.ravel(Y)
    goal = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])
    obs = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])#np.ones(2)*100#

    d = np.ones(length)
    for i in range(length):
        p_robot = np.array([px[i], py[i]])
        re = ReachingReward(with_priority=0, goal=goal, armpos=p_robot, basepos=np.zeros(2), opos=obs)
        force = re.reward_field()

        d[i] = force  # + OrnsteinUhlenbeckProcess(1, n_steps_annealing=100).generate(i)[0]#np.random.normal(0,(force)/25)#OUNoise(1).noise()[0]#-np.log(force)#-force**2/10 #- 20*force**2 #- 10*force#shaped_r1(force)#1/force#np.exp(force)#d1[i] #+ d2[i]
        if d[i] < -15:
            d[i] = -15

    Z = d.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-D_WS, D_WS)
    ax.set_ylim(-D_WS, D_WS)
    ax.plot(np.array([goal[0]]), np.array([goal[1]]), np.array([0]), 'r.', markersize=10, label='goal')
    ax.plot(np.array([obs[0]]), np.array([obs[1]]), np.array([2]), 'b.', markersize=10, label='obstacle')
    ax.legend(numpoints=1)
    ax.view_init(azim=0, elev=90)
    plt.show()