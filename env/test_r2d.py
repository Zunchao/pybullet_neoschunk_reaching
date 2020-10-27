import matplotlib.pyplot as plt
import numpy as np
k=2

if __name__ == '__main__':
    x = np.linspace(0.01, k, 1000)
    x1 = np.linspace(0.01, 1/k, 1000)
    x2 = np.linspace(1/k, np.log(k), 1000)
    x3 = np.linspace(np.log(k), k, 1000)
    x4 = np.linspace(0.05, 1/np.sqrt(k), 1000)
    x5 = np.linspace(1/np.sqrt(k), np.log(k), 1000)
    y1 = -x
    y3 = -x**2
    y4 = -x**3

    y2 = -k*x2 + np.log(k) + 1
    y5 = -np.exp(x3) + (1-k)*np.log(k) + k + 1
    y6 = -np.log(x1)

    y7 = 1/x4
    y8 = -k*x5 + 2*np.sqrt(k) #+ np.sqrt(k)
    y9 = -np.exp(x3) + k - k * np.log(k) + 2*np.sqrt(k)

    dy1 = 5*x
    dy2 = -1/x
    dy3 = -np.exp(x)

    y10 = -np.exp(x4) + 3.44
    #plt.plot(x4, y7, x4, y10)


    x6 = np.linspace(0.01, 4.5, 1000)
    z1 = -x6#1/x6 - 20*x6
    z2 = -x6**2#1/x6 - x6**3
    z3 = -np.exp(x6)#1/x6 - np.exp(x6)
    z4 = -np.log(x6/(x6+1))#-np.log(x6) - 20*x6
    z5 = np.exp(-x6)#-np.log(x6) - x6**3
    z6 = 1/x6#-np.log(x6) - np.exp(x6)-x6+np.exp(-x6)
    z7 = z3+z4+z6
    plt.figure(1)
    plt.plot(x, z1, x, z2, x, z3, x, z4, x, z5, x, z6, x, z7, linewidth=2)
    plt.legend(['$r=-d$', '$r=-d^2$', '$r=-e^d$', '$r=-ln(d)$', '$r=e^{-d}$', '$r=1/d$', '$sum$'], loc='upper right', numpoints=1, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(-50,50)
    plt.grid()
    #plt.plot(x, y1, x, y2, x, y3, x, y4, x, y5, x, y6)
    #plt.plot(x2, y2, x3, y5, x1, y6, x4, y7, x5, y8, x3, y9)
    #plt.legend(['y=-log(x)', 'y=-3x+2.10', 'y=log(x)', 'y=1/x', 'y=-3x+3.46', 'y=-e^x+3.17'],loc='upper right', numpoints=1, fontsize=10)
    #plt.figure(2)
    #plt.plot(x, dy1, x, dy2, x, dy3)
    taux=[]
    tau0 = np.linspace(0,1.6,1600)
    tau = tau0[0:1000]
    taux = tau**(1/3)

    d1 = tau
    d2 = 1 - tau
    d3 = taux
    d4 = 1 - taux
    plt.figure(2)
    plt.plot(tau, d1, 'b-', tau, d2, 'r-', tau, d3, 'b-.', tau, d4, 'r-.', linewidth=2)
    plt.legend(['$r_{bg}$ with $\\tau$', '$r_{eg}$ with $1-\\tau$', '$r_{bg}$ with $\sqrt[3]{\\tau}$', '$r_{eg}$ with $1-\sqrt[3]{\\tau}$'], loc='center right', numpoints=1, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.show()