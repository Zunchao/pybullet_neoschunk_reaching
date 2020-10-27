import matplotlib.pyplot as plt
import numpy as np
r1 = 0
r2 = 0
r3 = 0
r = 0
returns = []
ree=[]
rbase=[]
ddee=[]
ddbase=[]
ts = []
discount = 0.995

N=300
return1=[]
return2=[]
return3=[]
return4=[]
return5=[]
successr = 0#*d0
def intd(t):
    if t<500:
        d0 = np.random.uniform(0.8, 1)
    elif t < 1000:
        d0 = np.random.uniform(1, 1.5)
    elif t < 1500:
        d0 = np.random.uniform(1.5, 2)
    else:
        d0 = np.random.uniform(2, 2.5)
    return d0

d0 = intd(N)

def penalty(t, de):
    #d0 = intd(t)
    # de = np.random.uniform(0, (1-t/1500)*2+0.1)
    dm = de + np.random.uniform(-0.2, 0.2)
    return np.abs(dm)

def dee(t):
    de = (1-t/N)*d0 #+ np.random.uniform(-0.2, 0.2)
    return de

def test_if(a):
    if a == 1:
        return a
    elif a == 2:
        return a
    else:
        return 3
    return False

def normalfun(d):
    d = -np.log(d)
    return d

def plotreturn(r1, r2, r3):
    for t in range(N):
        de = dee(t)
        dm = penalty(t, de)
        if de<0.05:
            de=0.05

        # print(de,dm,p)
        tau = de/d0
        tau = tau**2#np.cbrt(tau)#

        if np.abs(dm) < 0.3:
            rt2 = normalfun(de)
        else:
            if tau < 1:
                rt2 = (1-tau)*normalfun(de)+tau*normalfun(dm)
                rt2 = 1*rt2
            else:
                rt2 = 1*normalfun(dm)

        rt1 = (1-tau)*normalfun(de)+tau*normalfun(dm) #- t/N/2

        #print(p)
        #rt2 = (1-tau)*normalfun(de)+tau*normalfun(dm) - p - t/N/2
        #rt1 = -rt1
        #rt2 = -rt2
        rt = normalfun(de) #+ np.random.normal(0,0.1)
        dm = normalfun(dm)
        if t == N:
            r1 += successr * discount ** t
            r2 += successr * discount ** t
            r3 += successr * discount ** t
        else:
            r1 += rt1 * discount ** t
            r2 += rt2 * discount ** t
            r3 += rt * discount ** t
        r4 = r3 + (rt+successr) * discount ** (t+1)#big reward each epi end
        r5 = r3 + (rt-successr/2) * discount ** (t+1)#big panalty each epi end

        ts.append(t)
        return1.append(r1)
        return2.append(r2)

        return3.append(r3)
        return4.append(r4)
        return5.append(r5)

        returns.append(rt2)
        ree.append(rt)
        rbase.append(dm)
        ddee.append(rt-rt1)
        ddbase.append(dm-rt1)

if __name__ == '__main__':
    plotreturn(0, 0, 0)
    fig, ax = plt.subplots()
    plt.plot(ree, 'b-', rbase, 'g-', returns, 'r-', lw=2)
    plt.legend(['$r_{arm}$', '$r_{base}$', '$r_{pri}$'],loc='upper left', numpoints=1, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.tick_params(labelsize=15)
    plt.grid()

    plt.figure(2)
    plt.plot(ddee, 'b-', ddbase, 'g-', lw=2)
    plt.legend(['$r_{pri}-r_{arm}$', '$r_{pri}-r_{base}$'],loc='upper left', numpoints=1, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    #plt.figure(3)
    #plt.plot(return3, 'r', return4, 'b', return5, 'k')
    plt.figure(4)
    plt.plot(return3, 'r', return2, 'b', return1, 'k')
    #plt.plot(return1, 'bo-', return2, 'g*-')
    plt.grid()
    print(returns)
    plt.show()
    #plt.plot(returns)
    print(test_if(3))
