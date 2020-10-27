import matplotlib.pyplot as plt
import numpy as np
r1 = 0
r2 = 0
r3 = 0
r4 = 0
r5 = 0
r6 = 0
r11 = 0
r21 = 0
r31 = 0
r41 = 0
r51 = 0
r61 = 0
r = 0
returns = []
ree=[]
rbase=[]
ddee=[]
ddbase=[]
ts = []
discount = 0.99
d0 = 1
N=20000
return1=[]
return2=[]
return3=[]
return4=[]
return5=[]
return6=[]

return11=[]
return21=[]
return31=[]
return41=[]
return51=[]
return61=[]
successr = 400#*d0

def penalty(t, de):
    #d0 = intd(t)
    # de = np.random.uniform(0, (1-t/1500)*2+0.1)
    dm = de + np.random.uniform(-0.5, 0.5)
    if de/d0>1:
        p = -(1-de/d0)*de
    else:
        p = 0
    return d0, de, dm, p

def dee(t):
    de = (1-t/N)*d0 + np.random.uniform(-0.01, 0.01)
    return de

def plotreturn(r1, r2, r3, r4,r5,r6, r11, r21, r31, r41,r51,r61):
    for t in range(N):
        de = dee(t)
        d0, de, dm, p = penalty(t, de)

        # print(de,dm,p)
        de1 = (1-t/N)*np.random.uniform(0,d0) + np.random.uniform(-0.01, 0.01)#random state for return
        tau = de/d0
        tau = tau#**2
        rt1 = (1-tau)*de+tau*dm
        #print(p)
        rt2 = (1-tau)*de + dm*tau - p - t/N/2
        rt1 = -de
        rt2 = -de**2
        rt3 = -np.exp(de)
        rt4 = -np.log(de)
        rt5 = 1/de
        rt6 = np.exp(-de)

        rt11 = -de1
        rt21 = -de1**2
        rt31 = -np.exp(de1)
        rt41 = -np.log(de1)
        rt51 = 1 / de1
        rt61 = np.exp(-de1)
        dm = -dm
        if t == N:
            r1 += successr * discount ** t
            r2 += successr * discount ** t
            r3 += successr * discount ** t
            r4 += successr * discount ** t
            r5 += successr * discount ** t
            r6 += successr * discount ** t
        else:
            r1 += rt1 * discount ** t
            r2 += rt2 * discount ** t
            r3 += rt3 * discount ** t
            r4 += rt4 * discount ** t
            r5 += rt5 * discount ** t
            r6 += rt6 * discount ** t
            r11 += rt11 * discount ** t
            r21 += rt21 * discount ** t
            r31 += rt31 * discount ** t
            r41 += rt41 * discount ** t
            r51 += rt51 * discount ** t
            r61 += rt61 * discount ** t
        #r4 = r3 + (rt+successr) * discount ** (t+1)#big reward each epi end
        #r5 = r3 + (rt-successr/2) * discount ** (t+1)#big panalty each epi end

        ts.append(t)
        return1.append(r1)
        return2.append(r2)
        return3.append(r3)
        return4.append(r4)
        return5.append(r5)
        return6.append(r6)

        return11.append(r11)
        return21.append(r21)
        return31.append(r31)
        return41.append(r41)
        return51.append(r51)
        return61.append(r61)

        #returns.append(rt1)
        #ree.append(rt)
        #rbase.append(dm)
        #ddee.append(rt-rt1)
        #ddbase.append(dm-rt1)

if __name__ == '__main__':
    plotreturn(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    #plt.plot(ree, 'bo-', rbase, 'g*-', returns, 'r.-')
    #plt.legend(['ER', 'BR', 'PR'],loc='upper left', numpoints=1, fontsize=25)
    #plt.figure(2)
    #plt.plot(ddee, 'bo-', ddbase, 'g*-')
    #plt.legend(['error of PR and ER', 'error of PR and BR'],loc='upper left', numpoints=1, fontsize=25)
    plt.figure(3)
    #plt.plot(return1, 'r', return11, 'r--', return2, 'b', return21, 'b--', return3, 'k', return31, 'k--', return4, 'm', return41, 'm--', return5, 'c', return51, 'c--', return6, 'g', return61, 'g--')
    plt.plot(return1, 'r', return2, 'b', return3, 'k', return4, 'm', return5, 'c', return6, 'g')
    plt.plot(return11, 'r--', return21, 'b--', return31, 'k--', return41, 'm--', return51, 'c--', return61, 'g--')

    plt.legend(['return of -d', 'return of -d^2', 'return of -e^d', 'return of -log(d)', 'return of 1/d', 'return of e^(-d)'],loc='upper left', numpoints=1, fontsize=25)
    #plt.plot(return1, 'bo-', return2, 'g*-')
    plt.show()
    print(returns)