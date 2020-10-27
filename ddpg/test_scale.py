import numpy as np
import matplotlib.pyplot as plt
r=[]
r1=[]
for i in range(1000):
    dis = 1/(i+1)-1
    step = 1/(i+1)
    delta = np.random.random_integers(-99,99)/100000
    v = [dis, -step, delta]
    r.append(np.sum(v)/np.linalg.norm(v))
    r1.append(np.sum(v))
plt.plot(r,'r', r1,'b')
plt.show()