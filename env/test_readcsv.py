import csv
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt
import numpy as np
import glob



def outvalues(path_string):
    timesteps = []
    values = []
    with open(os.path.join('/home/zheng/fromremote/results/normalr/rewardmean/combined',
                           path_string)) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # print(float(row['Step'])+1, row['Step'])
            # timesteps.append(float(row['Step']))
            values.append(float(row['Value']))
            # print(row, timesteps, values)
    return values


if __name__ == '__main__':
    #addfiles = os.listdir(os.path.join(parentdir, 'neobotix_schunk_pybullet/results'))
    addfiles = os.listdir('/home/zheng/fromremote/results/normalr/rewardmean/combined')
    print(addfiles)
    i = 0
    plotlegend = ['$r=-d$', '$r=e^{-d}$', '$r=-e^d$', '$r=-ln(d)$', '$r=1/d$', '$r=-d^2$']
    for addfile in addfiles:
        value = outvalues(addfile)
        plt.plot(value)#[:500])
        print(addfile, plotlegend[i])
        i += 1
    #plt.axis([0,1000,-12,7])
    plt.legend(plotlegend, loc='lower right', numpoints=1, fontsize=15,ncol=2 )
    plt.show()