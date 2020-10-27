import csv
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
FONTSIZE = 15

if __name__ == '__main__':
    import pandas as pd

    data1 = pd.read_csv('/home/zheng/fromremote/results/combinedr/dividnormal0.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data2 = pd.read_csv('/home/zheng/fromremote/results/combinedr/combineddivln-d.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data3 = pd.read_csv('/home/zheng/fromremote/results/combinedr/combinednew/div-ln-d-randomc.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    fig, ax = plt.subplots()
    #sns.relplot(x='Episode', y='Success Rate', kind="line", data=data1, color="r", ax=ax, lw=2)
    #sns.relplot(x='Episode', y='Success Rate', kind="line", data=data2, color="g", ax=ax, lw=2)
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data3, color="b", ax=ax, lw=2)

    #g = sns.catplot(x="episode", y="sr", jitter=False,  data=datas)
    #plt.plot(result[:,1], result[:,2])
    #sns.set(font_scale=2)
    #sns.set_context(font_scale=2)
    plotlegend = ['$r=c_{1}/d-c_{2}*ln(d)-c_{3}*d+w_{1}*p_{1}$']
    ax.legend(labels=plotlegend, loc='lower right', numpoints=1, fontsize=FONTSIZE)
    #ax.tick_params(labelsize=15)
    ax.set_xlim(xmax=100000)
    ax.set_xlabel('Episode', fontsize=FONTSIZE)
    ax.set_ylabel('Success Rate', fontsize=FONTSIZE)
    #ax.set_xlabel('xlabel', fontsize=10)
    #plt.xticks(fontsize=25)
    ax.tick_params(axis="x", labelsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    #ax.tick_params(labelsize=8)
    plt.show()
    # plt.savefig('/home/zheng/fromremote/results/diff_r.png')



