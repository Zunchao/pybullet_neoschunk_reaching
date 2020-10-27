import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import pandas as pd

    data1 = pd.read_csv('/home/zheng/fromremote/results/normalr/success_1dis.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data2 = pd.read_csv('/home/zheng/fromremote/results/normalr/success_dis2.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data3 = pd.read_csv('/home/zheng/fromremote/results/normalr/success_1disexp.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data4 = pd.read_csv('/home/zheng/fromremote/results/normalr/success_1dislog.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data5 = pd.read_csv('/home/zheng/fromremote/results/normalr/success_10disexp-x.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data6 = pd.read_csv('/home/zheng/fromremote/results/normalr/success_1disdivid.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    fig, ax = plt.subplots()
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data1, color="r", ax=ax, lw=5)
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data2, color="g", ax=ax, lw=5)
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data3, color="b", ax=ax, lw=5)
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data4, color="k", ax=ax, lw=5)
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data5, color="c", ax=ax, lw=5)
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data6, color="m", ax=ax, lw=5)
    #g = sns.catplot(x="episode", y="sr", jitter=False,  data=datas)
    #plt.plot(result[:,1], result[:,2])
    #sns.set(font_scale=1.5)
    plotlegend = ['$r=-d$', '$r=-d^2$', '$r=-e^d$', '$r=-ln(d)$', '$r=e^{-d}$', '$r=1/d$']
    ax.legend(labels=plotlegend, loc='upper left', numpoints=1, fontsize=50)#, ncol=3)
    ax.tick_params(labelsize=50)
    ax.set_xlim(xmax=14500)
    #ax.set_ylim(ymin=-0.1)
    ax.set_xlabel('Episode',fontsize=50)
    ax.set_ylabel('Success Rate', fontsize=50)
    ax.tick_params(labelsize=50)
    plt.show()
    # plt.savefig('/home/zheng/fromremote/results/diff_r.png')



