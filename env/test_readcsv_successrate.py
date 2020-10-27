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


def outvalues(path_string):
    timesteps = []
    values = []
    with open(os.path.join(parentdir, 'neobotix_schunk_pybullet/results/successdata.csv',
                           path_string)) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # print(float(row['Step'])+1, row['Step'])
            # timesteps.append(float(row['Step']))
            values.append(float(row['Value']))
            # print(row, timesteps, values)
    return values


if __name__ == '__main__':
    '''
    addfiles = os.listdir(os.path.join(parentdir, 'neobotix_schunk_pybullet/results'))
    print(addfiles)
    i = 0
    plotlegend = ['r = 1/dis', 'r=-dis', 'r = exp(dis)', 'r=1/dis', 'r = -10dis', 'r = 1/dis-10dis', 'r = -log(dis)']
    for addfile in addfiles:
        value = outvalues(addfile)
        plt.plot(value)#[:500])
        print(addfile, plotlegend[i])
        i += 1

    plt.legend(plotlegend, loc='lower right', numpoints=1, fontsize=25)
    plt.show()
    '''
    import pandas as pd

    # Read data from file 'filename.csv'
    # (in the same directory that your python process is based)
    # Control delimiters, rows, column names with read_csv (see later)
    #datas = pd.read_csv(os.path.join(parentdir, 'neobotix_schunk_pybullet/results/success_1dislog.csv'), delimiter=',', names=['total', 'episode', 'sr'])
    # Preview the first 5 lines of the loaded data
    #datas.head()

    #print(datas)

    '''
    with open(os.path.join(parentdir, 'neobotix_schunk_pybullet/results/successdata.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        x = list(reader)
        result = np.array(x).astype("float")
        resultsorted = sorted(result, key=lambda row: row[1])
        #print(resultsorted)
    '''


    #dataframe = pd.DataFrame(resultsorted)
    #print('df', dataframe)
    #df = pd.DataFrame({'timesteps': dataframe[:,1], 'sr': dataframe[:,2]})
    #print(df)
    #dataframe = dataframe.transpose()
    #print(dataframe)
    #dataframe.to_csv(os.path.join(parentdir, 'neobotix_schunk_pybullet/results/successdata.csv'), sep=' ', header=False, float_format='%.2f', index=False)
    #with open(os.path.join(parentdir, 'neobotix_schunk_pybullet/results/successdatanew.csv'), 'w', newline='') as csvfile:
        #spamwriter = csv.writer(csvfile, delimiter=', ')
        #spamwriter.writerow(resultsorted)
    #print(dataframe[:,2:3])
    #fmri = sns.load_dataset(datas, kws=dataframe)
    #print(fmri)
    '''
    data1 = pd.read_csv('/home/zheng/fromremote/results/success_0.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data2 = pd.read_csv('/home/zheng/fromremote/results/success_400.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data3 = pd.read_csv('/home/zheng/fromremote/results/success_4000.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data4 = pd.read_csv('/home/zheng/fromremote/results/success_1e4.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    fig, ax = plt.subplots()
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data1, color="r", ax=ax)
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data2, color="g", ax=ax)
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data3, color="b", ax=ax)
    sns.relplot(x='Episode', y='Success Rate', kind="line", data=data4, color="k", ax=ax)
    #g = sns.catplot(x="episode", y="sr", jitter=False,  data=datas)
    #plt.plot(result[:,1], result[:,2])
    sns.set(font_scale=15)
    plotlegend = ['success terminating reward 0', 'success terminating reward 4e2', 'success terminating reward 4e3', 'success terminating reward 1e4']
    ax.legend(labels=plotlegend, loc='upper left', numpoints=1, fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xlim(xmax=12500)
    ax.set_xlabel('Episode',fontsize=15)
    ax.set_ylabel('Success Rate', fontsize=15)
    #ax.set_xlabel('xlabel', fontsize=10)
    plt.show()
    '''
    # plt.savefig('/home/zheng/fromremote/results/diff_r.png')

    data1 = pd.read_csv('/home/zheng/fromremote/results/prio/1divid.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data2 = pd.read_csv('/home/zheng/fromremote/results/prio/1dividpri.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data3 = pd.read_csv('/home/zheng/fromremote/results/prio/exp-d.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data4 = pd.read_csv('/home/zheng/fromremote/results/prio/exp-dpri.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data5 = pd.read_csv('/home/zheng/fromremote/results/prio/ln.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data6 = pd.read_csv('/home/zheng/fromremote/results/prio/lnpri.csv', delimiter=',',
                        names=['total', 'Episode', 'Success Rate'])
    data7 = pd.read_csv('/home/zheng/PycharmProjects/spinningup/data/trpo_pendulum/trpo_pendulum_s0/progress.txt', sep='\s+' )
    data8 = pd.read_csv('/home/zheng/PycharmProjects/spinningup/data/trpo_pendulum_rand/trpo_pendulum_rand_s0/progress.txt', sep='\s+' )
    #print(data7)
    fig, ax = plt.subplots()
    sns.relplot(x='TotalEnvInteracts', y='AverageEpRet', kind="line", data=data7, color="r", ax=ax, lw=5)
    sns.relplot(x='TotalEnvInteracts', y='AverageEpRet', kind="line", data=data8, color="g", ax=ax, lw=5)
    #sns.relplot(x='Episode', y='Success Rate', kind="line", data=data3, color="b", ax=ax)
    #sns.relplot(x='Episode', y='Success Rate', kind="line", data=data4, color="k", ax=ax)
    #sns.relplot(x='Episode', y='Success Rate', kind="line", data=data5, color="c", ax=ax)
    #sns.relplot(x='Episode', y='Success Rate', kind="line", data=data6, color="m", ax=ax)
    # g = sns.catplot(x="episode", y="sr", jitter=False,  data=datas)
    # plt.plot(result[:,1], result[:,2])
    sns.set(font_scale=50)
    #plotlegend = ['$r=1/d$', 'prioritized $r=1/d$']
    #plotlegend = ['$r=ln(d)$', 'prioritized $r=ln(d)$']
    #plotlegend = ['$r=e^{-d}$', 'prioritized $r=e^{-d}$']
    plotlegend = ['normal', 'noisy env']
    ax.legend(labels=plotlegend, loc='upper left', numpoints=1, fontsize=15)
    ax.tick_params(labelsize=50)
    #ax.set_xlim(xmax=2000)
    ax.set_xlabel('TotalEnvInteracts', fontsize=50)
    ax.set_ylabel('AverageEpRet', fontsize=50)
    # ax.set_xlabel('xlabel', fontsize=10)
    plt.show()