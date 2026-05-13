import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# headers = ['Time', 'iter', 'Tincr', 'relTincr', 'errConstraint']

Tmax = 2.0
maxit = 1
dt = 0.1

df = pd.read_csv(f'results_maxit{maxit}_dt{dt}/text_files/constraint_data.txt', skipinitialspace=True)#, names=headers)
headers = df.keys()
time = np.unique(df['Time'].values)

ok_iter = np.unique(df.loc[np.abs(df['Time']-Tmax)<0.9*(time[1]-time[0]), 'iter'].values)[-1]
ok_vars = {h: df.loc[df['iter'] == ok_iter, h] for h in headers}

# constraint residual w.r.t. time
fig, ax = plt.subplots()
plt.plot(ok_vars['Time'], ok_vars['errConstraint'])
ax.set_xlabel('Time')
ax.set_ylabel('errConstraint')
fig.savefig(f'heat_t1_{ok_iter}subiter_dt{dt}_errVStime.png')

# constraint residual w.r.t. subiterations
if maxit > 1:
    fig, ax = plt.subplots()
    colors = plt.cm.jet(time)
    for i in range(time.size):
        qq = ax.semilogy(df.loc[np.abs(df['Time']-time[i])<0.5*dt, 'iter'], df.loc[np.abs(df['Time']-time[i])<0.5*dt, 'errConstraint'], '+-', color=colors[i])
    aaa = df.loc[np.abs(df['Time']-Tmax)<0.9*(time[1]-time[0]), 'iter'].values
    # print(f"time={np.abs(df['Time']-Tmax)<0.9*(time[1]-time[0])}")
    print(aaa)
    slope = (np.log(df.loc[np.abs(df['Time']-time[i])<0.5*dt, 'errConstraint'].values[-2])-np.log(df.loc[np.abs(df['Time']-time[i])<0.5*dt, 'errConstraint'].values[-1]))/(aaa[-2]-aaa[-1])
    bbb = np.sqrt(df.loc[np.abs(df['Time']-time[i])<0.5*dt, 'errConstraint'].values[0]*df.loc[np.abs(df['Time']-time[i])<0.5*dt, 'errConstraint'].values[0]) \
        * np.exp(aaa*slope)/np.exp(aaa[0]*slope)
    ax.semilogy(aaa, bbb, '--')
    ax.set_title(f'slope=iter^{slope}')
    ax.set_xlabel('iter')
    ax.set_ylabel('errConstraint')
    fig.savefig(f'heat_t1_{ok_iter}subiter_dt{dt}_errVSiter.png')

# convergence w.r.t. dt
fig, ax = plt.subplots()
dt_all = [dt/(ii+1) for ii in range(4)]
# dt_all = [0.2, 0.1, 0.05]
maxit_all = [maxit] * len(dt_all)
df_all = [pd.read_csv(f'results_maxit{maxit}_dt{dt}/text_files/constraint_data.txt', skipinitialspace=True) \
          for maxit, dt in zip(maxit_all, dt_all)]
ddd = [np.unique(df_all[ii].loc[np.abs(df_all[ii]['Time']-Tmax)<0.5*dt_all[ii], 'Time'].values)[-1] for ii in range(len(dt_all))]
print(dt_all)
print(ddd)
bbb = [np.unique(df_all[ii].loc[np.abs(df_all[ii]['Time']-Tmax)<0.5*dt_all[ii], 'L2error'].values)[-1] for ii in range(len(dt_all))]
print(bbb)
ax.loglog(dt_all, bbb, '-+')
ax.loglog(dt_all, [dt*0.9*bbb[0]/dt_all[0] for dt in dt_all], '--')
ax.set_xlabel('dt')
ax.set_ylabel('L2 error')
plt.title(str(maxit_all))
fig.savefig(f'heat_t1_{ok_iter}subiter_dt{dt}_convdt.png')

plt.show()