import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# https://anilbs.me/generation-of-power-law-samples-with-inverse-transform-sampling-python-r-and-julia/
def get_times_and_steps(ta, alpha, n):
    times = ta*np.power((1-np.random.uniform(size=n)), 1/(1-alpha))
    steps = np.random.choice([-1,1], n)
    return times, steps

def get_ctrw_realization(max_time, ta, alpha, n, raw=False, clean=True):
    while True:
        wt, s = get_times_and_steps(ta, alpha, n)

        if raw:
            t,x = np.cumsum(wt), np.cumsum(s)
        else:
            t = np.cumsum(np.array([wt, [0]*n]).flatten('F'))
            x = np.cumsum(np.array([[0]*n , s]).flatten('F'))

        last_step = np.argmax(t>max_time)

        if last_step != 0 or not clean:
            break
    
    if last_step == 0:
        return np.array([0, max_time+1]), np.array([0, 0])

    return t[:last_step], x[:last_step]

@njit(parallel=True)
def temporal_msd(times, steps, max_time, T):
    return [sum(np.power(steps[t:T]-steps[0:T-t], 2))/(T-t) for t in range(1, max_time)]

        

"""
# fig 1
plt.figure()
ax = plt.subplot(1,1,1)
for _ in range(3):
    ts, xs = get_ctrw_realization(1000,1,3/2,500)
    ax.plot([0]+ts, [0]+xs, alpha=0.9)

plt.show()"""

#figure 2
"""plt.figure()
ax = plt.subplot(1,1,1)

ensMSD = np.array([0]*500)
for _ in range(1000):
    ts, xs = get_ctrw_realization(500,1,3/2,200, raw=True, clean=False)
    steps2 = np.array([0]+[xs[np.argmax(ts>i)-1] for i in range(1, 500)])
    ensMSD += np.power(steps2,2)
ensMSD = ensMSD / (1000)
ax.plot(ensMSD, alpha=0.9)

for _ in range(3):
    ts, xs = get_ctrw_realization(2*(10**6),1,3/2,1000, raw=True)
    steps2 = np.array([0]+[xs[np.argmax(ts>i)-1] for i in range(1, 2*(10**6))])
    tmsds = temporal_msd(ts, steps2, 500, 2*(10**6))
    ax.plot(tmsds, alpha=0.9)

plt.show()"""

"""plt.figure()
ax = plt.subplot(1,1,1)
k = [0]*100
bigT = 2*(10**6)
littlet = 500
for it in range(100):
    ts, xs = get_ctrw_realization(bigT,1,3/2,800, raw=True)
    steps2 = np.array([0]+[xs[np.argmax(ts>i)-1] for i in range(1, bigT)])
    tmsds = temporal_msd(ts, steps2, littlet, bigT)
    k[it] = tmsds[-1]/(2*(len(tmsds)-1))
    print(f"did: {k[it]}")
    
ax.hist(k, 20, alpha=0.9, density=True)

plt.show()"""

plt.figure()
ax = plt.subplot(1,1,1)
k = [0]*6
log_bigT = [13.6,14,14.5,15,15.5,16.2]
for kit, bigT in enumerate(log_bigT):
    k_avg = 0
    for _ in range(20):
        ts, xs = get_ctrw_realization(int(2**bigT),1,3/2,800, raw=True)
        steps2 = np.array([0]+[xs[np.argmax(ts>i)-1] for i in range(1, int(2**bigT))])
        tmsds = temporal_msd(ts, steps2, 300, int(2**bigT))
        k_avg += tmsds[-1]/(2*(len(tmsds)-1))
    k[kit] = k_avg / 20

k = np.log2(np.array(k))
log_bigT = np.array(log_bigT)
ax.scatter(log_bigT, k, alpha=0.9)
a, b = np.polyfit(log_bigT, k, 1)
ax.plot(log_bigT,a*log_bigT+b)
ax.text(15, k[0]-.25, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)

plt.show()
