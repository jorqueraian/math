import numpy as np
import matplotlib.pyplot as plt
import random
import math
from numpy.random import default_rng


def get_step_gaussian(n):
    for _ in range(n):
        rng = default_rng()
        r = rng.normal()
        yield r

def get_step_cauchy(n):
    for _ in range(n):
        rng = default_rng()
        r = rng.standard_cauchy()
        yield r



steps = 200
x = np.zeros(steps+1) 
avg = np.zeros(steps+1)
secmom = np.zeros(steps+1)

for i, dx in enumerate(get_step_gaussian(steps)):
    x[i+1]=x[i]+dx
    avg[i+1] = avg[i] + x[i+1]
    avg[i] /= max(i,1)
    #avg[i+1]=x[i+1]/(i+1)
    secmom[i+1]=secmom[i]+x[i+1]**2
    secmom[i] /= max(i,1)

secmom[steps] /= (steps)
avg[steps] /= (steps)


    
plt.figure()
ax = plt.subplot(1,1,1)
ax.plot([i for i in range(steps+1)], x, alpha=0.9)
ax.plot([i for i in range(steps+1)], [0]*(steps+1), alpha=1, color="black")
plt.show()

plt.figure()
ax = plt.subplot(1,1,1)
#ax.plot([i for i in range(steps+1)], avg, alpha=0.9)
ax.plot([i for i in range(steps+1)], secmom, alpha=0.9, color="orange")
ax.plot([i for i in range(steps+1)], [0]*(steps+1), alpha=1, color="black")
#ax.plot([i for i in range(steps+1)], [i for i in range((steps+1))], alpha=1, color="black")
plt.show()

plt.figure()
ax = plt.subplot(1,1,1)
ax.plot([i for i in range(steps+1)], avg, alpha=0.9)
#ax.plot([i for i in range(steps+1)], secmom, alpha=0.9)
ax.plot([i for i in range(steps+1)], [0]*(steps+1), alpha=1, color="black")
#ax.plot([i for i in range(steps+1)], [i for i in range((steps+1))], alpha=1, color="black")
plt.show()


