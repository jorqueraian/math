import numpy as np
import matplotlib.pyplot as plt
import random
import math


def get_step_p2(n):
    for _ in range(n):
        direction = random.randint(0,3)
        if direction == 0:
            yield (0,1)
        elif direction == 1:
            yield (-1,0)
        elif direction == 2:
            yield (0,-1)
        elif direction == 3:
            yield (1,0)

def get_step_p3(n):
    for _ in range(n):
        t = random.uniform(0, 2*math.pi)
        yield (math.cos(t),math.sin(t))


def get_step_p4(n):
    for _ in range(n):
        t = random.uniform(0, 2*math.pi)
        r = np.random.exponential(1.0)
        yield (r*math.cos(t),r*math.sin(t))


steps = 300
x = np.zeros(steps+1) 
y = np.zeros(steps+1)
#for i, (dx, dy) in enumerate(get_step_p2(steps)):
#for i, (dx, dy) in enumerate(get_step_p3(steps)):
for i, (dx, dy) in enumerate(get_step_p4(steps)):
    x[i+1]=x[i]+dx
    y[i+1]=y[i]+dy
    print(f"{i+1}, {{{dx}, {dy}}}")
    
plt.figure()
ax = plt.subplot(1,1,1)
ax.plot(x, y, alpha=0.9)
ax.scatter(x[0],y[0])
ax.scatter(x[-1],y[-1])
print()
plt.show()


