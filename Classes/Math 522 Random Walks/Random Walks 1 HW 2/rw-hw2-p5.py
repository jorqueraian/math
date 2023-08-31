import numpy as np
import matplotlib.pyplot as plt
import random


def is_strictly_ahead(walk):
    x = 0
    for is_p in walk:
        x += (1 if is_p else -1)
        if x <= 0:
            return False
    return True


votes = 100
bignumber = 50000

vals =[0]*(votes-(votes//2 +1))

for p in range(votes//2 +1, votes):
    # p > q
    q = votes - p

    trues = 0

    for _ in range(bignumber):
        walk = [True]*p+[False]*q
        random.shuffle(walk)

        if is_strictly_ahead(walk):
            trues += 1

    vals[p-votes//2-1] = trues/bignumber
    print(p, trues/bignumber, (p-q)/(p+q))

plt.figure()
ax = plt.subplot(1,1,1)
ax.plot([i for i in range(votes//2 +1, votes)], vals, alpha=0.9)
ax.plot([i for i in range(votes//2 +1, votes)], [(2*i-100)/(100) for i in range(votes//2 +1, votes)], alpha=0.5)
plt.show()
