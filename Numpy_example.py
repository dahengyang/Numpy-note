# %% codeblock

# import packages
import numpy as np
import matplotlib.pyplot as plt
from tools import timeit

#


# %% codeblock
# write random walk, object oriented
class RandomWalker:
    def __init__(self):
        self.position = 0

    def walk(self,n):
        self.position = 0
        for i in range(n):
            yield self.position
            self.position += 2*np.random.randint(0,2) -1


walker = RandomWalker()
timeit("[position for position in walker.walk(1000)]",globals())


# %% codeblock
def random_walk_faster(n= 1000):
    from itertools import accumulate
    steps = 2*np.random.randint(0,2,n) -1

timeit('random_walk_faster(n=10000)',globals())


grid = np.indices((2, 3))

grid

grid_2 = np.indices((2, 3), sparse = True)

grid_2


y = np.arange(35).reshape(5,7)

y[np.array([1,2])]
y[,np.array([1,2])]

y = np.arange(30).reshape(1,1,2,3,5)

y[0,...,1,1].shape

 z = np.arange(81).reshape(3,3,3,3)

  z[[1,1,1,1]]


Z = np.arange(9).reshape(3,3).astype(np.int16)

Z

V = Z[::2,::2]

V

Z = np.zeros(9)
Z_view = Z[:3]
Z_view[...] = 1
print(Z)

Z = np.zeros(9)
Z_copy = Z[[0,1,2]]
Z_copy[...] = 1
print(Z)


Z = np.random.uniform(0,1,(5,5))
Z1 = Z[:3,:]
Z2 = Z[[0,1,2], :]
Z1.base is Z # True
Z2.base is Z # False
Z2.base is None # True
