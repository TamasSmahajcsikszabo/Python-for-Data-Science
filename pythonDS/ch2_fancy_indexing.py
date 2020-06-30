import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

rand = np.random.RandomState(42)
x = rand.randint(100, size = 30)

# when doing fancy indexing, the result shape depends on the slice array, not
# the target array

ind = np.array([[2,5,8],
                [11, 5, 22]])
x[ind]

# multiple slices apply for multidimension
# rules of broadcasting apply for indices
x = np.arange(12).reshape((3,4))
ind1 = np.array([1,2,0])
ind2 = np.array([2,1,2])
x[ind1, ind2] # for rows and columns

# broadcasting
ind1 = ind1[:, np.newaxis]
x[ind1, ind2] # the index matrix is broadcast

# fancy indexing can be combined with simple indexing, slicing and masking
mask = np.array([1, 0, 1, 1], dtype = bool)

new_x = x[ind1[:, np.newaxis], mask]
np.shape(new_x)


# example for fancy indexing
mean = [0,0]
cov = [[1,2],
       [2,5]]

X = np.random.multivariate_normal(mean, cov, 100)
X.shape[0]
plt.scatter(X[:,0], X[:,1])
plt.savefig("2_dim_normal_dist.png")

indices = np.random.choice(X.shape[0], 20, replace=False)
selection = X[indices,]
selection.shape
plt.scatter(X[:,0], X[:,1], alpha = 1/3)
plt.scatter(selection[:,0], selection[:,1], s=200, alpha = 1/4)
plt.savefig("2_dim_normal_dist_selection.png")


x = np.zeros(10)
i = np.array([1,2,3,4])
x[i] += 1 # increment is just a shorthand of x[i] + 1
np.add.at(x, i, 1)

# example: create a histogram by hand
np.random.seed(42)
x = np.random.randn(100)
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)
i = np.searchsorted(bins, x)
help(np.searchsorted)

np.add.at(counts, i, 1)
plt.plot(bins, counts, linestyle='steps')
plt.savefig("histrogram.png")
np.histogram(x, bins = 20)
# to see the source code of a function in IPython:
np.histogram??
