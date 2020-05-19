# selection sort
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

x = np.random.randint(20, size=6)
i = 1
np.argmin(x[i:])
selection_sort(x)

help(np.argmin)

def bogosort(x):
    while np.any(x[:-1] > x[1:]):
        np.random.shuffle(x)
    return x

x = np.random.randint(20, size=6)
bogosort(x)

#numpy variants
x = np.random.randint(20, size=6)
np.sort(x)

#sorting by indices
x = np.random.randint(20, size=6)
indices = np.argsort(x)
x[indices]

#sorting along rows and columns
rand = np.random.RandomState(42)
X = rand.randint(0,6, (4,7))

#columnwise
np.sort(X, axis=0)

#rowwise
np.sort(X, axis=1)

# partial sorts
# returns the two arrays in arbitrary order each
X = np.array([7, 3, 4, 5, 1, 7, 8, 0])
np.partition(X, 2)
np.argpartition(X, 2)

# example: applying sorting to perform KNN
x = rand.rand(10, 2)
plt.scatter(x[:, 0], x[:, 1], s=100)
plt.savefig("knn_scatterplot.png")

dist_sqr = np.sum((x[:,np.newaxis] - x[np.newaxis,:]) ** 2, axis = -1)
#breakdown:

#estimate differences in coordinates
differences = x[:, np.newaxis] - x[np.newaxis, :]
differences.shape

#square the differences
sq_differences = differences ** 2
sq_differences.shape

#sum the differences
sum_dist = sq_differences.sum(-1)
sum_dist.shape
sum_dist.diagonal()

# to get the nearest neighbours' indices
nearest = np.argsort(sum_dist, axis=1)

k = 2
i = 1
j = nearest_partition[i, :k+1]
nearest_partition = np.argpartition(sum_dist, k+1, axis = 1)
plt.scatter(x[:,0], x[:,1], s=100)
for i in range(x.shape[0]):
    for j in nearest_partition[i, :k+1]:
        plt.plot(*zip(x[j], x[i]), color='black')
plt.savefig("knn_scatter_neighbours.png")
help(zip)

# efficient alternative for KNN especially for large datasets
import sklearn.neighbors as skn
help(skn.KDTree)
