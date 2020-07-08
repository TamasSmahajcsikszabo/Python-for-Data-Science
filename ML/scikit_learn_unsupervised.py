from sklearn import datasets
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

## k-means klustering
from sklearn import cluster

k_means_est = cluster.KMeans(n_clusters = 3)
k_means_est.fit(iris_x)

### number of clusters is hard to find
### unsupervised methods are sensitive to the initialization
### WE NEVER INTERPRET CLUSTERING RESULTS

import numpy as np
import scipy as sp
try:
    face = sp.face(gray = True)
except AttributeError:
    from scipy import misc
    face = misc.face(gray=True)


X = face.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters = 5, n_init = 1)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

### CLUSTERING: AGGLOMERATIVE AND DIVISIVE
### for agglomerative clusering connectivity graph
import matplotlib.pyplot as plt

from skimage.data import coins
from skimage.transform import rescale

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering


# #############################################################################
# Generate data
orig_coins = coins()

# Resize it to 20% of the original size to speed up the processing
# Applying a Gaussian filter for smoothing prior to down-scaling
# reduces aliasing artifacts.
smoothened_coins = gaussian_filter(orig_coins, sigma=2)
rescaled_coins = rescale(smoothened_coins, 0.2, mode="reflect")

X = np.reshape(rescaled_coins, (-1, 1))

# #############################################################################
# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*rescaled_coins.shape)