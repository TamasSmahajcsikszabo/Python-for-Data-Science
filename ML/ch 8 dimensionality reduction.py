import numpy as np

## Singular Value Decomposition
x = np.random.rand(5000,50) + np.random.poisson(5000,50)
x_centered = x - x.mean(axis = 0)

# to get the decomposite matrices
U, s, Vt = np.linalg.svd(x_centered)

# to get the components
C1 = Vt.T[:,0]
C2 = Vt.T[:,1]

# projecting the principal components to d dim. hyperplane:
W2 = Vt.T[:, :2]
X2D = x_centered.dot(W2)

from sklearn.decomposition import PCA
# it centers the data automatically

pca = PCA(n_components = 2)
X2D = pca.fit_transform(x)

# accessing the components
C1 = pca.components_[:, 0]
C2 = pca.components_[:,1]

# to access the explained variance by the component
pca.explained_variance_ratio_

# instead of choosing the number of dimensions, 
# we can predefine % of explained variance and determine the
# number of dimensions

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

n_of_components = np.argmax(cumsum > 0.80) + 1

# or define n_components as between 0.0 and 1.0
pca = PCA(n_components = 0.8)
pca_expl = pca.fit_transform(x)

# ELBOW: where the explained variance stop growing 
import matplotlib.pyplot as plt
x = list(range(1,51))

plt.plot(x, cumsum)
plt.show()

# this tells about the intrinsic dimensionality of the data

from scipy.io import loadmat
import numpy as np
mnist = loadmat("/home/tamassmahajcsikszabo/OneDrive/python_code/mnist-original.mat")
x = np.transpose(mnist["data"])
y = np.transpose(mnist["label"])

pca = PCA(n_components = 0.95)
x_DR = pca.fit_transform(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
x = list(range(1, len(x_DR)))

spot = np.argmax(cumsum>0.95)+1

plt.plot(cumsum, color = "black")
plt.axis([-10,160,0,1])
plt.plot([0, spot],[0.95,0.95], '--', color = "grey")
plt.plot(154,0.95, "ko", alpha = 1)
plt.show()

# decompression
x_recovered = pca.inverse_transform(x_DR)

## examine one picture
digit = x_recovered[36000]
digit_image = digit.reshape(28,28)
digit = x[36000]
orig_image = digit.reshape(28,28)

import matplotlib
import matplotlib.pyplot as plt
plt.subplot(121)
plt.imshow(orig_image,
           cmap = matplotlib.cm.binary,
           interpolation = "nearest")
plt.axis("off")
plt.subplot(122)
plt.imshow(digit_image,
           cmap = matplotlib.cm.binary,
           interpolation = "nearest")
plt.axis("off")
plt.show()


# to speed up PCA, a stochastic apporach is the 
# randomized pca

randomizedPCA = PCA(svd_solver="randomized")
# it quickly makes an approximation of the first d dimension

autoPCA = PCA(svd_solver="auto")

# incremental PCA (IPCA)
# instead of fitting the whole training set into memory
# it uses mini-batches of the data
# for large datasets or online learning

from sklearn.decomposition import IncrementalPCA
IPCA = IncrementalPCA(n_components = 154)
n_batches = 100
for x_batch in np.array_split(x, n_batches):
	IPCA.partial_fit(x_batch)

x_dimreduced = IPCA.transform(x)

# applying kernel PCA
from sklearn.decomposition import KernelPCA
kPCA = KernelPCA(
	n_components = 2,
	kernel = "rbf",
	gamma = 0.04
	)

# it's unsupervised
# grid search can be used to finetune it
# when tied to a supervised model (logistic regression), then finetune with gridsearch

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

clf = Pipeline([
	('kpca', KernelPCA(n_components=2)),
	('log_reg',LogisticRegression())
	])

param_grid = [{
	'kpca__gamma':np.linspace(0.03, 0.05, 10),
	'kpca__kernel':['rbf','sigmoid']
	}]


x = np.transpose(mnist["data"])
y = np.transpose(mnist["label"])

x_train, x_test = train_test_split(x, test_size = 0.5)
y_train, y_test = train_test_split(y, test_size = 0.5)


## another approach is to select kernel and hyperparameters that minimize the reconstruction error
x = np.random.rand(5000,50) + np.random.poisson(5000,50)

#fit_inverse_transform=True allows to do reconstruction
rbf_pca = KernelPCA(n_components=2, kernel="rbf",gamma=0.0433,fit_inverse_transform=True)
x_reduced = rbf_pca.fit_transform(x)
x_preimage=rbf_pca.inverse_transform(x_reduced)

from sklearn.metrics import mean_squared_error
mean_squared_error(x, x_preimage)


param_grid = [{
	'gamma':np.linspace(0.03, 0.05, 10),
	'kernel':['rbf','sigmoid']
	}]


grid_search = GridSearchCV(rbf_pca,param_grid, cv = 3, scoring='neg_mean_squared_error')
grid_search.fit(x_preimage,x)

## Locally Linear Embedding
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# a Swiss roll sample
x,y = make_swiss_roll(n_samples=2000, noise=0.1)

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=plt.cm.Spectral)
plt.show()

from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
x_reduced = lle.fit_transform(x)

z1 = x_reduced[:,0]
z2 = x_reduced[:,1]

# the unfolded manifold
plt.scatter(z1, z2, c=z1, alpha = 1/2)


plt.show()