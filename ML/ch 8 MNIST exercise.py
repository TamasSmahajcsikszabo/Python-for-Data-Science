## 1.
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
mnist = loadmat("C:\\Users\\tamas\\OneDrive\\python_code\\mnist-original.mat")
x = np.transpose(mnist["data"])
y = np.transpose(mnist["label"])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=10000, random_state=1478)

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(random_state=1478)

import time

start = time.time()
rnd_clf.fit(x_train, y_train)
end = time.time()
execution_time = end - start
print(execution_time)



from sklearn.decomposition import PCA
rnd_clf_pca = RandomForestClassifier(random_state=1478)
pca = PCA(n_components = 0.95)
x_reduced = pca.fit_transform(x_train)
components = len(pca.components_)

start = time.time()
rnd_clf_pca.fit(x_reduced, y_train)
end = time.time()
execution_time_pca = end - start
print(execution_time_pca)

from sklearn.metrics import accuracy_score
pred = rnd_clf.predict(x_test)

pca2 = PCA(n_components = components) 
x_test_reduced = pca2.fit_transform(x_test)
pred_pca = rnd_clf_pca.predict(x_test_reduced)

original_accuracy = accuracy_score(y_test, pred)
pca_reduced_accuracy = accuracy_score(y_test, pred_pca)

## 2.

subset = x_train[:1000]
y = y_train[:1000]
from sklearn.manifold import TSNE
tsne_dr = TSNE(n_components = 2, n_iter = 250)
x_train_TSNE = tsne_dr.fit_transform(subset,y)

z1 = x_train_TSNE[:,0]
z2 = x_train_TSNE[:,1]

data = np.array(list(zip(z1,z2,y)))
c = np.array(range(1,10))
plt.scatter(data[:,0], data[:, 1], c = data[:, 2])
plt.show()

## LLE
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding()
x_train_lle = lle.fit_transform(subset)

z1 = x_train_lle[:,0]
z2 = x_train_lle[:,1]

data = np.array(list(zip(z1,z2,y)))
c = np.array(range(1,10))
plt.scatter(data[:,0], data[:, 1], c = data[:, 2])
plt.show()

## MDS
from sklearn.manifold import MDS
mds = MDS()
x_train_MDS = mds.fit_transform(subset)

z1 = x_train_MDS[:,0]
z2 = x_train_MDS[:,1]

data = np.array(list(zip(z1,z2,y)))
c = np.array(range(1,10))
plt.scatter(data[:,0], data[:, 1], c = data[:, 2])
plt.show()