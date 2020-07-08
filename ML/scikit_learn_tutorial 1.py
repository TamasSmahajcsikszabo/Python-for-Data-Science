# loading a dataset
from sklearn import datasets
boston = datasets.load_boston()
iris = datasets.load_iris()
new_data = pandas.read_csv('C:\\Users\\tamas\\OneDrive\\SQL\\rental.csv')

# learning and predicting
digits = datasets.load_digits()
# estimators for ML, with methods fit() and predict()

# SVM classifier estimator:
from sklearn import svm
clf = svm.SVC(gamma = 0.001, C = 100)

## clf is first fitted to the data, to learn with the fit() method
# data splitting with slicing
clf.fit(digits.data[:-1], digits.target[:-1])

### predicting for the left-out last sample
clf.predict(digits.data[-1:])

### model persistence (saving the model)
# another model for the iris data
clf = svm.SVC(gamma = 'scale')
iris = datasets.load_iris()
X,y = iris.data, iris.target
clf.fit(X,y)

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])

## instead of pickle, joblib can be used, too
## more efficient of big data
dump(clf, 'C:\\Users\\tamas\\OneDrive\\python_code\\clf_iris.joblib')
# loading back
clf_restore = load('C:\\Users\\tamas\\OneDrive\\python_code\\clf_iris.joblib')

## type casting
import numpy as np
from sklearn import random_projection
import matplotlib
rng = np.random.RandomState(0)
X = rng.rand(10, 2000) ## float64 is default cast
X = np.array(X, dtype = 'float32')

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)

### managing targets
clf.fit(iris.data, iris.target)
list(clf.predict(iris.data[:3]))
clf.fit(iris.data, iris.target_names[iris.target])
list(clf.predict(iris.data[:3]))

### refitting and updating parameters
# after been constructed, set_params() can be used
rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

clf = svm.SVC()
clf.set_params(kernel = 'linear').fit(X,y)
clf.predict(X_test)

clf.set_params(kernel = 'rbf').fit(X,y)
clf.predict(X_test)

## Multiclass vs. multilabel fitting
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator = clf(gamma = 'scale',
                                              random_state = 0))
classif.fit(X,y).predict(X)


### reshaping data
digits = datasets.load_digits()
digits.images.shape
import matplotlib.pyplot as plt
plt.imshow(digits.images[-1], cmap = plt.cm.gray_r)

# to be usable in scikitlearn, we transform it into a feature vector of length 64
data = digits.images.reshape((digits.images.shape[0],-1))

#estimators
estimator.fit(data) # the fit method
estimator = Estimator(param1 = 1 ...) # setting the parameters
estimator.estimated_param_ # getting the estimated parameters

### nearest neighbors and the curse of dimensionality (CD)
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
np.unique(iris_y)

# using the knn classifier
# splitting the data
np.random.seed(0)
indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]

iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier() # the estimator created

knn.fit(iris_x_train, iris_y_train)
knn.predict(iris_x_test)

# CD:
## As p becomes large, the number of training points required for a
## good estimator grows exponentially.



## Linear model: from regression to sparsity
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
coeffs = regr.coef_
# the mean suared error
MSE = round(np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2),3)

# explained variance
regr.score(diabetes_X_test, diabetes_y_test)

## shrinkage
# few data points, high variance
# noise has  proportionally large impact
import numpy as np
from sklearn import linear_model
X = np.c_[.5, 1].T
y = [0.5, 1]
test = np.c_[0,2].T
regr = linear_model.LinearRegression()

import matplotlib.pyplot as plt
plt.figure()

np.random.seed(0)
for _ in range(6):
    this_X = .1 * np.random.normal(size = (2,1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s = 3)
plt.show()

# ridge regression
regr = linear_model.Ridge(alpha = .1)
plt.figure()
np.random.seed(0)
for _ in range(6):
    this_X = .1 * np.random.normal(size = (2,1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s = 3)
plt.show()

# raising alpha results in lower variance, but higher bias

## choosing alpha to minimize left out error
from __future__ import print_function
alphas = np.logspace(-4, -1, 6)

alpha_training = [regr.set_params(alpha = alpha)
                  .fit(diabetes_X_train, diabetes_y_train)
                  .score(diabetes_X_test, diabetes_y_test) for alpha in alphas]
alphas = np.array(alphas)
alpha_training = np.array(alpha_training)

## plotting explained variance vs. alphas
plt.figure()
np.random.seed(0)
plt.plot(alphas, alpha_training)
plt.show()

## sparsity - dealing with the CD
### selecting 2 important of the 10 predictors
## ridge decreases their importance, but lasso can set them to 0

# least absolute shrinkage and selection operator
# it's a sparse model
regr = linear_model.Lasso()
scores_lasso = [
regr.set_params(alpha = alpha).fit(diabetes_X_train, diabetes_y_train).
score(diabetes_X_test, diabetes_y_test) for alpha in alphas]

alphas = np.array(alphas)
alpha_training = np.array(scores_lasso)

plt.figure()
np.random.seed(0)
plt.plot(alphas, alpha_training)
plt.show()

# choosing the best alpa 
best_alpha = alphas[scores_lasso.index(max(scores_lasso))]
regr.alpha = best_alpha # feeding the best alpha back into the estimator
regr.fit(diabetes_X_train, diabetes_y_train)
regr.coef_

## for sparse data, use the LassoLars object

### MOVING ON TO CLASSIFICATION
log = linear_model.LogisticRegression(
                                      solver = 'lbfgs',
                                      C = 1e5,
                                      multi_class = 'multinomial')
log.fit(iris_x_train, iris_y_train)

### exercise
from sklearn import datasets, neighbors, linear_model
import numpy as np

digits = datasets.load_digits()

div = round(len(digits.data)*0.9)


digits_train_x = digits.data[:div]
digits_train_y = digits.target[:div]

digits_test_x = digits.data[div+1:]
digits_test_y = digits.target[div+1:]

## LINEAR MODEL
regr = linear_model.LogisticRegression(solver = 'lbfgs',
                                      C = 1e5)

regr.fit(digits_train_x, digits_train_y)
regr.predict(digits_test_x)

result = zip(digits_test_y, regr.predict(digits_test_x))
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(digits_test_y, regr.predict(digits_test_x))
plt.show()

## KNN
regr = neighbors.KNeighborsClassifier()
ks = [1,2,3,4,5,6,7,8,9]

results = [regr.set_params(n_neighbors = k).fit(digits_train_x, digits_train_y).predict(digits_test_x) for k in ks]

plt.figure()
plt.scatter(digits_test_y, results[4])
plt.show()

### INTERLUDE - PLOT CREATON
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

logreg = LogisticRegression(C = 1e5, solver = 'lbfgs', multi_class = 'multinomial')
logreg.fit(X,Y)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .05
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .05
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(X[:, 0], X[:, 1], c = Y, edgecolors = 'k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()

## SVM
from sklearn import svm
scv_estimator = svm.SVC(kernel = 'linear')
# preprocessing is needed to get a good prediction
from sklearn import preprocessing
import numpy as np
iris_x_train_PP = preprocessing.scale(iris_x_train)


linear_fit = scv_estimator.fit(iris_x_train_PP, iris_y_train)

## adjusting the kernel
# groups can spread nonlinearly in the feature space
3degreepoly_fit = scv_estimator.set_params(kernel = 'poly', degree = 3).fit(iris_x_train_PP, iris_y_train)


### exercise
from sklearn import datasets
iris = datasets.load_iris()
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


X = iris.data
Y = iris.target

# filtering for classes
X = X[Y !=0, :2]
Y = Y[Y !=0]

n_samples = len(Y)

np.random.seed(0)
order = np.random.permutation(n_samples)
X = X[order]
Y = Y[order].astype(np.float)

x_train = X[:int(.9 * n_samples)]
y_train = Y[:int(.9 * n_samples)]
x_test = X[int(.9 * n_samples):]
y_test = Y[int(.9 * n_samples):]

# linear SVM
lin_est = svm.SVC(kernel = 'linear')
linear_result = lin_est.fit(x_train, y_train).predict(x_test)

# polinomial SVM
pol_est = svm.SVC(kernel = 'poly', degree = 3)
pol_result = pol_est.fit(x_train, y_train).predict(x_test)

# radial basis function SVM
RBF_est = svm.SVC(kernel = 'rbf')
RBF_result = RBF_est.fit(x_train, y_train).predict(x_test)

for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(x_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(x_test[:, 0], x_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()