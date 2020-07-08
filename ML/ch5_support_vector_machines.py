# SVM
# fits well small or medium-sized datasets

# Linear SVM CLF
# large margin classification: fitting the broadest set of classification lines
# the supporting vectors are fully supported by the edge variables in the subset, i.e. adding more, in-territory
# data doesn't influence them

# the supporting edge instances are called "support vectors"
# scaling highly influence support vector positions, preprocessing is needed

# hard marging classifiers:
# 1. they are only work if the data is linearly separable
# 2. outliers can influence them greatly

# therefore a more flexible model is advised
# the goal is twofold; finding balance in:
# 1. keeping the "street" as broad as possible
# 2. and limiting margin violations
# this is called soft margin classification
# in scikitlearn this is set by the C hyperparameter
# smaller C makes wider roads, but allows more violations

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

x = iris["data"][:, (2,3)]
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('svm_linear_clf', LinearSVC(C=1, loss="hinge"))
    ])

# LinearSVC regularizes the bias term, too so prior to model training, the centering is needed

svm_clf.fit(x,y)
svm_clf.predict([[5.5,1.7]])

## alternatively:
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(loss="hinge", alpha=1/(m*C)) # it's out-of-core and also it can handle online training

# when the data is not linearly separable:
# add more terms like polinomial terms
# it can make a linearly unseparable set to separable
# to implement this, add at the start of the pipeline
from sklearn.preprocessing import PolynomialFeatures
polinomial_linear_SVC = Pipeline([
    ('polynomial', PolynomialFeatures(degree=3)),
    ('standard_scaler', StandardScaler()),
    ('svm_linear_clf', LinearSVC(C=1, loss="hinge"))
      ])

x,y = make_moons(n_samples=100, noise=0.15, random_state=42)

from sklearn.datasets import make_moons
polinomial_linear_SVC.fit(x,y)

import matplotlib.pyplot as plt

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
plt.show()
axes = [-1.5, 2.5, -1, 1.5]
X = x
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = polinomial_linear_SVC.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
    

plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
plt.axis(axes)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.)
plt.grid(True, which='both')
plt.xlabel(r"$x_1$", fontsize=20)
plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
plt.show()



def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

plot_predictions(polinomial_linear_SVC, [-1.5, 2.5, -1, 1.5])
plt.show()
plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])

### Polynomial Kernel
## kernel trick: applying high degree polynomial computation without really adding these terms

from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
    ('polynomial', PolynomialFeatures(degree=3)),
    ('standard_scaler', StandardScaler()),
    ('svm_clf', SVC(kernel = "poly", degree = 3, coef0 = 1, C = 5)) # coef0 is the independent term
      ])

### Adding Similarity features
# another way of dealing with nonlinearity is to apply a similarity function:
# how much an instance resembles a particluar landmark
# for example, Gaussian Radial Basis Function:
# Thetaˇgamma * L = exp(-gamma|| x - l ||^2)
# it's bell shaped and ranges from 0 to 1 (1 is to be at the landmark)
# Xˇ2 and Xˇ3 are two landmarks for which similarity can be estimated
# then the dataset is transformed so that a linearly unseparable data can be linearly separated (by dropping the
# original features
# the downside is: originally we have m instances and n features, 
# as the result we get m instances with m features, if m is large, it results in a large number of features
# each value becomes landmark

rbf_kernel_svc = Pipeline([
    ('polynomial', PolynomialFeatures(degree=3)),
    ('standard_scaler', StandardScaler()),
    ('svm_clf', SVC(kernel = "rbf", degree = 3, gamma = 5, C = 0.001)) 
      ])

rbf_kernel_svc.fit(x,y)

# tuning:
# increasing gamma makes the bell curves narrower, each instance less influencing - the decision boundary is wiggly
# decreasing gamma broadens the bell shape, the decision boundary is smoother
# if overfitting reduce gamma, if underitting, increase gamma
# string kernel for text data

########################### SVM REGRESSION #########################################################
## it tries to balance the opposite of SVM classifier
## it tries to fit as many instances as possible on the "street"
## while limiting margin violations
# the width of the street is controlled by epsilon (margin)
# epsilon-insensitive regression: adding more training instances within the margins doesn't influence the 

from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon = 1.5)
svm_reg.fit(x,y)

# C can be used as a regulator hyperparameter, by decreasing C we apply more regularization
# nonlinearity with polynomial

from sklearn.svm import SVR
svm_poly_reg = SVR(kernel = "poly", degree = 2, epsilon=0.1, C=100)
svm_poly_reg.fit(x,y)

############## SVM in detail ####################
# notation:

# b - the bias term
# w - the feature weights

# 1. linear SVM CLF
# it predicts a new class by computing the decision function: w^TX + b 
# if it's positive, the prediction is the positive (1) class, otherwise the negative (0)

# the decision boundary is a set of points in a line where the function is 0
# -1 and 1 function points form margins around the decision boundary
# training the model means: finding w and b that make this margin as wide as possile 
# while avoiding margin violations (hard margin)
# or limiting them (soft margin)
# 
# the slope of the decision function =  ||w|| (the norm of w)
# dividing the slope by 2, multiplies the margin by 2
# => the smaller weight vector results in larger margin
# therefore to get a broad margin, we need to minimize ||w||

# for HARD MARGIN, we want to avoid margin violations
# the decision function has to be greater than 1 for positive instances, and lower than 1 for negatives
# this means a constrained optimization problem:
# t^(i)(W^Tx + b) >= 1; 
# and for HARD MARGIN we want to minimize 1/2 * W^TW

# for SOFT MARGIN
# first we need a slack function 
# Zeta(i) for each instance
# this measures how much each instance is allowed to violate margins
# there the objective is to:
# 1. making the slack function as small as possible to reduce margin violations
# 2. and making 1/2 * W.T.W as small as possible to make the margin as wide as possible
# this trade-off is handled by the C hyperparameter

# BOTH OF THESE MARGINS 
# are convex quadratic optimization problems with linear constraints
# such problems are knwon as Quadratic Programming (QP) problems

# the DUAL problem
# if given a constrained optimization problem, we call it the *primal* problem,
# we can express a different but closely related problem, its *dual problem*
# the dual problem is faster to solve when n < P, it makes the kernel trick possible

# the kernel trick
# the dot product of two transformed vectors can be replaced by the square of the dot product of the original vectors
# K(a,b) = (a.T.b)^2 is the polynomial kernel
# a kernel is a function capable of computing the dot product of transformed vectors, 
# based only on the original vectors (it doesn't need to compute the phi transformation)

## Exercises

## linear SVC; SVC and SGD on Titanic

### data loading
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

train = ps.read_csv("C:\\Users\\tamas\\OneDrive\\python_code\\ML\\datasets\\titanic\\train.csv")
test = ps.read_csv("C:\\Users\\tamas\\OneDrive\\python_code\\ML\\datasets\\titanic\\test.csv")

y_train = train["Survived"].to_numpy()
x_train = train.drop(["Survived","Name", "Ticket", "PassengerId"], axis = 1)

## custom class definitions
class VariableSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes
    def fit (self, X, y=None):
        return self
    def transfrom(self, X):
        return X[self.attributes]

# feature engineering
## label encoding for sex
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
x_train["Sex"] = LE.fit_transform(x_train["Sex"])

## label encoding for embarking
from sklearn.impute import SimpleImputer
SImp = SimpleImputer(strategy = "constant", missing_values = np.nan, fill_value = "M")
x_train["Embarked"] = SImp.fit_transform(np.array(x_train["Embarked"]).reshape(-1,1))
x_train["Embarked"] = LE.fit_transform(x_train["Embarked"])

## feature engineering on cabin
SImp = SimpleImputer(strategy = "constant", missing_values = "NaN", fill_value = 0)
x_train["Cabin"] = SImp.fit_transform(np.array(x_train["Cabin"]).reshape(-1,1))

x_train["Deck"] = x_train["Cabin"].str.get(0)
SImp = SimpleImputer(strategy = "constant", missing_values = np.nan, fill_value = "X")
x_train["Deck"] = SImp.fit_transform(np.array(x_train["Deck"]).reshape(-1,1))
x_train["Deck"] = LE.fit_transform(x_train["Deck"])


## age imputation
from sklearn.impute import SimpleImputer
SImp2 = SimpleImputer(strategy = "median")
x_train["Age"] = SImp2.fit_transform(np.array(x_train["Age"]).reshape(-1,1))

x_train = x_train.drop(["Cabin"], axis = 1)


# pipelines
from sklearn.svm import LinearSVC

linear_svc = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('linear_svc',LinearSVC(C=1, loss="hinge"))
     ])

linear_svc.fit(x_train, y_train)
result = linear_svc.predict(x_train)

from sklearn.metrics import precision_score, recall_score,precision_recall_curve
precision_score(y_train, result)
recall_score(y_train, result)

from sklearn.svm import SVC
SVC = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('linear_svc',SVC(kernel = "linear", degree = 3, coef0 = 1, C = 5))
     ])

SVC.fit(x_train, y_train)
result = SVC.predict(x_train)

from sklearn.metrics import precision_score, recall_score,precision_recall_curve
precision_score(y_train, result)
recall_score(y_train, result)


from sklearn.linear_model import SGDClassifier
SGD = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('linear_svc',SGDClassifier())
     ])

SGD.fit(x_train, y_train)
result = SGD.predict(x_train)

from sklearn.metrics import precision_score, recall_score,precision_recall_curve
precision_score(y_train, result)
recall_score(y_train, result)