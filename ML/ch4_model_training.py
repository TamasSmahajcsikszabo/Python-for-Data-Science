### closed-form training
### gradient descent training

## Linear Regression
## it estimates a weighted sum of the input feautres and a bias term (intercept)
## it equals to the dot product of feature vector and parameter vector, where theta0 parameter is always 1 (intercept)

## the closed-form solution for minimizing RMSE for the linear regression is the normal equation:
# Theta = (X^TX)^-1x^Ty

import numpy as np
import matplotlib.pyplot as plt

x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.rand(100,1)

plt.scatter(x,y)
plt.show()

## estimating the Normal Equation

x_b = np.c_[np.ones((100,1)),x]
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

## making prediction with this:
x_new = np.array([[0],[2]])
x_new_b = np.c_[np.ones((2,1)),x_new]
y_predict = x_new_b.dot(theta_best)

## plot the predictions
plt.plot(x_new, y_predict, "r-", label = "Prediction")
plt.plot(x,y,"b.")
plt.axis([0,2,0,14])
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x,y)
linear_reg.intercept_
linear_reg.coef_
linear_reg.predict(x_new)



# it's based on the following function
import scipy
theta_best_svd, residuals, rank, s = scipy.linalg.lstsq(x_b, y)

## SVD - singular value decomposition

#### the Gradient Descent Approach
# if n or P is large
# GD is a general optimization algorithm to find optimal solutions
# it iteratively tweaks the parameters to minimize a cost funtion
# it measures the local gradient of the error function with regards to the parameter vector and goes in the direction of descending gradient
# once the gradient is zero, minimum is reached

# it starts with random initialization (picks theta randomly)
# it proceeds in steps to decrease a cost function (like RMSE)
# it's done in an iterative way until the algorithm converges to a minimum

# tuning parameters:
## the size of the step which is guided by the learning rate hyperparameter
## too small steps result in elongated calculation times
## too wide steps can jump over possible valleys, not necessarily picking up minimum points in the cost function
## wide steps might lead the algorithm to diverge from the minimum, picking up higher values in loss function than before
## convergence to a minimum is also affected by different forms of cost functions
## different terrains can make distinction between *local* and *global* minimums; the random initialization can have a huge impact on the possibility of
## finding the *global* minimum
## for Linear Regression, the MSE is a convex loss function, i.e. it has one global and no local minimum
## also it never changes abruptly
## if the learning rate is not high, it will pick close to the global minimum

## it has a bowl shape, but can be elongated if the features have differenct scales
## for GD, rescaling is therefore suggested

## GD searches in the parameter space: with more parameter, the number of dimensions is also increasing and thus
## the search is also more difficult


### BATCH GRADIENT DESCENT
## GD observes changes in the loss function for ThetaJ - the partial derivative, for all dimension in turn
## the gradient vector collects all the partial derivatives for each model parameter
## it's called BATCH GD, because it uses all of the training set features at each step!
## theta minus the gradient vector multiplied by the learning rate yields the lenth of the downhill step

## example

eta = 0.1 # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1) # random initialization
for iteration in range(n_iterations):
    gradients = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - eta * gradients

# to find a good learning rate, grid search can be used
# it's good to start with a high number of iterations, but break the run when the gradient vector becomes tiny, i.e.
# compared to epsilon (tolerance)

### STOCHASTIC GRADIENT DESCENT
## BGD uses the full training set at each step 
## SGD pick a random instance at each step
# quicker, and adapts to big training sets
# it's stochastic, less regular than BGD
# while in work, the loss function jump up and down, and decreases only on average
# it bounces even when minimum reached, i.e. it yields a good minimum, but not necessarily the optimal solution
# when dealing with irregularly shaped loss functions, bouncing around helps the algorithm to jump out of the local minima,
# and continue to the global minimum - it has better chance to find the global minimum
# to remedy the randomness, a gradually decreasing learning rate makes it sure that the alogrithm finds the global minimum
# this is called SIMULATED ANNEALING
# it's guided by the function learning schedule, which determines the learning rate at each step
# learning schedule has a big impact on the model:
# if it's quick, it leads to model to end up in a local minimum valley,
# if it's slow, the model might jump out of the global minimum

n_epochs = 50 ## epoch = each round of the iteration
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
theta

from sklearn.linear_model import SGDRegressor ## by default it uses RMSE as the loss function
sgd_reg = SGDRegressor(
    n_iter = 50,
    penalty = None,         ## no regularization
    eta0 = 0.1
    )

sgd_reg.fit(x, y.ravel())
sgd_reg.intercept_
sgd_reg.coef_

### the mini-batch GD estimated the gradient vector at each step using a random set of instances (mini-batches)
## estimated closer to the minimum than SGD
## although, it escapes harder from local minimum

### POLYNOMIAL REGRESSION
np.random.seed(1984)
m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m,1)
plt.scatter(x,y)
plt.axis([0,3,0,10])
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 2, include_bias=False)
x_poly = poly_features.fit_transform(x)
x_poly[0] # it contains the original plus the 2nd degree term too
# it also contains all the possible combinations of the original features and their degrees

lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
y_pred = lin_reg.predict(x_poly)
plt.plot(x,y, 'b.')
plt.plot(x, y_pred, 'r.')
plt.axis([0,3,0,10])
plt.show()

## LEARNING CURVES
# good training data fit, but poor CV goodness-of-fit indicates overfitting
# if the model fails on both, it's underfitting
# another way of identifying under- or overfitting is to examine the learning curves
# training set performance vs. training set size (or training iteration)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curve (model, x, y):
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.2)
    train_errors, val_errors = [], []

    plt.clf()
    
    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_val_predict = model.predict(x_val)

        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
     
        plt.plot(np.sqrt(train_errors),"r-+", linewidth = 2, label = "Training Set")
        plt.plot(np.sqrt(val_errors),"b-", linewidth = 3, label = "Validation Set")

        

lin_reg = LinearRegression()
plot_learning_curve(lin_reg, x,y)
plt.show()

# the training curve starts at 0 as the model can fit the few points perfectly
# the perfectly matching training curve initially is unable to fit the validation set
# the two curves meet in a plateau
# this is a typical underfitting profile
# both errors stagnate at a high plateau
### => against underfitting, use more complex models or better features.
### additional data is not a remedy

# the same plot for a 10th degree polynomial linear regression model
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
    ('polynomial_features', PolynomialFeatures(degree = 10, include_bias=False)),
    ('lin_reg', LinearRegression())
    ])

plot_learning_curve(polynomial_regression, x,y)
plt.show()

## here the training error is lower than the validation error
## there is gap between the two => OVERFITTING
## but the two curves converge as the training size increases
### => use more data against OVERFITTING until the training error reaches the validation error

### Variance / Bias Trade-Off
## variance is related to overfitting, bias to underfitting

##### REGULARIZATION
## reducing overfitting by constraining the model
## fewer degrees of freedom reduce the chance of overfitting
## examples:
# - reducing the number of polynomial terms
# - constraining the weights of a linear regression model


## 1. RIDGE regression
# a.k.a. Tikhonov regularization
# a regularized term is added to the cost function:
# alpha * SUM(theta ^2)
# this forces the model to fit the data and keep the model weights as small as possible
# regularization is only for model trainig, model performance is evaluated using unregluraized
# measures
# alpha dictates the amount of regularization; if alpha = 0, it's simple linear regression
# if alpha is set too high, all weights converge to zero and the model is just a straight 
# line going through the mean of the data

# the bias term is NOT regularized
# scaling is needed before regularization, as it's sensitive to feature scales
## example:
# for polynomial regression:
# 1. apply polynomial features
# 2. scale all the features
# 3. apply regularization

# applying Ridge Regression using the close-form solution
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha = 1, solver = "cholesky")
ridge_reg.fit(x,y)
ridge_reg.predict(x)

# using Stochastic Gradient Descent
from sklearn.linear_model import SGDRegressor
ridge_sgd = SGDRegressor(penalty="l2") ## indicates adding 1/2 * L2 norm of weight vector 

# 2. LASSO
# Least Absolute Shrinkage and Selection Operator Regression
# L1 norm of weight vector
# it eliminates the weights of the least important features (sets them to zero)
# it automatically performs feature selection and outputs a sparse model

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha = 0.1)
lasso_reg.fit(x,y)

lasso_sgd = SGDRegressor(penalty="l1")

# 3. Elastic Net
# the regularization term is a mixture of Lasso and Ridge regularization terms
# the mixture parameter is r
# r = 0 => Ridge
# r = 1 => Lasso

## General Guideline:
# never use linear regression alone
# Ridge is a good starting point with slight regularization
# If only a handful of features are useful, use Lasso or Elastic Net
# 
# Elastic net is preferred when multicollinearity or where P > n in the training set

from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
elastic.fit(x,y)

## Early Stopping
# to regularize iterative learning algorihms like Gradient Descent is to use
# early stopping: to stop as soon as the validation error reaches a minimum

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion

class DataSelector (BaseEstimator, TransformerMixin):
    def __init__(self, attributenames):
        self.attributenames = attributenames
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attributenames]

train = ps.read_csv("C:\\Users\\tamas\\OneDrive\\python_code\\ML\\datasets\\titanic\\train.csv")

train = train[["Survived", "Age", "SibSp", "Parch", "Fare"]]
train = train[:50]

x_train = train.drop(["Survived"], axis = 1)
y_train = train["Survived"]


poly_scaler = Pipeline([
    ('selector', DataSelector(num_names)),
    ('imputer', SimpleImputer(strategy="median", missing_values = np.nan)),
    ('poly_features', PolynomialFeatures(degree = 90, include_bias=False)),
    ('std_scaler', StandardScaler())
    ])


x_train, x_val = train_test_split(train, test_size = 0.2)
y_train = x_train["Survived"]
x_train = x_train.drop(["Survived"], axis = 1)
y_val = x_val["Survived"]
x_val = x_val.drop(["Survived"], axis = 1)


x_train_poly_scaled = poly_scaler.fit_transform(x_train)
x_val_poly_scaled = poly_scaler.fit_transform(x_val)

sgd_reg = SGDRegressor(n_iter = 1,
                       warm_start = True,
                       penalty = None,
                       learning_rate = "constant",
                       eta0 = 0.0005)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(x_train_poly_scaled, y_train)
    y_val_predict = sgd_reg.predict(x_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)

### LOGISTIC REGRESSION
# a.k.a. logit regression
# it predicts the probability of an instance being class 1 (positive, 1) or class 2(negative, 0)
# just like linear regression, it estimates the weighted sum of features, but instead of
# direct output, it outputs a logistic of the result (sigma(.))
# the logistic is a sigmoid function (S-shaped) that outputs a number between 0 and 1
#sigma(t) = 1 / (1 + exp(-t))
# the estimated probability:
# p^ = hˇtheta(x) = sigma(theta^Tx)
# y^ = 0 if p^ < 0.5, 1 if p^ >= 0.5
# sigma(t) < 0.5, if t < 0 and >=0.5 if t >= 0
# so a logistic regression model predicts 0 if theta^T(x) is positive, otherwise 0


# TRAINING a Logistic Regression model means to find theta parameter vector so that the model
# estimates high probabilities for positive instances and low probabilities for negatives
# the cost function of the logistic regression:

#c(theta) = -log(p^) if y = 1, and -log(1-p^) if y = 0
# the cost is large if t approaches 0 and we would predict it to be 1
# if y is close to one, -log(t) is close to zero, so if the prediction is correct, the cost is really low

# the full training set loss funtcion is simply the average cost over all training instances
# it's the log loss (logistic regression cost function):
# J(theta) = -(1/m) * SUM[y(log(p^) + (1-y)log(1-p^)]

# this cost function is convex, so a Gradient Descent can find its global minimum
# the partial derivative is again the sum of the dot product of the prediction error multiplied by x(i)

from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())

x = iris["data"][:,3:]
y = (iris["target"] == 2).astype(np.int)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x,y)

x_new = np.linspace(0,3, 1000).reshape(-1,1)
y_proba = log_reg.predict_proba(x_new)

import matplotlib.pyplot as plt
plt.plot(x_new, y_proba[:,1], 'g-', label = "Iris-Virginica")
plt.plot(x_new, y_proba[:,0], 'b--', label = "Not Iris-Virginica")
plt.axvline()
plt.legend()
plt.show()

## regularization can be added to this model with C which is the inverse
## of alpha: the higher is C, it means less regularization

x = iris["data"][:,0:2]
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression()
log_reg.fit(x,y)

probas = log_reg.predict_proba(x)

plt.clf()
plt.scatter(x[:,0], x[:,1], c = y)
plt.scatter(probas[:,0], probas[:,1])
plt.show()


### SoftMax regression (a.k.a. Multinomial Logistic Regression)
# SM is used for accomodating the Logistic Regression for multiple class situations
# without the need of training multiple binary classifiers

# steps
# 1. it calulates a score for an instance for each of the classes sˇk(x)
# 2. using the score it estimates a probability for each class using the softmax function

## the softmax function is known as normalized exponential function
# the softmax score for instance x:
# s˘k(x) = (theta^(k))^T * X
# each class has its own dedicated parameter vector theta^(k)
# these vectors are stored as rows in the parameter matrix THETA

# the probability for these classes
# first it computes the exponetial for each score, then it normalizes them

# P^ˇ(k) = sigma(s(x))ˇk = exp(sˇk(x)) / sum(K) (exp(sˇj(x))
# K is the number of outcome classes
# s(x) it the vectotr containing the scores for each class for the instance x
# sigma(s(x))ˇk is the estimated probability that the instance x belongs to class k given the scores 
# for each class for that instance

# the prediction is made with the highest estimated probability, i.e. the class with the highest score
# when predicting it returns the value k that maximizes (argmax) the estimated probability sigma(s(x))ˇk 
# y^ = argmax( sigma(s(x))ˇk  )
# SM is multiclass, but not multioutput, it returns only one class as prediction

## training this model means to estimate high probability for the target class and low for the other classes
# minimizing its cost funtion is called cross entropy
# it penalizes the model when it estimates low probability to the target class
# CROSS ENTROPY is frequently used to measure how well a set of estimated class probabilities match the target class

# J (THETA) = -(1/m) * SUM(m) * SUM(K) yˇk(i) * log(p^(i))
# yˇk(i) is the target probability that the ith instance belongs to class k, it'w either equal to 1 or 0
# if K = 2, it's equal to log loss
# a gradient vector can be created using the usual formula:
# AVERAGE (prediction error * x(i))

# cross entropy gradient vector is estimated for each class and Gradient Descent is used to find the THETA
# parameter matrix that minimizes the cost function

# example on Iris

# for multiclass LogistiRegression uses  OVA
# to use Softmax, swith multi_class to "multinomial"
# for SM, a solver needs to be specified, too
# by default, it applay L2 regularization, which can be used by C.

x = iris["data"][:, (2,3)]
y = iris["target"]

from sklearn.linear_model import LogisticRegression
softmax_reg = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", C = 10)
softmax_reg.fit(x,y)

softmax_reg.predict([[5, 2]])
iris["target_names"][2]
softmax_reg.predict_proba([[5,2]])

###  Batch Gradient Descent with early stopping for SoftMax regression
x = iris["data"][:, (2,3)]
y = iris["target"]

# adding the bias term for every instance
x_with_bias = np.c_[np.ones([len(x), 1]), x]

np.random.seed(2042)

### splitting the data
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(x_with_bias)
test_size = int(test_ratio * total_size)
validation_size = int(validation_ratio * total_size)

train_size = total_size - test_size - validation_size
rnd_indices = np.random.permutation(total_size)
x_train = x_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
x_test = x_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
x_validation = x_with_bias[rnd_indices[train_size:-test_size]]
y_validation = y[rnd_indices[train_size:-test_size]]


# to turn class indices into target class probabilities:
def to_one_hot(y):
    n_class = y.max() + 1
    m = len(y)
    y_one_hot = np.zeros((m, n_class))
    y_one_hot[np.arange(m),y] = 1
    return y_one_hot

y_train[:10]
to_one_hot(y_train[:10])

y_train_OH = to_one_hot(y_train)
y_test_OH = to_one_hot(y_test)
y_validation_OH = to_one_hot(y_validation)

def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exp_sums

n_inputs = x_train.shape[1]
n_outputs = len(np.unique(y_train))

## translate the cost function and the gradients into Python code
eta = 0.01
n_iterations = 5001
m = len(x_train)
epsilon = 1e-7

theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = x_train.dot(theta)
    y_proba = softmax(logits)
    loss = -np.mean(np.sum(y_train_OH * np.log(y_proba + epsilon), axis = 1))
    error = y_proba - y_train_OH
    if iteration % 500 == 0:
        print(iteration, loss)
    gradients = 1/m * x_train.T.dot(error)
    theta = theta - eta * gradients

# making predictions
logits = x_validation.dot(theta)
y_proba = softmax(logits)
y_predict = np.argmax(y_proba, axis = 1)

accuracy = np.mean(y_predict == y_validation)


## adding L2 regularization to the model
eta = 0.1
n_iterations = 5001
m = len(x_train)
epsilon = 1e-7
alpha = 0.1 ## the regularization hyperparameter
theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = x_train.dot(theta)
    y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(y_train_OH * np.log(y_proba + epsilon), axis = 1))
    l2_loss = 1/2 * np.sum(np.square(theta[1:])) # we don't penalize the bias term
    loss = xentropy_loss + alpha * l2_loss
    error = y_proba - y_train_OH
    if iteration % 500 == 0:
        print(iteration, loss)
    gradients = 1/m * x_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha * theta[1:]]
    theta = theta - eta * gradients

logits = x_validation.dot(theta)
y_proba = softmax(logits)
y_predict = np.argmax(y_proba, axis = 1)

accuracy = np.mean(y_predict == y_validation)

### add early stopping
eta = 0.1
n_iterations = 5001
m = len(x_train)
epsilon = 1e-7
alpha = 0.1 ## the regularization hyperparameter
theta = np.random.randn(n_inputs, n_outputs)
best_loss = np.infty

for iteration in range(n_iterations):
    logits = x_train.dot(theta)
    y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(y_train_OH * np.log(y_proba + epsilon), axis = 1))
    l2_loss = 1/2 * np.sum(np.square(theta[1:])) # we don't penalize the bias term
    loss = xentropy_loss + alpha * l2_loss
    error = y_proba - y_train_OH
    gradients = 1/m * x_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha * theta[1:]]
    theta = theta - eta * gradients

    logits = x_validation.dot(theta)
    y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(y_validation_OH * np.log(y_proba + epsilon), axis = 1))
    l2_loss = 1/2 * np.sum(np.square(theta[1:])) # we don't penalize the bias term
    loss = xentropy_loss + alpha * l2_loss
    if iteration % 500 == 0:
        print(iteration, loss)
    if loss < best_loss:
        best_loss = loss
    else:
        print(iteration - 1, best_loss)
        print(iteration, loss, "early stopping!")
        break

logits = x_validation.dot(theta)
y_proba = softmax(logits)
y_predict = np.argmax(y_proba, axis = 1)

accuracy = np.mean(y_predict == y_validation)

### plotting the predictions on the full dataset
x0, x1 = np.meshgrid(
    np.linspace(0,8,500).reshape(-1,1),
    np.linspace(0, 3.5, 200).reshape(-1, 1))

x_new = np.c_[x0.ravel(), x1.ravel()]
x_new_with_bias = np.c_[np.ones([len(x_new), 1]), x_new]

logits = x_new_with_bias.dot(theta)
y_proba = softmax(logits)
y_predict = np.argmax(y_proba, axis = 1)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(x[y == 2, 0], x[y == 2, 1], "g^", label = "Iris-Virginica")
plt.plot(x[y == 1, 0], x[y == 1, 1], "bs", label = "Iris-Virginica")
plt.plot(x[y == 0, 0], x[y == 0, 1], "yo", label = "Iris-Virginica")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap = custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap = plt.cm.brg)
plt.clabel(contour, inline = 1, fontsize = 12)
plt.xlabel("Petal length", fontsize = 14)
plt.ylabel("Petal width", fontsize = 14)
plt.acis([0,7,0,3.5])
plt.show()

### final predictive measure
logits = x_test.dot(theta)
y_proba = softmax(logits)
y_predict = np.argmax(y_proba, axis = 1)

accuracy = np.mean(y_predict == y_test)