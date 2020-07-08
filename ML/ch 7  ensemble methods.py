from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

x,y = make_moons(n_samples=10000, noise=0.4)
index = np.random.permutation(np.array(range(len(x))))
train_index, test_index = train_test_split(index, test_size=0.3)
x_train,y_train = x[train_index],y[train_index]
x_test,y_test = x[test_index],y[test_index]

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rnd_clf),
        ('svc',svm_clf)],
    voting = 'hard'
    )

voting_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# applying soft voting (classifiers has to have the predict_proba() method)
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rnd_clf),
        ('svc',svm_clf)],
    voting = 'soft'
    )

voting_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

## Bagging and Pasting
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier

#BaggingClassifier can bag any model
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), # it automatically soft voting, if probability estimation is allowed for the base classifier
    n_estimators = 500,
    max_samples = 100,  # can be a float between 0.0 and 1.0
    bootstrap = True)   # set it to False for Pasting!
bag_clf.fit(x_train, y_train)
ypred = bag_clf.predict(x_test)
accuracy_score(y_test,y_pred)

### OOB evaluation
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators = 500,
    max_samples = 100,  
    bootstrap = True, 
    oob_score = True)  # automatic OOB evaluation after training! 
bag_clf.fit(x_train, y_train)
OOB = bag_clf.oob_score_
bag_clf.oob_decision_function_

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
rnd_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    n_jobs = -1
    )

rnd_clf.fit(x_train,y_train)
pred = rnd_clf.predict(x_test)
accuracy_score(y_test, pred)

# bagged decision tree similar to random forest
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter = "random", max_leaf_nodes=16),
    n_estimators=500,
    max_samples=1.0,
    bootstrap=True)
bag_clf.fit(x_train, y_train)
pred = bag_clf.predict(x_test)
accuracy_score(y_test, pred)

# Extra-Trees (Extremely Randomized Trees)
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
extra_clf = ExtraTreesClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    n_jobs = -1)
extra_clf.fit(x_train, y_train)
pred = extra_clf.predict(x_test)
accuracy_score(y_test, pred)

# feature importance for RF
from sklearn.datasets import load_iris
iris = load_iris()
x = iris["data"]
y = iris["target"]

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(x,y)

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

from scipy.io import loadmat
import numpy as np
mnist = loadmat("/home/tamassmahajcsikszabo/OneDrive/python_code/mnist-original.mat")
x = np.transpose(mnist["data"])
y = np.transpose(mnist["label"])
rnd_clf = RandomForestClassifier(n_estimators=10)
rnd_clf.fit(x,y)
varimp = rnd_clf.feature_importances_
   
### BOOSTING
# adaBoost

# SKL uses SAMME (Stagwise Modeling using a Multiclass Exponential loss function)
# for two-class cases it's just adaBoost
# if the predictor model can estimate class probabilities, SKL uses SAMME.R, where
# R is for "real" - it performs better

# using AdaBoost on Decision Stumps (depth=1 trees: 1 node with 2 leaf nodes)
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    algorithm="SAMME.R",
    learning_rate=0.5
    )
ada_clf.fit(x_train,y_train)
# against overfitting:
# try reducing n_estimators
# or regularize the base estimator more

## Gradient Boosting
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
tree_reg1 = DecisionTreeRegressor(max_depth = 1)

x = np.random.randn(1000)
y = np.square(x)+np.random.rand(1000)
x = x.reshape(-1,1)
y = y.reshape(-1,1)

plt.scatter(x,y, alpha=0.2)
plt.show()

tree_reg1.fit(x,y)

# now train a new model on the residual errors made by the first one

y2 = y - tree_reg1.predict(x)
tree_reg2 = DecisionTreeRegressor(max_depth = 1)
tree_reg2.fit(x,y2)

# and again...
y3 = y2 - tree_reg2.predict(x)
tree_reg3 = DecisionTreeRegressor(max_depth=1)
tree_reg3.fit(x, y3)

# the ensemble produces predictions by adding up the predictions of all the trees

x_new =  x + np.random.randn(1000)
x_new = x_new.reshape(-1,1)
y_pred2 = sum(tree.predict(x) for tree in (tree_reg1, tree_reg2, tree_reg3))


from sklearn.metrics import accuracy_score, mean_squared_error
mean_squared_error(y, y_pred2)


## the same with a built in
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
gbrt = GradientBoostingRegressor(max_depth = 2, n_estimators = 3, learning_rate=1.0)
gbrt.fit(x,y)
y_pred = gbrt.predict(x)

# learning rate is a shrinkage parameter
# lower values will make sure each tree contribute less, so more trees are needed
# to utilize the full training set
# lower LR: needs enough trees to be trained properly, but too many estimators will 
# lead to overfitting
# early stopping can be used to find the optimal number of trees
# a simple way of doing this:
staged_predict() # method

# 1. train n number of trees
# 2. measure the validation error at each one
# 3. based upon this return the optimal number of trees
# 4. refit that number of trees

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
x_train, x_val, y_train, y_val = train_test_split(x,y)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(x_train, y_train)

errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(x_val)]
best_n_estimators = np.argmin(errors)
min_error = min(errors)
bgrt_best = GradientBoostingRegressor(max_depth = 2, n_estimators = best_n_estimators)
bgrt_best.fit(x_train, y_train)

custom_text = str('Minimum at '+ str(best_n_estimators) + ' estimators: ' + str(round(min_error,2)))

plt.plot(errors, color = 'orange', linewidth = 4)
plt.axis([-5,120,0,2.5])
plt.plot([best_n_estimators,best_n_estimators],[0,min_error], linestyle = '--', color = 'black', alpha = 1/4)
plt.plot([0, 120], [min_error,min_error],linestyle = '--', color = 'black', alpha = 1/4)
plt.plot(best_n_estimators, min_error, "ko", alpha = 1)
plt.text(best_n_estimators-50, min_error*1.25, s = custom_text)
plt.title('Optimal Number of Estimators')
plt.xlabel('Number of Estimators [0-120]')
plt.ylabel('Estimated validation error')
plt.show()

# implementing early stopping
# allowing incremential model training with the warm_start = True parameter
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth = 1, warm_start = True)
error_going_up = 0
min_val_error = float('inf')
for n_estimators in range(1,120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(x_train, y_train)
    y_pred = gbrt.predict(x_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break

# Stochastic Gradient Boosting can be achieved by setting
# the subsample parameter
# to set other loss function the loss parameter can be tweaked

