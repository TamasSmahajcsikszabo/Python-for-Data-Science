from scipy.io import loadmat
import numpy as np
mnist = loadmat("C:\\Users\\tamas\\OneDrive\\python_code\\mnist-original.mat")
x = np.transpose(mnist["data"])
y = np.transpose(mnist["label"])

from sklearn.model_selection import train_test_split

x_train, x_other, y_train, y_other = train_test_split(x,y, test_size = 20000)
x_test, x_val, y_test, y_val = train_test_split(x_other, y_other, test_size = 0.5)

# setting up voting and different classifiers

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

svc_clf = LinearSVC()
rnd_clf = RandomForestClassifier()
etree_clf = ExtraTreesClassifier()

voting_clf = VotingClassifier(
    estimators=[
        ('svc',svc_clf),
        ('rnd', rnd_clf),
        ('extra',etree_clf)
        ],
    voting='hard'
    )

svc_clf.fit(x_train, y_train)
rnd_clf.fit(x_train, y_train)
etree_clf.fit(x_train, y_train)
voting_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
## validation set performance
for model in (svc_clf, rnd_clf, etree_clf, voting_clf):
    pred = model.predict(x_val)
    print(model.__class__.__name__,'=>', accuracy_score(y_val, pred))


## test set
for model in (svc_clf, rnd_clf, etree_clf, voting_clf):
    pred = model.predict(x_test)
    print(model.__class__.__name__,'=>', accuracy_score(y_test, pred))


### STACKING ###
# training on the training set
for clf in (svc_clf, rnd_clf, etree_clf):
    clf.fit(x_train, y_train)

# prediction on the validation set
layer = {'y' : y_val}

for clf in (svc_clf, rnd_clf, etree_clf):
    pred = clf.predict(x_val)
    layer[clf.__class__.__name__] = pred

# creating the blender
y = layer['y'].ravel()
keys = list(layer.keys())[1:]
x = [layer[key] for key in keys]
x = np.array(x).T
from sklearn.ensemble import RandomForestClassifier

blender = RandomForestClassifier(n_estimators = 500)
blender.fit(x,y)

results = {}
for clf in (svc_clf, rnd_clf, etree_clf):
    pred = clf.predict(x_test)
    results[clf.__class__.__name__] = pred
    print(clf.__class__.__name__, '=>',accuracy_score(y_test, pred))

keys = results.keys()
x_new = [results[key] for key in keys]
x_new = np.array(x_new).T

blender_pred = blender.predict(x_new)
blender_test_performance = accuracy_score(y_test, blender_pred)
