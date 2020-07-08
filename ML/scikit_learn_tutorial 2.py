from sklearn import datasets, svm
import numpy as np

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')

## every estimator has a score method to judge quality of fit
svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])

### using k-fold cross validation to finetune models
x_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = []

for k in range(3):
     x_train = list(x_folds)
     x_test = x_train.pop(k)
     x_train = np.concatenate(x_train)
     y_train = list(y_folds)
     y_test = y_train.pop(k)
     y_train = np.concatenate(y_train)
     scores.append(svc.fit(x_train, y_train).score(x_test, y_test))
print(scores)

## repeated k-fold:
main = []
for r in range(10):
    x_folds = np.array_split(X_digits, 3)
    y_folds = np.array_split(y_digits, 3)
    scores = []
    for k in range(3):
         x_train = list(x_folds)
         x_test = x_train.pop(k)
         x_train = np.concatenate(x_train)
         y_train = list(y_folds)
         y_test = y_train.pop(k)
         y_train = np.concatenate(y_train)
         scores.append(svc.fit(x_train, y_train).score(x_test, y_test))
         main.append(scores)

len(main)
repeated_R2 = np.average(main)

### cross validation generators
## SKL has many generators for train / test indices
## they use the split method

from sklearn.model_selection import KFold, cross_val_score
X = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
k_fold = KFold(n_splits = 5)

for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | Test %s' % (train_indices, test_indices))

results = [svc.fit(X_digits[train], y_digits[train]).
            score(X_digits[test], y_digits[test]) 
            for train, test in k_fold.split(X_digits)]

## using the cross_val_score
cross_val_score(svc, X_digits, y_digits, cv = k_fold, n_jobs = -1)
# njobs = -1 uses all cores of the CPU


## EXERCISE
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets, svm

digits = datasets.load_digits()
X = digits.data
y = digits.target


svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

scores = []
scores_std = []
for C in C_s:
    svc.C = C
    this_scores = cross_val_score(svc, X, y, cv = 5, n_jobs = -1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))


import matplotlib.pyplot as plt
plt.figure(1, figsize = (4,3))
plt.clf()
plt.semilogx(C_s, scores)
plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score')
plt.xlabel('Parameter C')
plt.ylim(0, 1.1)
plt.show()

# grid search
# chooses a maximizer on CV-score

from sklearn.model_selection import GridSearchCV, cross_val_score
Cs = np.logspace(-6, -1, num = 4)

clf = GridSearchCV(
	estimator = svc,
	param_grid = dict(C = Cs),
	n_jobs = -1) 
# by default CV is 3-fold 
# but for classifiers it tuns stratified 3-fold

clf.fit(X_digits[:1000], y_digits[:1000])
clf.best_estimator_.C # best cost parameter
clf.best_score_		  # best CV-score

clf.score(X_digits[1000:], y_digits[1000:])
clf.score(X_digits[1000:], y_digits[1000:])

# nest a grid-search within a cross_val_score
# n_jobs can only be one
svc = svm.SVC(kernel='poly')
d = list(range(4))

clf = GridSearchCV(
	estimator = svc,
	param_grid = dict(C = Cs,
				      degree = d),
	CV = 5,
	n_jobs = 1) 

cross_val_score(clf, X_digits, y_digits)

clf.fit(X_digits, y_digits)
clf.best_params_
clf.score(X_digits, y_digits)

## cross-validated estimators (model name + 'CV')
from sklearn import linear_model, datasets
lasso = linear_model.LassoCV(cv = 3)
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
lasso.fit(X_diabetes, y_diabetes)

lasso.alpha_


## EXERCISE
## OPTIMIZING ALPHA FOR LASSO

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()

X_diabetes = diabetes.data[:150]
y_diabetes = diabetes.target[:150]

original_index = list(range(1,len(X_diabetes)))
index = np.random.permutation(list(range(1, len(y_diabetes))))
train_index = index[:round(len(index) * 0.9)]
test_index = [i for i in original_index if i not in list(train_index)]

train_x = X_diabetes[train_index]
train_y = y_diabetes[train_index]

test_x = X_diabetes[test_index]
test_y = y_diabetes[test_index]

alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{'alpha': alphas}]
n_folds = 5
lasso = Lasso(random_state=0)


lassoGrid = GridSearchCV(lasso,
                         tuned_parameters,
                         cv = n_folds,  
                         refit = False)

results = lassoGrid.fit(X_diabetes, y_diabetes)
scores = results.cv_results_['mean_test_score']
scores_std = results.cv_results_['std_test_score']
results.best_score_
scores_std = scores_std / np.sqrt(n_folds)

plt.figure().set_size_inches(8,6)
plt.clf()
plt.semilogx(alphas, scores)
plt.semilogx(alphas, scores + scores_std, 'g--')
plt.semilogx(alphas, scores - scores_std, 'g--')
plt.fill_between(alphas, scores + scores_std, scores - scores_std, alpha = 0.2)
plt.xlabel('Alpha parameter')
plt.ylabel('CV score  +/- standard error')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=0)
k_fold = KFold(3)

print("Answer to the bonus question:",
      "how much can you trust the selection of alpha?")
print()
print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k, (train, test) in enumerate(k_fold.split(X_diabetes, y_diabetes)):
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
print()
print("Answer: Not very much since we obtained different alphas for different")
print("subsets of the data and moreover, the scores for these alphas differ")
print("quite substantially.")
plt.show()


k_fold = KFold(3)
split = k_fold.split(X_diabetes, y_diabetes)

folds = enumerate(k_fold.split(X_diabetes, y_diabetes))
fold[1]
