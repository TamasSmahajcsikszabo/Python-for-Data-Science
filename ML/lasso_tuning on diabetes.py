import numpy as np
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()

X_diabetes = diabetes.data
y_diabetes = diabetes.target

original_index = list(range(1,len(X_diabetes)))
index = np.random.permutation(list(range(1, len(y_diabetes))))
train_index = index[:round(len(index) * 0.9)]
test_index = [i for i in original_index if i not in list(train_index)]

train_x = X_diabetes[train_index]
train_y = y_diabetes[train_index]
test_x = X_diabetes[test_index]
test_y = y_diabetes[test_index]


alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = {'alpha' : alphas}
cv = 10
est = Lasso()


lasso = GridSearchCV(
    est, 
    tuned_parameters,
    cv = cv ,
    refit=False
    )

results = lasso.fit(train_x, train_y)
scores = results.cv_results_['mean_test_score']
std = results.cv_results_['std_test_score']
se = std / np.sqrt(cv)

best = round(results.best_params_['alpha'],4)

plt.figure().set_size_inches(8,6)
plt.clf()
plt.semilogx(alphas, scores, 'b-')
plt.semilogx(alphas, scores + se, 'g-')
plt.semilogx(alphas, scores - se, 'g-')
plt.fill_between(alphas, scores + se, scores - se, alpha = 0.2)
plt.axvline(best, linestyle='--', color='.7')
plt.text(np.mean(alphas)-0.008, 0.42, r'best alpha = 0.0196', color = 'blue')
plt.xlabel('Alphas')
plt.ylabel('CV-scores +/- standard error')
plt.title('Training profile of lasso with 10-fold CV')
plt.show()

results.score(test_x, test_y)

best_model = results.best_params_['alpha']

est.fit(test_x, test_y, results.best_params_)


cv = 10
est = LassoCV(cv = cv, random_state = 0, alphas = alphas)
for c in cv:
    est.fit(train_x, train_y)
    print(
        " For fold %s, alpha is %s, score is % " % (c, est.alpha_, est.score(test_x, test_y)))