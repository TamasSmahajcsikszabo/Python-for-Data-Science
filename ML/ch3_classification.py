from scipy.io import loadmat
import numpy as np
mnist = loadmat("C:\\Users\\tamas\\scikit_learn_data\\mnist-original.mat")
x = np.transpose(mnist["data"])
y = np.transpose(mnist["label"])

## examine one picture
digit = x[36000]
digit_image = digit.reshape(28,28)

import matplotlib
import matplotlib.pyplot as plt
plt.imshow(digit_image,
           cmap = matplotlib.cm.binary,
           interpolation = "nearest")
plt.axis("off")
plt.show()

## splitting the data
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

## shuffling the training set
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

## a simple binary classifier which identifies "5": 5 or not-5

# the target vectors
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# choosing the starting algorithm as SGD (Stochastic Gradient Descent)
# efficiently handles large datasets
# good for online learning, as handles one instance of training one at a time
from sklearn.linear_model import SGDClassifier
SGD_clf = SGDClassifier(random_state = 42) # it relies on randomness during training, hence the name stochastic
SGD_clf.fit(x_train, y_train_5)

SGD_clf.predict([digit])

# evaluate performance
# CV:

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits = 3, random_state = 42)

for train_index, test_index in skfolds.split(x_train, y_train_5):
    clone_clf = clone(SGD_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train_5[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train_5[test_index]

    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_folds)
    n_correct = sum(y_pred == y_test_folds)

    print(n_correct / len(y_pred))

## using the cross_val_score
from sklearn.model_selection import cross_val_score
SDG_score = cross_val_score(SGD_clf, x_train, y_train_5, cv = 3, scoring = "accuracy")

### validating results with a "non-5" tool
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y = None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype = bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, x_train, y_train_5, cv = 3, scoring = "accuracy")

##predicting just one class always produces good accuracy, 10% of the full set is 5, so guessing
##that a digit is not 5, results in 90% accuracy
## => accuracy is not a good measurement for classifiers, especially if classes are imbalanced

## Confusion Matrix
from sklearn.model_selection import cross_val_predict # it returns the predictions 
y_train_pred = cross_val_predict(SGD_clf, x_train, y_train_5, cv = 3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred) # row is actual, column is predicted

## PRECISION = TP / (TP + FP)
## RECALL / SENSITIVITY = TP /(TP + FN)
### they are in trade-off
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

## F1 score: the harmonic mean of PRECISION AND RECALL
## gives more weight to lower values, so F1 is only high if both indicators are good

# F1 = TP / (TP + (FN + FP) / 2)
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

## decision scores can be accessed and allow finetuning the decision boundary
## instead of predict(), call decision_function()
y_scores = SGD_clf.decision_function(x_train)

import pandas as pd
pd.DataFrame(y_scores).describe()
threshold = 0
y_zero_threshold_pred = (y_scores > threshold)

threshold = 1.164468e+05
y_custom_threshold_pred = (y_scores > threshold)

# raising the threshold decreases the recall
thresholds = SGD_clf.decision_function(x_train)
index = list(range(0, len(thresholds), 250))
precisions = []
recalls = []

for i in thresholds[index]:
    y_predict = (thresholds > i)
    precision_i = precision_score(y_train_5, y_predict)
    recall_i = recall_score(y_train_5, y_predict)
    precisions.append(precision_i)
    recalls.append(recall_i)

plt.clf()
plt.plot(thresholds[index], precisions, 'r.', alpha = 1/2)
plt.plot(thresholds[index], recalls, 'b.', alpha = 1/2)
plt.axvline(0, linestyle='--', color='.5')
plt.xlabel('Threshold of decision function')
plt.ylabel('Precision and Recall %')
plt.suptitle('The Precision ~ Recall Trade-Off')
plt.show()