from scipy.io import loadmat
import numpy as np
mnist = loadmat("C:\\Users\\Tapi\\OneDrive\\python_code\\mnist-original.mat")
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
# good for online learning, as handles one instance of training samples at a time
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
from sklearn.model_selection import cross_val_predict # it returns the predictions, not the CV scores 
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
## SGD uses 0 as the decision function by default

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
plt.plot(thresholds[index], precisions, 'g.', alpha = 1/2, label = "Precision")
plt.plot(thresholds[index], recalls, 'b.', alpha = 1/2, label = "Recall")
plt.axvline(0, linestyle='--', color='.5')
plt.xlabel('Threshold of decision function')
plt.ylabel('Precision and Recall %')
plt.suptitle('The Precision ~ Recall Trade-Off')
plt.legend()
plt.show()

### how to decide which threshold to use?
### get the decision function scores for the data
y_scores = cross_val_predict(SGD_clf, x_train, y_train_5, cv = 3,
                           method = "decision_function")
from sklearn.metrics import precision_recall_curve # this function estimates precision - recall pairs for given thresholds
precision, recall, threshold = precision_recall_curve(y_train_5, y_scores)

## let's plot these
def plot_decision_recall_vs_threshold (precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "b--", label = "Precision")
    plt.plot(threshold, recall[:-1], "g-", label = "Recall")
    plt.xlabel("Threshold")
    plt.legend(loc = "center left")
    plt.ylim([0,1])
    plt.show()

plot_decision_recall_vs_threshold(precision, recall, threshold)

### plot precision against recall directly
plt.clf()
plt.plot(recall[:-1], precision[:-1], "b-")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

## if 90% precision would be the goal:
precision_metrics = pd.DataFrame(zip(precision, recall, threshold), columns = ['Precision', 'Recall', 'Threshold'])
precision_metrics[precision_metrics.Precision in list(range(0.89, 0.91, 0.1))]

y_train_pred_90 = (y_scores > 70000)

precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

### the ROC curve with BINARY classifiers
# true positive vs false positive (1-true negative) rates, i.e. sensitivity vs. 1-specificity

from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_train_5, y_scores)

def roc_curve_plot(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label = label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

roc_curve_plot(fpr, tpr)
plt.show()
### again, there is a trade-off:
# with higher sensitivity, the classifier produces more false positives, too
# the axis is the random classifier

# measuring the AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

## whenever the overall count of true events is low, Precision vs. Recall plots should be preferred over ROC

### A RANDOM FOREST CLASSIFIER for the same problem
from sklearn.ensemble import RandomForestClassifier
# instead of decision_function(), this classifier has a predict_proba() method
# it returns an instance by class row-column table with the corresponding probabilities

frst_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(frst_clf, x_train, y_train_5, cv = 3, method = "predict_proba")

# turning probabilities into scores
pd.DataFrame(y_probas_forest)
y_scores_forest = y_probas_forest[:,1]
fpr_f, tpr_f, threshold_f = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label = "SGD")
roc_curve_plot(fpr_f, tpr_f, 'Random Forest')
plt.legend(loc = "lower right")
plt.show()

y_pred_forest = cross_val_predict(frst_clf, x_train, y_train_5, cv = 3)
roc_auc_score(y_train_5, y_scores_forest)
precision_score(y_train_5, y_pred_forest)
recall_score(y_train_5, y_pred_forest)


### MULTICLASS CLASSIFIERS
## a.k.a. multinomial classifiers

## linear classifiers and SVM are binary
## RF and naive Bayes can handle more classes

## binary classifiers can be used for each outcome, this is called one-versus-all (one versus the rest) strategy
## another strategy is one-vs-one, where a binary classifier is trained on each pair of classes, 
## resulting in n*(n-1)/2 classifiers
# which class wins the most duels

# if binary classifier is used for multiclass problems, Scikitlearn uses OVA for most of them, and for SVM it uses OVO
# example, using stochastic gradient descent
SGD_clf.fit(x_train, y_train)

SGD_clf.predict(digit.reshape(1, -1))
SGD_clf.predict([digit])
# to see the underlying decision function scores:
digit_scores = SGD_clf.decision_function([digit])

## finding the highest score
np.argmax(digit_scores) # indeed, class of 5
SGD_clf.classes_
SGD_clf.classes_[5]

### manually selecting between OVO or OVA
## OVA with OneVersusRestClassifier
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state = 42)) ## OVO with the SDG classifier

ovo_clf.fit(x_train, y_train)
ovo_clf.predict([digit])
# indeed 45 binary classifiers:
len(ovo_clf.estimators_)

# the RF in the same manner // no need for setting OVO or OVA, as RF predicts muliclass outcome 
forest_clf = RandomForestClassifier(random_state = 42)
forest_clf.fit(x_train, y_train)
forest_clf.predict([digit])
# to get the underlying classified probabilities for "digit"
forest_clf.predict_proba([digit])

## measuring performance
cross_val_score(SGD_clf, x_train, y_train, cv = 3, scoring = "accuracy")
cross_val_score(forest_clf, x_train, y_train, cv = 3, scoring = "accuracy")

## imporving performance using scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
cross_val_score(SGD_clf, x_train_scaled, y_train, cv = 5, scoring = "accuracy")

### ERROR ANALYSIS: 
## finding ways of optimising a model by analyzing the types of errors it makes

#1. analyzing the confusion matrix
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(SGD_clf, x_train_scaled, y_train, cv = 5)
cnf_mx = confusion_matrix(y_train, y_train_pred)

plt.matshow(cnf_mx, cmap = plt.cm.gray)
plt.show()
# classes are along the diagonal
# darker squares mean fewer data points

## instead of absolute count of errors, observe the error rates:
row_sums = cnf_mx.sum(axis = 1, keepdims = True)
norm_conf_mx = cnf_mx / row_sums
pd.DataFrame(cnf_mx)
pd.DataFrame(norm_conf_mx)

np.fill_diagonal(norm_conf_mx, 0) ## keeping only the errors
pd.DataFrame(norm_conf_mx)

plt.matshow(norm_conf_mx, cmap = plt.cm.viridis)
plt.show()

## 3-5, and 8, and 9 are problematic
## Improving the model:
# ~ more training data on these values
# ~ creating an algorhithm that detects closed loopes in digits -- engineering new features
# ~ preprocess the images, using Scikit-image, Pillow, OpenCV


cl_a, cl_b = 3,5
x_aa = x_train[(y_train == cl_a) & (y_train_pred == cl_a)]
x_ab = x_train[(y_train == cl_a) & (y_train_pred == cl_b)]
x_ba = x_train[(y_train == cl_b) & (y_train_pred == cl_a)]
x_bb = x_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize = (8,8))
plt.subplot(221);
plot_digits(x_aa[:25], images_per_row = 5)
plt.subplot(222);
plot_digits(x_ab[:25], images_per_row = 5)
plt.subplot(2231);
plot_digits(x_ba[:25], images_per_row = 5)
plt.subplot(224);
plot_digits(x_bb[:25], images_per_row = 5)

#### Multilabel Classification
# a classifier yields multiple class predictions
# outputs multiple binary labels

from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_multilabel)

knn_clf.predict([digit])

## averaging the f1 score assuming classes are equally important
from sklearn.model_selection import cross_val_predict
y_train_pred_knn = cross_val_predict(knn_clf, x_train, y_multilabel, cv = 3)
f1_score(y_mnultilabel, y_train_pred_knn, average = "macro")

# weighting can be achieved by using average = "weighted"

#### Multioutput Classification
# each label can be multiclass
import numpy as np
noise = np.random.randint(0, 100, (len(x_train), 784))
x_train_mod = x_train + noise

noise = np.random.randint(0, 100, (len(x_test), 784))
x_test_mod = x_test + noise
y_train_mod = x_train
y_test_mod = x_test

digit = x_train_mod[36000]
digit_image = digit.reshape(28,28)

import matplotlib
import matplotlib.pyplot as plt
plt.imshow(digit_image,
           cmap = matplotlib.cm.binary,
           interpolation = "nearest")
plt.axis("off")
plt.show()


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_mod, y_train_mod)

# cleaned image:
digit = knn_clf.predict(x_test_mod)[36000]
digit_image = digit.reshape(28,28)

import matplotlib
import matplotlib.pyplot as plt
plt.imshow(digit_image,
           cmap = matplotlib.cm.binary,
           interpolation = "nearest")
plt.axis("off")
plt.show()
