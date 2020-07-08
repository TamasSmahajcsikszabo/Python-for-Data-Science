from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

### growing an optimal tree
x,y = make_moons(n_samples=10000, noise=0.4)
index = np.random.permutation(list(range(len(x))))
train_index, test_index = train_test_split(index, test_size=0.3)

x_train,y_train = x[train_index],y[train_index]
x_test,y_test = x[test_index],y[test_index]

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

clf = DecisionTreeClassifier()

parms = [
    {'min_samples_leaf':list(range(2,10,1)),
     'max_depth': list(range(2,10,1)),
     'max_leaf_nodes':list(range(2,5,1))
     }]

clf_grid_search = GridSearchCV(
    clf,
    param_grid = parms,
    cv = 5)

clf_grid_search.fit(x,y)

best_model = clf_grid_search.best_estimator_

best_model.fit(x_train,y_train)
y_pred = best_model.predict(x_test)

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

precision_score(y_pred, y_test)
recall_score(y_pred, y_test)

confusion_matrix(y_pred, y_test)
f1_score(y_pred, y_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

### growing a forest
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
from scipy.stats import mode

selector = ShuffleSplit(n_splits = 2, test_size = 100)
train_index, test_index = selector.split(x_train)

len(train_index[0])
len(train_index[1])

# average clf performance over 1000 subsets (n = 100)
accuracies = []
selector = ShuffleSplit(n_splits = 2, test_size = 100)

def average_performance(model, n, x_train, y_train, x_test, y_test):
    for i in range(1,n):
        clone_clf = clone(model)
        not_needed, train_index = selector.split(x_train)
        train_x_s, train_y_s = x_train[train_index[1]], y_train[train_index[1]]
        clone_clf.fit(train_x_s,train_y_s)
        prediction = clone_clf.predict(x_test)
        accuracy = accuracy_score(y_test, prediction)
        accuracies.append(accuracy)
    average_performance = np.mean(accuracies)
    return average_performance

average_performance(best_model, 1000, x_train, y_train, x_test, y_test)

# mode predictions

accuracies = []
predictions = {'y':y_test}
selector = ShuffleSplit(n_splits = 2, test_size = 100)
i = 1
for i in range(1,1000):
    clone_clf = clone(best_model)
    not_needed, train_index = selector.split(x_train)
    train_x_s, train_y_s = x_train[train_index[1]], y_train[train_index[1]]
    clone_clf.fit(train_x_s,train_y_s)
    prediction = clone_clf.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    accuracies.append(accuracy)
    model_name = 'tree_' + str(i)
    predictions[model_name] = prediction


main_pred = {'y' :y_test,
             'average_pred':[]}

for i in range(len(y_test)):
    predicted_value = []
    for n in predictions.keys():
        from scipy.stats import mode
        predicted_value.append(int(predictions[n][i]))
    pred = np.array(predicted_value)
    mode = int(mode(pred)[0])
    main_pred['average_pred'].append(mode)
main_pred

average_res = np.array(main_pred['average_pred'])
accuracy_score(y_test, average_res)

def random_forest(model, n, x_train, y_train, x_test, y_test):
    accuracies = []
    predictions = {'y':y_test}
    selector = ShuffleSplit(n_splits = 2, test_size = 100)
    i = 1
    for i in range(1,n):
        clone_clf = clone(model)
        not_needed, train_index = selector.split(x_train)
        train_x_s, train_y_s = x_train[train_index[1]], y_train[train_index[1]]
        clone_clf.fit(train_x_s,train_y_s)
        prediction = clone_clf.predict(x_test)
        accuracy = accuracy_score(y_test, prediction)
        accuracies.append(accuracy)
        model_name = 'tree_' + str(i)
        print('Trained model:', model_name, 'with test set accuracy:', accuracy)
        predictions[model_name] = prediction

    main_pred = {'y' :y_test,
                 'average_pred':[]}

    for i in range(len(y_test)):
        predicted_value = []
        for m in predictions.keys():
            from scipy.stats import mode
            predicted_value.append(int(predictions[m][i]))
        pred = np.array(predicted_value)
        mode = int(mode(pred)[0])
        main_pred['average_pred'].append(mode)
    main_pred

    average_res = np.array(main_pred['average_pred'])
   
    return print('Random Forest accuracy with %s trees grown is %s' % (n, round(accuracy_score(y_test, average_res),4)))

original = accuracy_score(y_test, y_pred)
train_performance_RF = average_performance(best_model, 1000, x_train, y_train, x_test, y_test)
RF = random_forest(best_model, 1000, x_train, y_train, x_test, y_test)
