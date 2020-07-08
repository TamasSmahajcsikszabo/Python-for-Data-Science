# GETTING THE DATA
from scipy.io import loadmat
import numpy as np
mnist = loadmat("C:\\Users\\tamas\\OneDrive\\python_code\\mnist-original.mat")
x = np.transpose(mnist["data"])
y = np.transpose(mnist["label"])

# DATA SPLITTING STRATEGY:
# 60% - training
# 10% - validation and hyperparameter tuning
# 30% - testing
length = len(x)
index = np.random.permutation(np.array(range(0,len(x))))
train_index = index[:round(length*0.6)]
validation = index[round(length*0.6):round(length*0.7)]
testing = index[round(length*0.7):]

x_train = x[train_index]
x_validate = x[validation]
x_testing = x[testing]

y_train = y[train_index]
y_validate = y[validation]
y_testing = y[testing]

### validation subsets
from sklearn.model_selection import train_test_split

validate_index = np.random.permutation(np.array(range(len(x_validate))))
val_1, val_2 = train_test_split(validate_index, test_size = 0.5)

val_1_x_subset = x_validate[val_1]
val_1_y_subset = y_validate[val_1]

# CLASSIFIER
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
clf = SVC(kernel = 'rbf')
## gamma and C for gridsearch
from sklearn.model_selection import GridSearchCV
C = np.logspace(-6, -0.5, num = 5)
gamma = np.logspace(-2,0.75, num = 5)

param_grid = [
    {'C':C, 'gamma': gamma}
]

svm_grid_search = GridSearchCV(
    clf,
    param_grid,
    cv =3)

svm_grid_search.fit(val_1_x_subset,val_1_y_subset)

    