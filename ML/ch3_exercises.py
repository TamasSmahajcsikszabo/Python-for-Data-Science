#### 1.

from scipy.io import loadmat
import numpy as np
mnist = loadmat("C:\\Users\\Tapi\\OneDrive\\python_code\\mnist-original.mat")
x = np.transpose(mnist["data"])
y = np.transpose(mnist["label"])
## splitting the data
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

## shuffling the training set
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

### the classifier
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()

### setting up the gridsearch
param_grid = [
    {'weights' : ['uniform','distance'],
     'n_neighbors' : list(np.arange(1,7))}
    ]

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(knn_model,
    param_grid,
    cv = 3,
    scoring = 'accuracy',
    n_jobs = -1)

grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimate_

## Accuracy measure
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score

cross_val_score(best_model, x_train, y_train, cv = 3, scring = "accuracy")

## confusion matrix
## AUC
## ROC
## Precision-Recall-Plot


## param search from the book
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(x_train, y_train)

### 2. data augmentation / training set expansion

image = x[36000]
image = image.reshape(28,28)
new = image[5]
image[0:27]

import matplotlib
import matplotlib.pyplot as plt
plt.imshow(image,
           cmap = matplotlib.cm.binary,
           interpolation = "nearest")
plt.axis("off")
plt.show()

def shifter(image, direction = "up"):
    image = image.reshape(28,28)
    
    if direction == "up":
        a = image[1:28,:]
        b = image[0]
        new_image = np.vstack((a,b))
    elif direction == "down":
        a = image[27]
        b = image[0:27,:]
        new_image = np.vstack((a,b))
    elif direction == "left":
        a = image[:,1:28]
        b = np.array(image[:,0])
        new_image = np.concatenate((a,b.reshape(-1,1)), axis = 1)
    else:
        a = np.array(image[:,27])
        b = image[:,0:27]
        new_image = np.concatenate((a.reshape(-1,1),b), axis = 1)
    return new_image


x_train_augmented = x_train

for i in list(range(1, 60000)):
    for o in ['up', 'down','left','right']:
        image = x_train[i].reshape(28,28)
        new_image = shifter(image, o).reshape(1,784)
        np.concatenate((x_train_augmented, new_image),axis = 0)


#### Solution by the author
from scipy.ndimage.interpolation import shift
def shift_image(image, dx, dy):
    image = image.reshape(28,28)
    shifted_image = shift(image, [dx, dy], cval = 0, mode="constant")
    return shifted_image.reshape([-1])

x_train_augmented = [image for iamge in x_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
    for image, label in zip(x_train, y_train):
        x_train_augmented.append(shift_image(image, dx,dy))
        y_train_augmented.append(label)