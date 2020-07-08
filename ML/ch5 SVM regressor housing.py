from sklearn.datasets import load_boston
boston = load_boston()

x = boston['data']
y = boston['target']

## data split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 1753)

## preprocessing and model fitting pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

clf = SVR()
## outside the pipeline

# preprocessing
scaling = StandardScaler()
x_train_preproc = scaling.fit_transform(x_train)

# param_grid search
param_grid = [
	{
	'kernel':['rbf'],
	'gamma': np.logspace(-10,5,20),
	'epsilon':np.logspace(-10,5,20)
	}]

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(
	clf,
	param_grid,
	cv = 10,
	scoring='neg_mean_squared_error'
	)

grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
print(best_model)

# training the model
best_model.fit(x_train, y_train)

# fitting the best model
x_test = scaling.fit_transform(x_test)
y_pred = best_model.predict(x_test)

# evaluation
from sklearn.metrics import mean_squared_error

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print(RMSE)

###### POLYNOMIAL

param_grid =	[{'degree':[1,2],
		'kernel' : ['poly']}]
grid_search = GridSearchCV(
	clf,
	param_grid,
	cv = 10,
	scoring='neg_mean_squared_error'
	)

grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
print(best_model)

# training the model
best_model.fit(x_train, y_train)

# fitting the best model
x_test = scaling.fit_transform(x_test)
y_pred = best_model.predict(x_test)

# evaluation
from sklearn.metrics import mean_squared_error

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print(RMSE)



import pandas as pd
y_test.describe()

## in pipeline
pipeline = ([
	('scaler', StandardScaler()),
	('grid_search', GridSearchCV(clf)),
	('svm_reg', clf.fit(x_train, y_test))
	]) 

