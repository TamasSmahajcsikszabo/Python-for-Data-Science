import os
import tarfile
from six.moves import urllib
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

fetch_housing_data()

## load in the data
import pandas as pd
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path) ## returns Panda DataFrame object

data = load_housing_data()
housing = data
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()

housing.hist(bins = 50, figsize = (20,15))
plt.show()

# stratified sampling
import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

housing["income_cat"].hist()
plt.show()
import sklearn.model_selection 

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)
for train_index, test_index in split.split(housing):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_train_set["income_cat"].value_counts() / len(strat_train_set)
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)

 ## exploring
housing = strat_train_set.copy()
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)


housing.plot(kind = "scatter", 
             x = "longitude", 
             y = "latitude", 
             alpha = 0.4,
            s = housing["population"] / 100,
            label = "population",
            figsize = (10, 7),
            c = "median_house_value",
            cmap = plt.get_cmap("jet"),
            colorbar = True)
plt.legend()
plt.show()

## finding patterns, clusters
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)

# Pandas' scatter_matrix
from pandas.plotting import scatter_matrix
attributes = ["median_house_value","median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize = (12, 8))
housing.plot(kind = "scatter",
            x = "median_income",
            y = "median_house_value",
            alpha = 0.3)
plt.show()

### data cleaning
housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()


## dealing with missing values
## from the DataFrame

#housing.dropna(subset="total_bedrooms")
#housing.drop('total_bedrooms')
#median = housing["total_bedrooms".median()]
#housing["total_bedrooms"].fillna(median, inplace = TRUE)


## SKLEARN provides:
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
housing_num = housing.drop("ocean_proximity", axis = 1)
imputer.fit(housing_num)

imputer.statistics_
housing_num.median().values

x = imputer.transform(housing_num)
# applying the imputation, resulting in np array

# put it back into dataframe
housing_tr = pd.DataFrame(x, columns=housing_num.columns)

# fit_tranmsform() method does these steps in one

housing_tr.head()

### convert the ocean proximity to numbers
## using Pandas' factorize()

housing_cat = housing["ocean_proximity"]
housing_cat_encoded, housing_categories = housing_cat.factorize()

## custom transformers
from sklearn.base import BaseEstimator, TransformerMixin

# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


### feature scaling
# minmax scaling (x-min) / (max - min)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
## sensitive to outliers

# standardization ## NN expects inputs between 0 and 1
from sklearn.preprocessing import StandardScaler

## SKLEARN offers pipelines for data preprocessing
from sklearn.pipeline import Pipeline
num_pipeline = Pipeline(
[
    ('imputer', SimpleImputer(strategy = "median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit (self, x, y = None):
        return self
    def transform(self, x):
        return x[self.attribute_names].values
housing_num = housing.drop("ocean_proximity", axis = 1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline(
[
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy = "median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder())
])

## joining the pipelines
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion (transformer_list = [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

housing_prepared = full_pipeline.fit_transform(housing)


### TRAINING THE MODEL
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse ##x we are underfitting the data

# either the features are not sufficient
# or the model is not powerful enough

## fitting a better model - regressiontree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


## using cross-validation
from sklearn.model_selection import cross_val_score
CV_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                           scoring = "neg_mean_squared_error", cv = 10)
tree_rmse_scores = np.sqrt(-CV_scores)  ###  utility function, not a loss function
tree_rmse_scores

def display_scores(scores):
    print("scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
display_scores(tree_rmse_scores)

## cross-validating the linear model
lin_scores = cross_val_score(lin_reg, housing_prepared,housing_labels,
        scoring = "neg_mean_squared_error",
        cv = 10)

lin_rmse = np.sqrt(-lin_scores)
display_scores(lin_rmse)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()


forest_scores = cross_val_score(forest_reg,
                               housing_prepared,
                               housing_labels,
                               scoring = "neg_mean_squared_error",
                               cv = 10)



forest_rmse = np.sqrt(-forest_scores)
display_scores(forest_rmse)

### saving the models
from sklearn.externals import joblib
joblib.dump(forest_scores, "RF.pkl")
joblib.dump(lin_scores, "OLS.pkl")
joblib.dump(CV_scores, "REGTREE.pkl")


## gridsearch
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features': [2,3,4]}
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,
                          param_grid,
                          cv = 5,
                          scoring = 'neg_mean_squared_error')


grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_

cvres = grid_search.cv_results_
for mean_score, params in zip (cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)



## RandomizedSearchCV for random search when search space of hyperparameters is huge

### analyzing feature importance
feature_importance = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs, extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importance, attributes), reverse = True)


final_model = grid_search.best_estimator_
x_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)

from sklearn.metrics import mean_squared_error
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


############################## exercises ########################################################

### 1. SVM search

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = [
    {'kernel' : ['linear'], 'C' : [0.01, 0.1]},
    {'kernel' : ['rbf'], 'C' : [0.01, 0.1], 'gamma' : [0.01, 0.02]}
]

svm_svc_reg = SVR()
grid_search_svr = GridSearchCV(svm_svc_reg,
                          param_grid,
                          cv = 5,
                          scoring = 'neg_mean_squared_error')


grid_search_svr.fit(housing_prepared, housing_labels)
best_svm_model = grid_search_svr.best_estimator_

# fitting to predictors
x_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = best_svm_model.predict(x_test_prepared)

from sklearn.metrics import mean_squared_error
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

### 2. using randomized seach 
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

param_distributions = [
    {'kernel' : ['rbf'],
     'gamma' : [expon(scale = 1.0)],
     'C' : [reciprocal(20, 20000)]}
]

grid_search_svr_randomized = RandomizedSearchCV(svm_svc_reg,
                          param_distributions,
                          cv = 5,
                          scoring = 'neg_mean_squared_error',
                          n_jobs = -1,
                          verbose = 2)
grid_search_svr_randomized.fit(housing_prepared, housing_labels)
best_svm_model = grid_search_svr_randomized.best_estimator_

x_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = best_svm_model.predict(x_test_prepared)

from sklearn.metrics import mean_squared_error
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

### 3. modified pipeline
from sklearn.feature_selection import VarianceThreshold


def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])



class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

k = 5
feature_importances = grid_search.best_estimator_.feature_importances_
top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices


num_pipeline = Pipeline(
[
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy = "median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder())
])

## joining the pipelines
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion (transformer_list = [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])


x_training = strat_train_set.drop("median_house_value", axis = 1)
x_train_prepared = num_pipeline.transform(x_training)
VarianceThreshold(x_train_prepared)

full_pipeline = Pipeline (
    [('preparation', full_pipeline),
    ('feature_selector', TopFeatureSelector(feature_importances, k))],
    )

full_pipeline.fit_transform(x_training)


### 4. Single pipeline

single_pipeline = Pipeline (
    [('preparation', full_pipeline),
    ('feature_selector', TopFeatureSelector(feature_importances, k)),
    ('SVM_reg', RandomForestRegressor(**grid_search.best_params_))],
    )

single_pipeline.fit_transform(x_training)

### 5. GridSearchCV for automatically explore preparation options

param_grid = [{
    'strategy' : ['median', 'median', 'most_frequent'],
    #'selected_features': list(range(1, len(feature_importances) + 1))
    }]

search_prep = GridSearchCV(single_pipeline,
                           param_grid,
                           cv = 10,
                           scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
search_prep.fit(housing, housing_labels)

####################################### APPENDIX ################################################
### exercises
## 1.
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, 
                           param_grid, 
                           cv=5, 
                           scoring='neg_mean_squared_error', 
                           verbose=2, 
                           n_jobs=4)
grid_search.fit(housing_prepared, housing_labels)

negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
grid_search.best_params_


##2.
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=4, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
grid_search.best_params_
