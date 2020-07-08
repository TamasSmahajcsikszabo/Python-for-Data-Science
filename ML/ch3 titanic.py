## tackling the Titanic data ##

### data loading
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt

train = ps.read_csv("C:\\Users\\tamas\\OneDrive\\python_code\\ML\\datasets\\titanic\\train.csv")
test = ps.read_csv("C:\\Users\\tamas\\OneDrive\\python_code\\ML\\datasets\\titanic\\test.csv")

y_train = train["Survived"].to_numpy()
x_train = train.drop(["Survived","Name", "Ticket", "PassengerId"], axis = 1)


### preprocessing
## label encoding for sex
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
x_train["Sex"] = LE.fit_transform(x_train["Sex"])

## label encoding for embarking
from sklearn.impute import SimpleImputer
SImp = SimpleImputer(strategy = "constant", missing_values = np.nan, fill_value = "M")
x_train["Embarked"] = SImp.fit_transform(np.array(x_train["Embarked"]).reshape(-1,1))
x_train["Embarked"] = LE.fit_transform(x_train["Embarked"])

## feature engineering on cabin
SImp = SimpleImputer(strategy = "constant", missing_values = "NaN", fill_value = 0)
x_train["Cabin"] = SImp.fit_transform(np.array(x_train["Cabin"]).reshape(-1,1))

x_train["Deck"] = x_train["Cabin"].str.get(0)
SImp = SimpleImputer(strategy = "constant", missing_values = np.nan, fill_value = "X")
x_train["Deck"] = SImp.fit_transform(np.array(x_train["Deck"]).reshape(-1,1))
x_train["Deck"] = LE.fit_transform(x_train["Deck"])


## age imputation
from sklearn.impute import SimpleImputer
SImp2 = SimpleImputer(strategy = "median")
x_train["Age"] = SImp2.fit_transform(np.array(x_train["Age"]).reshape(-1,1))

x_train = x_train.drop(["Cabin"], axis = 1)


### model tuning
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

## grid search
from sklearn.model_selection import GridSearchCV

param_grid = [{'max_features': ['auto'],
               'n_estimators' : np.arange(1,19)}]

grid_search_rf = GridSearchCV(rf_clf,
                               param_grid,
                               cv = 5,
                               n_jobs = -1)

grid_search_rf.fit(x_train, y_train)
best_model = grid_search_rf.best_estimator_

cross_val_score(best_model, x_train, y_train, cv = 5)
predicted = cross_val_predict(best_model, x_train, y_train, cv = 5, method = "predict_proba")
predicted_survival = cross_val_predict(best_model, x_train, y_train, cv = 5)

import pandas as pd
pd.DataFrame(predicted)
predicted = predicted[:,1]

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

precision, recall, threshold = precision_recall_curve(y_train, predicted)
fpr, tpr, threshold_r = roc_curve(y_train, predicted)
auc = round(roc_auc_score(y_train, predicted),4)*100
text = 'AUC: ' + str(auc) + ' %'

cross_val_score(best_model, x_train, y_train, cv = 5).mean()

### result with plots
grid_search_rf.best_params_

### ROC curve

### Precision vs Recall Plot
plt.figure(figsize = (8,8))
plt.suptitle("Titanic Survival Prediction with Random Forest Classifier")
plt.subplot(221)
plt.plot(threshold, precision[:-1], '--', label = "Precision")
plt.plot(threshold, recall[:-1], ':', label = "Recall")
plt.legend(loc = "bottom center")
plt.title("Recall vs Precision Plot")
plt.subplot(222)
plt.plot(recall[:-1], precision[:-1], ':')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Recall-Precision Profile")
plt.ylim(0,1)
plt.subplot(223)
plt.plot(fpr, tpr, ':')
plt.xlabel('1- Specifitity')
plt.ylabel('Sensitivity')
plt.title('ROC profile for RF with 10 trees')
plt.text(0.4, 0.2, text, color = 'blue')
plt.show()


### testing
test["Sex"] = LE.fit_transform(test["Sex"])

## label encoding for embarking
from sklearn.impute import SimpleImputer
SImp = SimpleImputer(strategy = "constant", missing_values = np.nan, fill_value = "M")
test["Embarked"] = SImp.fit_transform(np.array(test["Embarked"]).reshape(-1,1))
test["Embarked"] = LE.fit_transform(test["Embarked"])

## feature engineering on cabin
SImp = SimpleImputer(strategy = "constant", missing_values = "NaN", fill_value = 0)
test["Cabin"] = SImp.fit_transform(np.array(test["Cabin"]).reshape(-1,1))

test["Deck"] = test["Cabin"].str.get(0)
SImp = SimpleImputer(strategy = "constant", missing_values = np.nan, fill_value = "X")
test["Deck"] = SImp.fit_transform(np.array(test["Deck"]).reshape(-1,1))
test["Deck"] = LE.fit_transform(test["Deck"])


## age imputation
from sklearn.impute import SimpleImputer
SImp2 = SimpleImputer(strategy = "median")
test["Age"] = SImp2.fit_transform(np.array(test["Age"]).reshape(-1,1))

test = test.drop(["Cabin"], axis = 1)

### SOLUTION FROM THE BOOK

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector (BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
    ("imputer", SimpleImputer(strategy = "median"))
    ])

num_pipeline.fit_transform(train)

## imputer for the string variables
# let's use the most frequent imputer

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index = X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

cat_pipeline = Pipeline([
    ("Select_cat", DataFrameSelector(["Pclass","Sex","Embarked"])),
    ("Imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False))
    ])
cat_pipeline.fit_transform(train)

## joining the pipelines

from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list = [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)])

x_train = preprocess_pipeline.fit_transform(train)
y_train = train["Survived"]

from sklearn.svm import SVC
svc_clf = SVC(gamma="auto")
svc_clf.fit(x_train, y_train)

x_test = preprocess_pipeline.fit_transform(test)
y_pred = svc_clf.predict(x_test)

from sklearn.model_selection import cross_val_score
svm_score = cross_val_score(svc_clf, x_train, y_train, cv = 10)
svm_score.mean()

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators = 100, random_state=42)
forest_scores = cross_val_score(forest_clf, x_train, y_train, cv = 10)

## using boxplot to compare the two models
plt.figure(figsize = ((8,4)))
plt.plot([1]*10, svm_score, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_score, forest_scores], labels = ["SVM", "Random Forest"])
plt.ylabel("Accuracy", fontsize = 14)
plt.show()

## age bucket
train["AgeBucket"] = train["Age"] // 15 * 15
train[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()

## survival chance by relatives on board
train["Relatives"] = train["SibSp"] + train["Parch"]

train[["Relatives", "Survived"]].groupby(["Relatives"]).mean()