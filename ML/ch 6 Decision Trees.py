from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
x = iris["data"][:,2:]
y = iris["target"]

tree_clf = DecisionTreeClassifier(max_depth = 2)
tree_clf.fit(x,y)

# visuals
from sklearn.tree import export_graphviz
export_graphviz(
    tree_clf,
    out_file = "C:\\Users\\tamas\\OneDrive\\python_code\\ML\\iris_tree.dot",
    feature_names = iris.feature_names[2:],
    class_names = iris.target_names,
    rounded = True,
    filled = True)

tree_clf.predict_proba([[5, 1.5]])
iris["target_names"][tree_clf.predict([[5, 1.5]])]

# decision trees for regression
from sklearn.tree import DecisionTreeRegressor

regr = DecisionTreeRegressor(max_depth = 2)

tree_reg = regr.fit(x,y)
tree_reg.predict([[5,1.5]])
iris["target_names"][int(tree_reg.predict([[5,1.5]]))]

export_graphviz(
    tree_reg,
    out_file = "C:\\Users\\tamas\\OneDrive\\python_code\\ML\\iris_tree_reg.dot",
    feature_names = iris.feature_names[2:],
    class_names = iris.target_names,
    rounded = True,
    filled = True)

from sklearn.datasets import load_boston

boston = load_boston()

import pandas as ps
x = boston["data"]
x_DF = ps.DataFrame(x)
x_DF.plot(kind = "scatter", alpha = 0.1)

import matplotlib.pyplot as plt
import numpy as np
lab = list(boston["feature_names"])
x_DF.hist(density=True,bins=30,label=lab)


y = boston["target"]
regr = DecisionTreeRegressor(max_depth = 4)
tree_reg = regr.fit(x,y)
tree_reg.fit(x,y)

import matplotlib.pyplot as plt
import numpy as np

predicted = tree_reg.predict(x)
plt.figure(figsize=(6,7))
plt.plot(np.round(predicted,1), y, 'bs')
plt.show()

