
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris

iris =  load_iris()

print(iris.data)
print(iris.feature_names)
print(iris.target_names)
print(iris.data.shape)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =4)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
classes = {0: "setosa", 1: "versicolor", 2:"virginica"}

X_new = [[4.7, 2.9, 5.7, 2.3],
         [5, 4, 2, 2]]

y_predict = knn.predict(X_new)


print(classes[y_predict[0]])
print(classes[y_predict[1]])

y_pred = knn.predict(X_test)
print(y_pred)

print(metrics.accuracy_score(y_test, y_pred))

