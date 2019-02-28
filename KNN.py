
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
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


accuracy = accuracy_score(y_test, y_pred)*100
k_list = list(range(1,50,2))
cv_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]
best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)