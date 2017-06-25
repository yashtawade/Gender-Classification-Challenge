# Importing necessary libraries

import numpy as np
from sklearn import tree  #DecisionTree
from sklearn.svm import SVC #Support Vector Machines
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive-Bayes
from sklearn.linear_model import Perceptron #Perceptron 
from sklearn.metrics import accuracy_score #Calculating accuracy


# Test data and labels
X_train = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
y_train = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


# Creating objects of all classifier classes
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_knn = KNeighborsClassifier()
clf_nb = GaussianNB()

# Training the models
clf_tree.fit(X_train, y_train)
clf_svm.fit(X_train, y_train)
clf_perceptron.fit(X_train, y_train)
clf_knn.fit(X_train, y_train)
clf_nb.fit(X_train, y_train)

# Test data
X_test=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
y_test=['male','male','male','female','female','female','male','male']

# Testing using above test data
pred_tree = clf_tree.predict(X_test)
acc_tree = accuracy_score(y_test, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(X_test)
acc_per = accuracy_score(y_test, pred_per) * 100
print('Accuracy for Perceptron: {}'.format(acc_per))

pred_KNN = clf_knn.predict(X_test)
acc_KNN = accuracy_score(y_test, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

pred_nb = clf_nb.predict(X_test)
acc_nb = accuracy_score(y_test, pred_nb) * 100
print('Accuracy for Naive Bayes: {}'.format(acc_nb))


#  Choosing the best classifier 
prediction_index = np.argmax([acc_svm, acc_per, acc_KNN ,acc_nb]) # np.argmax chooses the biggest value
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN' , 3: 'Naive Bayes'} # Creating a dictionary with numbers as the keys and Classifier names as values
print('Best classifier is {}'.format(classifiers[prediction_index]))