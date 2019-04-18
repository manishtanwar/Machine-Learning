import sys
import numpy as np
from sklearn import svm
import sklearn

def linear_svm(X, Y, X_test, Y_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(X,Y)
    Y_pred = clf.predict(X_test)
    # f1_score = sklearn.metrics.f1_score(y_actual, y_pred, average = None)

# def gaussian_svm(X, Y, X_test, Y_test):
#     clf = svm.SVC(kernel='rbf')
#     clf.fit(X,Y)
#     return clf.predict(X_test)

X = np.load("saved_small/X_reduced.npy")
Y = np.load("saved_small/Y.npy")
# print(type(X), type(Y), X.shape, Y.shape)
# (Xg, Yg) = generate_train_data()
