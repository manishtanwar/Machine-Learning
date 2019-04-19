import sys
import numpy as np
from sklearn import svm
import sklearn
import random

seq_pre_episode = 13

def generate_train_data(X, Y):
    X_gen = []
    Y_gen = []
    for episode in range(len(Y)):
        start_index = random.randint(0,Y[episode].shape[0]-8)
        y_label = Y[episode][]

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
