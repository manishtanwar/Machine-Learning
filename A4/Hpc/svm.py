import sys
import numpy as np
from sklearn import svm
import sklearn
import random

def decision(p):
	return random.random() < p

seq_per_episode = 13

def generate_train_data(X, Y):
	X_gen = []
	Y_gen = []
	till_now = 0
	for episode in range(len(Y)):
		for seq_no in range(seq_per_episode):
			start_index = random.randint(0,Y[episode].shape[0]-8) + till_now
			y_label = Y[episode][start_index+7]
			img_list = np.arange(start_index, start_index+7)
			for i in range(0,6):
				for j in range(i+1,6):
					final_img_list = np.delete(img_list,[i,j])
					X_gen.append(X[final_img_list,:])
					Y_gen.append(y_label)
		till_now += Y[episode].shape[0]
	return (X_gen, Y_gen)


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
