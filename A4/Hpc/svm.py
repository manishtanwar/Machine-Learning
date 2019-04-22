import sys
import numpy as np
from sklearn import svm
import sklearn
import random
from sklearn.externals import joblib

def decision(p):
	return random.random() < p

def generate_train_seq(X, Y):
	Xo = []
	Yo = []
	for img in range(7, Y.shape[0]):
		if(Y[img] == 1):
			if(decision(1./5.)):
				continue
			start_index = img-7
			y_label = Y[img]
			img_list = np.arange(start_index, start_index+7)
			for i in range(0,6):
				for j in range(i+1,6):
					if(decision(1./3.)):
						continue
					final_list = np.delete(img_list,[i,j])
					stacked = X[final_list[0]]
					for k in range(4):
						stacked = np.append(stacked, X[final_list[k+1]], axis=1)
					
					print("stacked.shape", stacked.shape)

					Xo.append(stacked)
					Yo.append(y_label)
					positive_cnt += 1
		elif(decision(1./20.)):
			start_index = img-7
			y_label = Y[img]
			img_list = np.arange(start_index, start_index+7)
			for i in range(0,6):
				for j in range(i+1,6):
					if(decision(1./4.)):
						continue
					final_list = np.delete(img_list,[i,j])
					stacked = X[final_list[0]]
					for k in range(4):
						stacked = np.append(stacked, X[final_list[k+1]], axis=1)
					
					print("stacked.shape", stacked.shape)
					
					Xo.append(stacked)
					Yo.append(y_label)
					negative_cnt += 1
	Xo = np.asarray(Xo)
	Yo = np.asarray(Yo)
	return (Xo,Yo)


def linear_svm(X, Y):
	clf = svm.SVC(kernel='linear', max_iter=10)
	clf.fit(X,Y)
	joblib.dump(clf, "saved_pca/svm_model_linear")
	return clf

def gaussian_svm(X, Y):
	clf = svm.SVC(kernel='rbf', max_iter=10)
	clf.fit(X,Y)
	joblib.dump(clf, "saved_pca/svm_model_gaussian")
	return clf

Xin = np.load("saved_pca/X.npy")
Yin = np.load("saved_pca/Y.npy")
print("Xin.shape", Xin.shape, "Yin.shape", Yin.shape)
(X, Y) = generate_train_seq(Xin, Yin)

Xtest = np.load("saved_pca/Xval")
Ytest = np.load("saved_pca/Yval")

# ********* Linear ***********
print('linear')
lin_clf = linear_svm(X,Y)
# lin_clf = joblib.load("saved_pca/svm_model_linear")
Ypred = lin_clf.predict(Xtest)
f1_score = sklearn.metrics.f1_score(Ytest, Ypred, average = None)
print("f1_score:")
print(f1_score)

accuracy = sklearn.metrics.accuracy_score(Ytest, Ypred)
print("accuracy:",accuracy)

confusion_matrix = sklearn.metrics.confusion_matrix(Ytest, Ypred)
print("confusion matrix:")
print(confusion_matrix)

# ********** Gaussian ******** 
print('Gaussian')
rbf_clf = gaussian_svm(X,Y)
# rbf_clf = joblib.load("saved_pca/svm_model_gaussian")
Ypred = rbf_clf.predict(Xtest)
f1_score = sklearn.metrics.f1_score(Ytest, Ypred, average = None)
print("f1_score:")
print(f1_score)

accuracy = sklearn.metrics.accuracy_score(Ytest, Ypred)
print("accuracy:",accuracy)

confusion_matrix = sklearn.metrics.confusion_matrix(Ytest, Ypred)
print("confusion matrix:")
print(confusion_matrix)