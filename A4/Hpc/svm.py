import sys
import numpy as np
from sklearn import svm
import sklearn
import random
from sklearn.externals import joblib


def decision(p):
	return random.random() < p

def generate_train_seq(X, Y):
	start = positive_cnt = negative_cnt = folder_cnt = 0
	Xo = []
	Yo = []
	for folder in range(len(Y)):
		y_folder = Y[folder]
		X_folder = X[start : start + y_folder.shape[0] + 1]
		start += y_folder.shape[0] + 1
		for img in range(7,y_folder.shape[0]):
			if(y_folder[img] == 1):
				if(decision(1./5.)):
					continue
				start_index = img-7
				y_label = y_folder[img]
				img_list = np.arange(start_index, start_index+7)
				for i in range(0,6):
					for j in range(i+1,6):
						if(decision(1./3.)):
							continue
						final_list = np.delete(img_list,[i,j])
						stacked = X_folder[final_list[0]]
						for k in range(4):
							stacked = np.append(stacked, X_folder[final_list[k+1]], axis=0)
						Xo.append(stacked)
						Yo.append(y_label)
						positive_cnt += 1
			elif(decision(1./20.)):
				start_index = img-7
				y_label = y_folder[img]
				img_list = np.arange(start_index, start_index+7)
				for i in range(0,6):
					for j in range(i+1,6):
						if(decision(1./4.)):
							continue
						final_list = np.delete(img_list,[i,j])
						stacked = X_folder[final_list[0]]
						for k in range(4):
							stacked = np.append(stacked, X_folder[final_list[k+1]], axis=0)
						Xo.append(stacked)
						Yo.append(y_label)
						negative_cnt += 1
		folder_cnt += 1
		print("folder_cnt:",folder_cnt)
		sys.stdout.flush()
	Xo = np.asarray(Xo)
	Yo = np.asarray(Yo)
	print("Xo.shape", Xo.shape, "Yo.shape", Yo.shape)
	print("+", positive_cnt, "-", negative_cnt, "total", start)
	return (Xo,Yo)


def linear_svm(X, Y):
	clf = svm.SVC(kernel='linear', max_iter=10000)
	clf.fit(X,Y)
	joblib.dump(clf, "saved_pca/svm_model_linear")
	return clf

def gaussian_svm(X, Y):
	clf = svm.SVC(kernel='rbf', max_iter=10000, gamma='auto')
	clf.fit(X,Y)
	joblib.dump(clf, "saved_pca/svm_model_gaussian")
	return clf

saved = 1
seq_cnt = 100000

if saved == 0:
	Xin = np.load("saved_pca/X0.npy")
	Xin = np.append(Xin, np.load("saved_pca/X1.npy"), axis=0)
	Xin = np.append(Xin, np.load("saved_pca/X2.npy"), axis=0)
	Xin = np.append(Xin, np.load("saved_pca/X3.npy"), axis=0)
	Xin = np.append(Xin, np.load("saved_pca/X4.npy"), axis=0)

	Yin = np.load("saved_pca/Y0.npy")
	Yin = np.append(Yin, np.load("saved_pca/Y1.npy"), axis=0)
	Yin = np.append(Yin, np.load("saved_pca/Y2.npy"), axis=0)
	Yin = np.append(Yin, np.load("saved_pca/Y3.npy"), axis=0)
	Yin = np.append(Yin, np.load("saved_pca/Y4.npy"), axis=0)
	print("Xin.shape", Xin.shape, "Yin.shape", Yin.shape)
	(X, Y) = generate_train_seq(Xin, Yin)
	np.save("saved_pca/X_seqf", X)
	np.save("saved_pca/Y_seqf", Y)
else:
	X = np.load("saved_pca/X_seqf.npy")[0:seq_cnt]
	Y = np.load("saved_pca/Y_seqf.npy")[0:seq_cnt]
print("X.shape", X.shape, "Y.shape", Y.shape)

Xtest = np.load("saved_pca/Xval.npy")
Ytest = np.load("saved_pca/Yval.npy")
print("Xtest.shape", Xtest.shape, "Ytest.shape", Ytest.shape)

# ********* Linear ***********
print('linear')
lin_clf = linear_svm(X,Y)
# lin_clf = joblib.load("saved_pca/svm_model_linear")
Ypred = lin_clf.predict(Xtest)
f1_score = sklearn.metrics.f1_score(Ytest, Ypred, average = None)
f1_b = sklearn.metrics.f1_score(Ytest, Ypred, average='binary')
print("f1_score:")
print(f1_score)
print("f1_b:", f1_b)

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
f1_b = sklearn.metrics.f1_score(Ytest, Ypred, average='binary')
print("f1_score:")
print(f1_score)
print("f1_b:", f1_b)

accuracy = sklearn.metrics.accuracy_score(Ytest, Ypred)
print("accuracy:",accuracy)

confusion_matrix = sklearn.metrics.confusion_matrix(Ytest, Ypred)
print("confusion matrix:")
print(confusion_matrix)