# import sys
# import numpy as np
# from sklearn import svm
# import sklearn
# import random

# X1 = np.genfromtxt("submission4.csv",delimiter=',',dtype=int)[1:30911]
# X2 = np.genfromtxt("submission5.csv",delimiter=',',dtype=int)[1:30911]
# X3 = np.genfromtxt("submission6.csv",delimiter=',',dtype=int)[1:30911]

# print(X1.shape)
# # print(X1)
# # print(X2)
# # print(X3)
# a = X1[:,1] + X2[:,1] + X3[:,1]
# a = np.where(a >= 2, 1, 0)
# for i in range(a.shape[0]):
# 	print(a[i])

import sys
import numpy as np
from sklearn import svm
import sklearn
import random

X1 = np.genfromtxt("submission9.csv",delimiter=',',dtype=int)[1:30911]
X2 = np.genfromtxt("submission5.csv",delimiter=',',dtype=int)[1:30911]
X3 = np.genfromtxt("submission6.csv",delimiter=',',dtype=int)[1:30911]
X4 = np.genfromtxt("submission7.csv",delimiter=',',dtype=int)[1:30911]
X5 = np.genfromtxt("submission8.csv",delimiter=',',dtype=int)[1:30911]

print(X1.shape)
# print(X1)
# print(X2)
# print(X3)
a = X1[:,1] + X2[:,1] + X3[:,1] + X4[:,1] + X5[:,1]
a = np.where(a >= 3, 1, 0)
for i in range(a.shape[0]):
	print(a[i])