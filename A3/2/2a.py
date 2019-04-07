import sys
import numpy as np
from sklearn import preprocessing

file = sys.argv[1]
out_file = sys.argv[2]
input = np.genfromtxt(file,delimiter=',')
X = input[:,:-1]
Y = input[:,-1]

a = 4
b = 13
list = []
for i in range(10):
	list_local = []
	for j in range(a):
		list_local.append(j+1)
	list.append(list_local)
	(a,b) = (b,a)

encoder = preprocessing.OneHotEncoder(categories=list)
encoder.fit(X)
labels = encoder.transform(X).toarray()
Y = Y[:,np.newaxis]
labels = np.append(labels, Y, axis=1)
np.save(out_file, labels)
# print(X.shape, Y.shape, labels.shape)