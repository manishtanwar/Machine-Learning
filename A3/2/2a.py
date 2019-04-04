import sys
import numpy as np
from sklearn import preprocessing

file = sys.argv[1]
out_file = sys.argv[2]
# start_time = time.time()
# end_time = time.time()
# print("Time taken:", end_time - start_time)

input = np.genfromtxt(file,delimiter=',')
X = input[:,:-1]
Y = input[:,-1]
encoder = preprocessing.OneHotEncoder(categories='auto')
encoder.fit(X)
labels = encoder.transform(X).toarray()
Y = Y[:,np.newaxis]
labels = np.append(labels, Y, axis=1)
# print(labels)
print(labels.shape)
la = labels[0:20]
print(la.shape)
np.save(out_file, la)