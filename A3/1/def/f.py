import sys
import numpy as np
import time
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
# np.set_printoptions(threshold=sys.maxsize)

np.random.seed(0)
train_file = sys.argv[1]
test_file = sys.argv[2]
valid_file = sys.argv[3]

# start_time = time.time()
# end_time = time.time()
# print("Time taken:", end_time - start_time)

input = np.genfromtxt(file,delimiter=',')
X = input[:,:-1]
Y = input[:,-1]
# encoder = preprocessing.OneHotEncoder(categories='auto')
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

