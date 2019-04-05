import sys
import csv
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

def read_input(file):
	# input
	x = []
	y = []
	with open(file, 'r') as csvfile:
		reader = csv.reader(csvfile)

		limit = 0
		for row in reader:
			limit += 1
			if(limit >= 3):
				xl = []
				ll = 0
				for i in row:
					ll += 1
					if(ll > 1):
						xl.append(int(i))
				x.append(xl)
	input = np.array(x)

	x = input[:,:-1]
	y = input[:,-1]
	y = y[:,np.newaxis]
	return (x,y)

(X,Y) = read_input(train_file)
# print(X.shape, Y.shape)

classifier = DecisionTreeClassifier()
classifier = classifier.fit(X, Y)

def test(file):
	(Xv,Yv) = read_input(file)
	Ypv = classifier.predict(Xv)
	return 100.0 * np.sum(Yv[:,0] == Ypv) / Ypv.shape[0]

# print(Xv.shape, Yv.shape, Ypv.shape)

print("Test Accuracy:", test(test_file))
print("Train Accuracy:", test(train_file))
print("Validation Accuracy:", test(valid_file))
# start_time = time.time()
# end_time = time.time()
# print("Time taken:", end_time - start_time)
