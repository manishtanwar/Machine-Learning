import sys
import csv
import numpy as np
import time
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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
	# y = y[:,np.newaxis]
	return (x,y)

(X,Y) = read_input(train_file)
# print(X.shape, Y.shape)

categorical_feat = [1,2,3,5,6,7,8,9,10]

# def one_hot(X):
# 	encoder = preprocessing.OneHotEncoder(categories='auto', categorical_features = categorical_feat)
# 	encoder.fit(X)
# 	labels = encoder.transform(X).toarray()
# 	return labels

# print(X.shape, one_hot(X).shape)

categorical_trans = Pipeline(steps=[('one_hot_encode',preprocessing.OneHotEncoder(handle_unknown='ignore'))])
preProc = ColumnTransformer(transformers=[('categorical', categorical_trans, categorical_feat)])
classifier = Pipeline(steps=[('preProc',preProc),('classifier',DecisionTreeClassifier())])
classifier.fit(X, Y)

def test(file):
	(Xv,Yv) = read_input(file)
	Ypv = classifier.predict(Xv)
	return 100.0 * np.sum(Yv[:,0] == Ypv) / Ypv.shape[0]

# print(Xv.shape, Yv.shape, Ypv.shape)

(Xtest,Ytest) = read_input(test_file)
(Xvalid,Yvalid) = read_input(valid_file)

print("Test Accuracy:", 100.0 * classifier.score(Xtest,Ytest))
print("Train Accuracy:", 100.0 * classifier.score(X,Y))
print("Validation Accuracy:", 100.0 * classifier.score(Xvalid,Yvalid))
# start_time = time.time()
# end_time = time.time()
# print("Time taken:", end_time - start_time)