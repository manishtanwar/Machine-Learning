import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling2D
from keras.models import load_model
import sys
import numpy as np
import random 
from sklearn.metrics import f1_score
import tensorflow as tf

# 2.66% 1
# seq_per_episode = 13
batch_size = 128
no_batches = 1500

# 3498
def f_score(y_true, y_pred):
	tp = tn = fp = fn = 0
	for i in range(len(y_true)):
		if(y_true[i] == 1 and y_pred[i] == 1):
			tp += 1
		if(y_true[i] == 0 and y_pred[i] == 0):
			tn += 1
		if(y_true[i] == 0 and y_pred[i] == 1):
			fp += 1
		if(y_true[i] == 1 and y_pred[i] == 0):
			fn += 1
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	return (2*precision*recall) / (precision + recall)

class Data_generator(keras.utils.Sequence):
	def __init__(self, batch_size, shuffle = True):
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.on_epoch_end()
	
	def __len__(self):
		return no_batches
	
	def __getitem__(self, index):
		x = np.load("batch_cnn/X" + str(index) + ".npy")
		y = np.load("batch_cnn/Y" + str(index) + ".npy")
		return x,y

	def on_epoch_end(self):
		self.indexes = np.arange(no_batches * batch_size)
		if(self.shuffle):
			np.random.shuffle(self.indexes)

def get_model():
	model = Sequential()
	# (210,160,15)
	model.add(Conv2D(32, (3,3), strides=2, activation='relu', input_shape = (183,154,15)))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))

	model.add(Conv2D(64, (3,3), strides=2, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))

	model.add(Flatten())
	model.add(Dense(2048, activation='relu'))

	model.add(Dense(1, activation='sigmoid'))
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model

class_weight = {0: 1.0,
                1: 1.0}

model = get_model()
training_generator = Data_generator(batch_size)

model.fit_generator(generator = training_generator, epochs=10, class_weight=class_weight, use_multiprocessing=True, workers=50)
model.save('model_cnn_epoch10_1500')

# model = load_model('model_cnn')

X_test = np.asarray(np.load("cnn_data_saved/val/X_val.npy"))
Y_test = np.asarray(np.load("cnn_data_saved/val/Y_val.npy"))

y_pred = model.predict_classes(X_test)
accuracy = (sum(Y_test == y_pred[:,0])) / y_pred.shape[0]
f1 = f1_score(Y_test, y_pred[:,0], average='binary')

unka_fscore = f_score(Y_test, y_pred[:, 0])
print("f1:",f1, "accuracy:", accuracy, "unka_fscore", unka_fscore)