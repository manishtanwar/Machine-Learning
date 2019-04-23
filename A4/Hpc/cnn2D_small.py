import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling2D
from keras.models import load_model
import sys
import numpy as np
import random 
from sklearn.metrics import f1_score
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU

# from tensorflow import set_random_seed
# from numpy.random import seed

# seed(3)
# set_random_seed(3)

# 2.66% 1
# seq_per_episode = 13
batch_size = 128
no_batches = 1000
batch_base = 200
# 3498
def f_score(y_true, y_pred):
	tp = tn = fp = fn = 0
	for i in range(y_true.shape[0]):
		if(y_true[i] == 1 and y_pred[i] == 1):
			tp += 1
		if(y_true[i] == 0 and y_pred[i] == 0):
			tn += 1
		if(y_true[i] == 0 and y_pred[i] == 1):
			fp += 1
		if(y_true[i] == 1 and y_pred[i] == 0):
			fn += 1
	if(tp + fp == 0):
		return 0.0
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	return (2*precision*recall) / (precision + recall)

def f1(Y_true, Y_pred):
	true_positives = K.sum(K.round(K.clip(Y_true * Y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(Y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	predicted_positives = K.sum(K.round(K.clip(Y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

class Data_generator(keras.utils.Sequence):
	def __init__(self, batch_size, shuffle = True):
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.on_epoch_end()
	
	def __len__(self):
		return no_batches
	
	def __getitem__(self, index):
		x = np.load("batch_cnn/X" + str(batch_base + index) + ".npy") / 255.
		y = np.load("batch_cnn/Y" + str(batch_base + index) + ".npy")
		return x,y

	def on_epoch_end(self):
		self.indexes = np.arange(no_batches * batch_size)
		if(self.shuffle):
			np.random.shuffle(self.indexes)

def get_model():
	model = Sequential()
	# (210,160,15)
	model.add(Conv2D(32, (3,3), strides=2, input_shape = (183,154,15)))
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))

	model.add(Conv2D(64, (3,3), strides=2))
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))

	model.add(Flatten())
	model.add(Dense(2048))
	model.add(LeakyReLU())

	# model.add(Dense(2, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
	# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', f1])
	return model

class_weight = {0: 1.0,
                1: 7.0/3.0}

model = get_model()
training_generator = Data_generator(batch_size)

# here X_test is noramlized
X_test = np.asarray(np.load("cnn_data_saved/val_crop/X_val.npy"))
Y_test = np.asarray(np.load("cnn_data_saved/val_crop/Y_val.npy"))

checkpointer = ModelCheckpoint(filepath='callback_best.hdf5', verbose=1, save_best_only=True)
model.fit_generator(generator = training_generator, validation_data=(X_test, Y_test), epochs=10, class_weight=class_weight, use_multiprocessing=True, workers=20)
model.save('cnn_models/model_base1k_batches200_epochs10')
# model = load_model('model_cnn')

y_pred = model.predict_classes(X_test)
accuracy = (sum(Y_test == y_pred[:,0])) / y_pred.shape[0]
f1 = f1_score(Y_test, y_pred[:,0], average='binary')

# unka_fscore = f_score(Y_test, y_pred[:, 0])
print("f1:",f1, "accuracy:", accuracy)