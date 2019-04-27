import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import load_model
import sys
import numpy as np
import random 
from sklearn.metrics import f1_score
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import os
from vgg19 import VGG19

os.environ["CUDA_VISIBLE_DEVICES"]="3"
batch_size = 128
no_batches = 1000
# total batches = 5670 of size 128
batch_base = 0
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

def one_hot(y):
	z0 = np.zeros(y.shape)
	z1 = np.zeros(y.shape)
	li0 = np.where(y==0)
	li1 = np.where(y==1)
	z0[li0] = 1
	z1[li1] = 1
	z0 = z0[:,np.newaxis]
	z1 = z1[:,np.newaxis]
	z0 = np.append(z0,z1,axis=1)
	return z0

class Data_generator(keras.utils.Sequence):
	def __init__(self, batch_size):
		self.batch_size = batch_size
	
	def __len__(self):
		return no_batches
	
	def __getitem__(self, index):
		x = np.asarray(np.load("batch_cnn/X" + str(batch_base + index) + ".npy"), dtype=np.float32)/ 255.
		y = one_hot(np.load("batch_cnn/Y" + str(batch_base + index) + ".npy"))
		return x,y

def get_model():
	model = VGG19()

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
	return model

class_weight = {0: 1.0,
                1: 7.0/3.0}

model = get_model()
training_generator = Data_generator(batch_size)

# here X_test is noramlized
X_test = np.asarray(np.load("X_val.npy"))
Y_test = np.asarray(np.load("Y_val.npy"))
Y_test_hot = one_hot(Y_test)

checkpointer = ModelCheckpoint(filepath='vgg128_0_1000.hdf5', verbose=1, save_best_only=True)
model.fit_generator(generator = training_generator, validation_data=(X_test, Y_test_hot), callbacks=[checkpointer], epochs=10, class_weight=class_weight, use_multiprocessing=True, workers=6)
model.save('vgg128_0_1000')

y_pred = model.predict(X_test)
y_ans = []
for i in range(y_pred.shape[0]):
	if(y_pred[i][0] > y_pred[i][1]):
		y_ans.append(0)
	else:
		y_ans.append(1)

y_ans = np.asarray(y_ans)
# print(y_ans.shape, y_ans)

accuracy = (sum(Y_test == y_ans)) / y_ans.shape[0]
f1 = f1_score(Y_test, y_ans, average='binary')

# unka_fscore = f_score(Y_test, y_ans)
print("f1:",f1, "accuracy:", accuracy)