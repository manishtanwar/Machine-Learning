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

def f1_(Y_true, Y_pred):
	true_positives = K.sum(K.round(K.clip(Y_true * Y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(Y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	predicted_positives = K.sum(K.round(K.clip(Y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1(Y_true, Y_pred):
	true_positives = K.sum((K.clip(Y_true * Y_pred, 0, 1)))
	possible_positives = K.sum((K.clip(Y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	predicted_positives = K.sum((K.clip(Y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

# def f1_loss(y_true, y_pred):
# 	return 1-f1(y_true, y_pred)

def f1_loss(Y_true, Y_pred):
	true_positives = K.sum(K.clip(Y_true * Y_pred, 0, 1))
	possible_positives = K.sum((K.clip(Y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	predicted_positives = K.sum((K.clip(Y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return 1.0 - (2*((precision*recall)/(precision+recall+K.epsilon())))

def f1_loss_(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

class Data_generator(keras.utils.Sequence):
	def __init__(self, batch_size):
		self.batch_size = batch_size
	
	def __len__(self):
		return no_batches
	
	def __getitem__(self, index):
		x = np.asarray(np.load("batch_cnn/X" + str(batch_base + index) + ".npy"), dtype=np.float32)/ 255.
		y = np.load("batch_cnn/Y" + str(batch_base + index) + ".npy")
		return x,y

def get_model():
	model = VGG19()
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
	model.compile(loss=f1_loss, optimizer='adam', metrics=['accuracy', f1])
	return model

class_weight = {0: 1.0,
                1: 1.0}

model = get_model()
training_generator = Data_generator(batch_size)

# here X_test is noramlized
X_test = np.asarray(np.load("X_val.npy"))
Y_test = np.asarray(np.load("Y_val.npy"))

checkpointer = ModelCheckpoint(filepath='vgg128_f3_0_1000.hdf5', verbose=1, save_best_only=True)
model.fit_generator(generator = training_generator, validation_data=(X_test, Y_test), callbacks=[checkpointer], epochs=10, class_weight=class_weight, use_multiprocessing=True, workers=6)
model.save('vgg128_f3_0_1000')

y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
accuracy = (sum(Y_test == y_pred[:,0])) / y_pred.shape[0]
f1 = f1_score(Y_test, y_pred[:,0], average='binary')

# unka_fscore = f_score(Y_test, y_pred[:, 0])
print("f1:",f1, "accuracy:", accuracy)