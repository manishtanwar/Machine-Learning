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
batch_per_file = 265
files_cnt = 1

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
	def __init__(self, batch_size, dimension=(210,160,15), shuffle = True):
		self.batch_size = batch_size
		self.dimension = dimension
		self.shuffle = shuffle
	
	def __len__(self):
		return batch_per_file * files_cnt
		# batch_per_epoch = batch_per_file * files_cnt
		# return int(total_seq // self.batch_size)
	
	def __getitem__(self, index):
		file_no = int(index // batch_per_file) + 1
		infile_index = index % batch_per_file
		start = infile_index * self.batch_size
		end = (infile_index + 1) * self.batch_size
		y_whole = np.load("cnn_data_saved/Y_generated" + str(file_no) + ".npy")

		if(end > y_whole.shape[0]):
			start -= end-y_whole.shape[0]
			end = y_whole.shape[0]
		x = np.load("cnn_data_saved/X_generated" + str(file_no) + ".npy")[start:end]
		y = y_whole[start:end]
		
		return x,y

	def on_epoch_end(self):
		self.indexes = np.arange(batch_per_file * files_cnt * batch_size)
		if(self.shuffle):
			np.random.shuffle(self.indexes)


def get_model():
	model = Sequential()
	model.add(Conv2D(32, (3,3), strides=2, activation='relu', input_shape = (210,160,15)))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))

	model.add(Conv2D(64, (3,3), strides=2, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))

	model.add(Flatten())
	model.add(Dense(2048, activation='relu'))

	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

class_weight = {0: 1.0,
                1: 1.3}

model = get_model()
training_generator = Data_generator(batch_size)

model.fit_generator(generator = training_generator, epochs=1, class_weight=class_weight)
model.save('model_cnn')
# model = load_model('model_cnn')

X_test = np.asarray(np.load("cnn_data_saved/val/X_val.npy"))
Y_test = np.asarray(np.load("cnn_data_saved/val/Y_val.npy"))

score = model.evaluate(X_test, Y_test, batch_size=128)
print("Score:",score)
y_pred = model.predict_classes(X_test)
# print(y_pred)
print(Y_test.shape, y_pred.shape)
print(sum(Y_test == y_pred[:,0]))

f1 = f1_score(Y_test, y_pred[:,0], average='binary')
print(f1)