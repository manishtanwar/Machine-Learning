from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling2D
import sys
import numpy as np
import random 
from keras.models import load_model
from sklearn.metrics import f1_score
import tensorflow as tf

def f1_score_call(y_true, y_pred):
	return f1_score(y_true, y_pred)
	# return f1_score(y_true, y_pred, average='binary')
# 2.66% 1
# seq_per_episode = 13
def generate_test_data(Xt,Yt):
	X = []
	for i in range(Yt.shape[0]):
		ele = Xt[i*5]
		for j in range(4):
			ele = np.append(ele,Xt[i*5+j+1],axis=2)
		X.append(ele)
	return (np.asarray(X), np.asarray(Yt))

def get_model():
	model = Sequential()
	model.add(Conv2D(32, (3,3), strides=2, activation='relu', input_shape = (210,160,15)))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))

	model.add(Conv2D(64, (3,3), strides=2, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))

	model.add(Flatten())
	model.add(Dense(2048, activation='relu'))

	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score_call])
	return model

X_train = np.load("cnn_data_saved/X_generated.npy")
Y_train = np.load("cnn_data_saved/Y_generated.npy")
class_weight = {0: 1.,
                1: 5.}

model = get_model()
model.fit(X_train, Y_train, epochs=4, batch_size=128, class_weight=class_weight)
model.save('model_cnn')

# model = load_model('model_cnn')

X_test = np.asarray(np.load("cnn_data_saved/X_val.npy"))
Y_test = np.asarray(np.load("cnn_data_saved/Y_val.npy"))
(X_test, Y_test) = generate_test_data(X_test, Y_test[:,1])

score = model.evaluate(X_test, Y_test, batch_size=128)
print("Score:",score)
y_pred = model.predict_classes(X_test)
# print(y_pred)
print(Y_test.shape, y_pred.shape)
print(sum(Y_test == y_pred[:,0]))

f1 = f1_score(Y_test, y_pred[:,0], average='binary')
print(f1)