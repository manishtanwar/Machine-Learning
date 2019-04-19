from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
import sys
import numpy as np
import random 
from keras.models import load_model

seq_per_episode = 13

def generate_train_data(X, Y):
	X_gen = []
	Y_gen = []
	till_now = 0
	for episode in range(len(Y)):
		for seq_no in range(seq_per_episode):
			start_index = random.randint(0,Y[episode].shape[0]-8) + till_now
			y_label = Y[episode][start_index+7-till_now]
			img_list = np.arange(start_index, start_index+7)
			for i in range(0,6):
				for j in range(i+1,6):
					final_img_list = np.delete(img_list,[i,j])
					X_gen.append(X[final_img_list,:])
					Y_gen.append(y_label)
		till_now += Y[episode].shape[0]
	return (np.asarray(X_gen), np.asarray(Y_gen))

def generate_test_data(Xt,Yt):
	X = []
	for i in range(Yt.shape[0]):
		X.append(Xt[i*5:(i+1)*5])
	return (np.asarray(X), np.asarray(Yt))

def get_model():
	model = Sequential()
	model.add(Conv3D(32, (1,3,3), strides=2, activation='relu', input_shape = (5,210,160,3)))
	model.add(MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2)))

	model.add(Conv3D(64, (1,3,3), strides=2, activation='relu'))
	model.add(MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2)))

	model.add(Flatten())
	model.add(Dense(2048, activation='relu'))

	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

X = np.load("saved_small/X_rgb_100.npy")
Y = np.load("saved_small/Y_100.npy")
print(X.shape[0])
(X_train, Y_train) = generate_train_data(X,Y)
print(X_train.shape, Y_train.shape)

X_test_in = np.asarray(np.load("saved_small/X_val_rgb_3k.npy"))
Y_test_in = np.asarray(np.load("saved_small/Yval_3k.npy"))

(X_test, Y_test) = generate_test_data(X_test_in, Y_test_in[:,1])

print(X_test.shape, Y_test.shape)
model = get_model()
model.fit(X_train, Y_train, epochs=20, batch_size=128)
model.save('model_cnn_100_episodes')

# model = load_model('model_cnn_100_episodes')

score = model.evaluate(X_test, Y_test, batch_size=128)
print("Score:",score)