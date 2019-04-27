from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
import sys
import numpy as np

def generate_train_data(X, Y):
	X_gen = []
	Y_gen = []
	till_now = 0
	for episode in range(len(Y)):
		for seq_no in range(seq_per_episode):
			start_index = random.randint(0,Y[episode].shape[0]-8) + till_now
			y_label = Y[episode][start_index+7]
			img_list = np.arange(start_index, start_index+7)
			for i in range(0,6):
				for j in range(i+1,6):
					final_img_list = np.delete(img_list,[i,j])
					X_gen.append(X[final_img_list,:])
					Y_gen.append(y_label)
		till_now += Y[episode].shape[0]
	return (X_gen, Y_gen)

model = Sequential()
model.add(Conv3D(32, (5,3,3), strides=2, activation='relu', input_shape = (5,210,160,3)))
model.add(MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2)))

model.add(Conv3D(64, (5,3,3), strides=2, activation='relu'))
model.add(MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2)))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train = 

model.fit(X_train, Y_train, epochs=20, batch_size=128)
score = model.evaluate(X_test, Y_test, batch_size=128)
print("Score:",score)
