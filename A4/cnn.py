from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
import sys
import numpy as np

model = Sequential()
model.add(Conv3D(32, (5,3,3), strides=2, activation='relu', input_shape = (5,210,160,3)))
model.add(MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2)))

model.add(Conv3D(64, (5,3,3), strides=2, activation='relu'))
model.add(MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2)))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

