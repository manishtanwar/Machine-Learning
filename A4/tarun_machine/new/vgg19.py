from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras import *
import os

def VGG19(include_top=True, input_tensor=None, input_shape=(183,154,15), pooling=None, classes=1, **kwargs):

	if input_tensor is None:
		img_input = layers.Input(shape=input_shape)
	else:
		if not backend.is_keras_tensor(input_tensor):
			img_input = layers.Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor

	x = layers.Conv2D(64, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block1_conv1')(img_input)
	x = layers.Conv2D(64, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block1_conv2')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = layers.Conv2D(128, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block2_conv1')(x)
	x = layers.Conv2D(128, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block2_conv2')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = layers.Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv1')(x)
	x = layers.Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv2')(x)
	x = layers.Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv3')(x)
	x = layers.Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv4')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = layers.Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv1')(x)
	x = layers.Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv2')(x)
	x = layers.Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv3')(x)
	x = layers.Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv4')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = layers.Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv1')(x)
	x = layers.Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv2')(x)
	x = layers.Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv3')(x)
	x = layers.Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv4')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	if include_top:
		# Classification block
		x = layers.Flatten(name='flatten')(x)
		x = layers.Dense(2048, activation='relu', name='fc1')(x)
		X = layers.Dropout(0.5)
		x = layers.Dense(1024, activation='relu', name='fc2')(x)
		X = layers.Dropout(0.3)
		# x = layers.Dense(classes, activation='softmax', name='predictions')(x)
		x =	layers.Dense(1, activation='sigmoid', name='predictions')(x)
	else:
		if pooling == 'avg':
			x = layers.GlobalAveragePooling2D()(x)
		elif pooling == 'max':
			x = layers.GlobalMaxPooling2D()(x)

	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	if input_tensor is not None:
		inputs = keras_utils.get_source_inputs(input_tensor)
	else:
		inputs = img_input
	# Create model.
	model = models.Model(inputs, x, name='vgg19')

	return model