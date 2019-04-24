# import sys
# import glob
# import numpy as np
# from PIL import Image
# from sklearn.decomposition import PCA
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling2D
# from keras.models import load_model
# import random 
# from sklearn.metrics import f1_score
# import tensorflow as tf
# from keras import backend as K
# from keras.callbacks import ModelCheckpoint
# from keras.layers import LeakyReLU

# def f1(Y_true, Y_pred):
# 	true_positives = K.sum(K.round(K.clip(Y_true * Y_pred, 0, 1)))
# 	possible_positives = K.sum(K.round(K.clip(Y_true, 0, 1)))
# 	recall = true_positives / (possible_positives + K.epsilon())
# 	predicted_positives = K.sum(K.round(K.clip(Y_pred, 0, 1)))
# 	precision = true_positives / (predicted_positives + K.epsilon())
# 	return 2*((precision*recall)/(precision+recall+K.epsilon()))

# model = load_model('cnn_models/model_base3k_batches800_epochs5', custom_objects={'f1': f1})

# def generate_test_data(Xt):
# 	ele = Xt[0]
# 	for j in range(4):
# 		ele = np.append(ele,Xt[j+1],axis=2)
# 	return ele

# start_folder = 0
# end_folder = 115000
# folder_cnt = 0
# w = 160
# h = 210
# Y_pred = np.zeros(0)
# X = []
# for folder in sorted(glob.glob("test_dataset/*")):
# 	data_point = []
# 	for img in sorted(glob.glob(folder + "/*.png")):
# 		im_rgb = Image.open(img).crop((3,27,w-3,h))
# 		data_rgb = np.asarray(im_rgb, dtype=np.float32)/255.0
# 		data_point.append(data_rgb)
# 	X.append(generate_test_data(data_point))
# 	folder_cnt += 1
# 	if(folder_cnt == 100):
# 		# print(np.asarray(X).shape)
# 		Y_pred = np.append(Y_pred, model.predict_classes(np.asarray(X)).flatten(), axis=0)
# 		X.clear()
# 		print("folder_cnt:",folder_cnt)
# 	# if(folder_cnt == 10):
# 	# 	break
# 	sys.stdout.flush()
# Y_pred = np.append(Y_pred, model.predict_classes(np.asarray(X)).flatten(), axis=0)
# print(Y_pred.shape)

# for i in range(Y_pred.shape[0]):
# 	print(Y_pred[i])
# # print(Y_pred)

import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling2D
from keras.models import load_model
import random 
from sklearn.metrics import f1_score
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU


def f1(Y_true, Y_pred):
	true_positives = K.sum(K.round(K.clip(Y_true * Y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(Y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	predicted_positives = K.sum(K.round(K.clip(Y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

# model1 = load_model('cnn_models/model_base1k_batches1k_epochs10', custom_objects={'f1': f1})
# model2 = load_model('cnn_models/model_base1k_batches3k_epochs10_bs512', custom_objects={'f1': f1})
# model3 = load_model('cnn_models/model_base3k_batches800_epochs5', custom_objects={'f1': f1})

model1 = load_model('cnn_models/base2k_bcnt300_bsize512_cpu', custom_objects={'f1': f1})

def generate_test_data(Xt):
	ele = Xt[0]
	for j in range(4):
		ele = np.append(ele,Xt[j+1],axis=2)
	return ele

start_folder = 0
end_folder = 115000
folder_cnt = 0
w = 160
h = 210
Y_pred = []
for folder in sorted(glob.glob("test_dataset/*")):
	data_point = []
	for img in sorted(glob.glob(folder + "/*.png")):
		im_rgb = Image.open(img).crop((3,27,w-3,h))
		data_rgb = np.asarray(im_rgb, dtype=np.float32)/255.0
		data_point.append(data_rgb)
	x = generate_test_data(data_point)[np.newaxis,:]
	''' multiple models
	pred = model1.predict_classes(x) + model2.predict_classes(x) + model3.predict_classes(x)
	if(pred >= 2):
		Y_pred.append(1)
	else:
		Y_pred.append(0)
	'''
	pred = model1.predict_classes(x)
	Y_pred.append(pred)
	folder_cnt += 1
	# if(folder_cnt == 10):
	# 	break
	print("folder_cnt:",folder_cnt)
	sys.stdout.flush()

Y_pred = np.asarray(Y_pred).flatten()
for i in range(Y_pred.shape[0]):
	print(Y_pred[i])
# print(Y_pred.shape)
# print(Y_pred)