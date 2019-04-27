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

def f1(Y_true, Y_pred):
	true_positives = K.sum(K.round(K.clip(Y_true * Y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(Y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	predicted_positives = K.sum(K.round(K.clip(Y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

# here X_test is noramlized already
X_test = np.asarray(np.load("cnn_data_saved/val_crop/X_val1.npy"))
Y_test = np.asarray(np.load("cnn_data_saved/val_crop/Y_val1.npy"))

# *********** multiple models

# checkpointer = ModelCheckpoint(filepath='callback_best.hdf5', verbose=1, save_best_only=True)
# model.fit_generator(generator = training_generator, validation_data=(X_test, Y_test), epochs=5, class_weight=class_weight, use_multiprocessing=True, workers=20)
# model.save('cnn_models/model_base3k_batches800_epochs5_gpu1')
'''
model1 = load_model('cnn_models/All_old_waste/model_base1k_batches1k_epochs10', custom_objects={'f1': f1})
model2 = load_model('cnn_models/All_old_waste/model_base1k_batches3k_epochs10_bs512', custom_objects={'f1': f1})
model3 = load_model('cnn_models/All_old_waste/model_base3k_batches800_epochs5', custom_objects={'f1': f1})

y_pred1 = model1.predict_classes(X_test)[:,0]
y_pred2 = model2.predict_classes(X_test)[:,0]
y_pred3 = model3.predict_classes(X_test)[:,0]

y_pred_sum = y_pred1 + y_pred2 + y_pred3
# y_pred = 1 if y_pred_sum >= 2 else 0
y_pred = np.where(y_pred_sum >= 2, 1, 0)
# print(y_pred_sum, y_pred)
'''
model = load_model('cnn_models/base2k_bcnt300_bsize512_cpu', custom_objects={'f1': f1})
y_pred = model.predict_classes(X_test)[:,0]

accuracy = (sum(Y_test == y_pred)) / y_pred.shape[0]
f1 = f1_score(Y_test, y_pred, average='binary')

# unka_fscore = f_score(Y_test, y_pred[:, 0])
print("f1:",f1, "accuracy:", accuracy)