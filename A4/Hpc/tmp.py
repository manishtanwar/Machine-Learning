# import sys
# import glob
# import numpy as np
# from PIL import Image
# from sklearn.decomposition import PCA
# import random

# im = Image.open("image.png")
# data = np.asarray(im)
# im.save("original.png")
# w, h = im.size
# print(im.size)
# crop = im.crop((3,27,w-3,h))
# print(crop.size)
# crop.save("crop.png")

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling2D
# from keras.models import load_model
import sys
import numpy as np
import random 
# from sklearn.metrics import f1_score
# import tensorflow as tf
# from keras import backend as K
# from keras.callbacks import ModelCheckpoint
# from keras.layers import LeakyReLU

# print(K.tensorflow_backend._get_available_gpus())

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# X_test = np.asarray(np.load("cnn_data_saved/val_crop/X_val.npy"))
# print(X_test.shape, X_test[0][0][0][0], X_test[0][0][0][1], type(X_test[0][0][0][1]))
# X_test = np.load("saved_pca/X0.npy")
# print(X_test.shape, X_test[0][0], type(X_test[0][0]))

batch_base = index = 0
# x = np.load("batch_cnn/X" + str(batch_base + index) + ".npy")
# x = np.asarray(x, dtype=np.float32)/ 255.
x = np.asarray(np.load("batch_cnn/X" + str(batch_base + index) + ".npy"), dtype=np.float32)/ 255.
print(x.shape, x[0][0][0][0], type(x[0][0][0][0]))