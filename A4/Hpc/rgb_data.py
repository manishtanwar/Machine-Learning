import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import random

def decision(p):
	return random.random() < p

X = []
Y = []
folder_cnt = 0
positive_cnt = 0
negative_cnt = 0
for folder in sorted(glob.glob("train_dataset/*")):
	y_folder = np.genfromtxt(folder + "/rew.csv",delimiter=',',dtype="uint8")
	X_folder = []
	for img in sorted(glob.glob(folder + "/*.png")):
		im = Image.open(img)
		data = np.asarray(im)
		X_folder.append(data)
	for img in range(7,y_folder.shape[0]):
		if(y_folder[img] == 1):
			positive_cnt += 1
			start_index = img-7
			y_label = y_folder[img]
			img_list = np.arange(start_index, start_index+7)
			for i in range(0,6):
				for j in range(i+1,6):
					final_list = np.delete(img_list,[i,j])
					stacked = X_folder[final_list[0]]
					for k in range(4):
						stacked = np.append(stacked, X_folder[final_list[k+1]], axis=2)
					X.append(stacked)
					Y.append(y_label)
		elif(decision(0.05)):
			negative_cnt += 1
			start_index = img-7
			y_label = y_folder[img]
			img_list = np.arange(start_index, start_index+7)
			for i in range(0,6):
				for j in range(i+1,6):
					final_list = np.delete(img_list,[i,j])
					stacked = X_folder[final_list[0]]
					for k in range(4):
						stacked = np.append(stacked, X_folder[final_list[k+1]], axis=2)
					X.append(stacked)
					Y.append(y_label)
	folder_cnt += 1
	print("folder_cnt:",folder_cnt)
	sys.stdout.flush()

np.save("saved/X_generated",X)
np.save("saved/Y_generated",Y)

print("+",positive_cnt, "-", negative_cnt)
print("X.len:",len(X))
print("Y.len:",len(Y))
# print("X[0].shape", X[0].shape)
# print("Y[0].shape", Y[0].shape)