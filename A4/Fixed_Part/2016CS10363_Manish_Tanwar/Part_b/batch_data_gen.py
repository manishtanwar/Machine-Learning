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
start_folder = 400
folder_cnt = 0
positive_cnt = 0
negative_cnt = 0

batch_size = 128
w = 160
h = 210
cur = 0
file_no = 0

def go(X,Y):
	global file_no
	np.save("batch_cnn/5/X" + str(file_no),X)
	np.save("batch_cnn/5/Y" + str(file_no),Y)
	file_no += 1
	X.clear()
	Y.clear()

# 183 * 154
for folder in sorted(glob.glob("train_dataset/*")):
	if(folder_cnt < start_folder):
		folder_cnt += 1
		continue
	y_folder = np.genfromtxt(folder + "/rew.csv",delimiter=',',dtype="uint8")
	X_folder = []
	for img in sorted(glob.glob(folder + "/*.png")):
		im = Image.open(img).crop((3,27,w-3,h))
		data = np.asarray(im)
		X_folder.append(data)
	for img in range(6,y_folder.shape[0]):
		if(y_folder[img] == 1):
			if(decision(1./5.)):
				continue
			start_index = img-6
			y_label = y_folder[img]
			img_list = np.arange(start_index, start_index+7)
			for i in range(0,6):
				for j in range(i+1,6):
					if(decision(1./3.)):
						continue
					final_list = np.delete(img_list,[i,j])
					stacked = X_folder[final_list[0]]
					for k in range(4):
						stacked = np.append(stacked, X_folder[final_list[k+1]], axis=2)
					X.append(stacked)
					Y.append(y_label)
					positive_cnt += 1
					cur += 1
					if(cur == batch_size):
						go(X,Y)
						cur = 0
		elif(decision(1./20.)):
			start_index = img-6
			y_label = y_folder[img]
			img_list = np.arange(start_index, start_index+7)
			for i in range(0,6):
				for j in range(i+1,6):
					if(decision(1./4.)):
						continue
					final_list = np.delete(img_list,[i,j])
					stacked = X_folder[final_list[0]]
					for k in range(4):
						stacked = np.append(stacked, X_folder[final_list[k+1]], axis=2)
					X.append(stacked)
					Y.append(y_label)
					negative_cnt += 1
					cur += 1
					if(cur == batch_size):
						go(X,Y)
						cur = 0
	folder_cnt += 1
	print("folder_cnt:",folder_cnt)
	sys.stdout.flush()
	# if(folder_cnt == 400):
	# 	break

print("+",positive_cnt, "-", negative_cnt, "file_cnts", file_no)
# np.save("cnn_data_saved/X_generated" + str(file_number),X)
# np.save("cnn_data_saved/Y_generated" + str(file_number),Y)