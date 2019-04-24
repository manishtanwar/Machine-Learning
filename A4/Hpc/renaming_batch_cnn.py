import sys
import glob
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA

base = 4519
# for file in sorted(glob.glob("batch_cnn/2/*")):
for file in sorted(glob.glob("batch_cnn/5/*")):
	# print(file)
	initial = "d" + file[1:13]
	num = int(file[13:-4]) + base
	newname = initial + str(num) + ".npy"
	# print(newname)
	os.rename(file, newname)