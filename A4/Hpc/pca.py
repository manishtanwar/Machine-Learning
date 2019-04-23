import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib

def get_pca_model():
	# Xpca = []
	# folder_cnt = 0
	# for folder in sorted(glob.glob("train_dataset/*")):
	# 	for img in sorted(glob.glob(folder + "/*.png")):
	# 		im = Image.open(img).convert("L")
	# 		data = np.asarray(im).flatten() / 255.
	# 		Xpca.append(data)
	# 	folder_cnt += 1
	# 	print("folder_cnt:",folder_cnt)
	# 	sys.stdout.flush()
	# 	if(folder_cnt == 50):
	# 		break
	# ipca = IncrementalPCA(n_components=50, batch_size=100)
	# ipca.fit(Xpca)
	# joblib.dump(ipca, "saved_pca/pca_model")

	ipca = joblib.load("saved_pca/pca_model")
	return ipca

ipca = get_pca_model()

print("pca variance ratio : \n", ipca.explained_variance_ratio_)
print("total variance :", np.sum(ipca.explained_variance_ratio_))
sys.stdout.flush()

X_red = []
Y = []
folder_cnt = 0
start_folder = 400

for folder in sorted(glob.glob("train_dataset/*")):
	if(folder_cnt < start_folder):
		folder_cnt += 1
		continue
	Y.append(np.genfromtxt(folder + "/rew.csv",delimiter=',',dtype="uint8"))
	for img in sorted(glob.glob(folder + "/*.png")):
		im = Image.open(img).convert("L")
		data = np.asarray(im).flatten()
		data = data.reshape(1,data.shape[0]) / 255.0
		X_red.append(ipca.transform(data).flatten())
	folder_cnt += 1
	print("folder_cnt:",folder_cnt)
	sys.stdout.flush()
	# if(folder_cnt == 400):
	# 	break

X_red = np.asarray(X_red)
print("X_red.shape", X_red.shape)
print("X_red.type", type(X_red[0][0]))
np.save("saved_pca/X4", X_red)
np.save("saved_pca/Y4", Y)

# validation data reduction

Xval = []
Yval = np.genfromtxt("validation_rewards.csv",delimiter=',',dtype="uint8")
folder_cnt = 0
for folder in sorted(glob.glob("validation_dataset/*")):
	x = np.zeros(0)
	for img in sorted(glob.glob(folder + "/*.png")):
		im = Image.open(img).convert("L")
		data = np.asarray(im).flatten()
		data = data.reshape(1,data.shape[0]) / 255.0
		x = np.append(x, ipca.transform(data).flatten(),axis=0)
		# Xval.append(ipca.transform(data).flatten())
	Xval.append(x)
	folder_cnt += 1
	print("folder_cnt:",folder_cnt)
	sys.stdout.flush()
	# if(folder_cnt == 5):
	# 	break

Xval = np.asarray(Xval)
Yval = np.asarray(Yval)[:,1]
print("Xval.shape", Xval.shape, "Yval.shape", Yval.shape)
np.save("saved_pca/Xval", Xval)
np.save("saved_pca/Yval", Yval)