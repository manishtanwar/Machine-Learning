import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

Xval = []
Yval = np.genfromtxt("validation_rewards.csv",delimiter=',',dtype="uint8")
print(Yval.shape)

def generate_test_data(Xt,Yt):
	X = []
	for i in range(Yt.shape[0]):
		ele = Xt[i*5]
		for j in range(4):
			ele = np.append(ele,Xt[i*5+j+1],axis=2)
		X.append(ele)
	return (np.asarray(X), np.asarray(Yt))

folder_cnt = 0
w = 160
h = 210
for folder in sorted(glob.glob("validation_dataset/*")):
    for img in sorted(glob.glob(folder + "/*.png")):
        im_rgb = Image.open(img).crop((3,27,w-3,h))
        data_rgb = np.asarray(im_rgb)
        Xval.append(data_rgb)
        # break
    folder_cnt += 1
    print("folder_cnt:",folder_cnt)
    sys.stdout.flush()
    # if(folder_cnt == 5):
    #     break

Yval = Yval[0:folder_cnt]
Yval = Yval[:,1]
(Xval, Yval) = generate_test_data(Xval, Yval)

print("Xval.len:",len(Xval))
print("Yval.len:",len(Yval))
print("Xval[0].shape", Xval[0].shape)
print("Yval.shape", Yval.shape)
# print(Yval)
sys.stdout.flush()

np.save("cnn_data_saved/val_crop/X_val", Xval)
np.save("cnn_data_saved/val_crop/Y_val", Yval)