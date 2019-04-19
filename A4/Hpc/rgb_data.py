import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# im = Image.open(sys.argv[1]).convert("L")
# data = np.asarray(im)
X = []
# X_rgb = []
Y = []
folder_cnt = 0
for folder in sorted(glob.glob("train_dataset/*")):
    for img in sorted(glob.glob(folder + "/*.png")):
        # im_rgb = Image.open(img)
        # im = im_rgb.convert("L")
        # data_rgb = np.asarray(im_rgb).flatten()
        # X_rgb.append(data_rgb)

        im = Image.open(img)
        data = np.asarray(im)
        X.append(data)
        # break
    Y.append(np.genfromtxt(folder + "/rew.csv",delimiter=',',dtype="uint8"))
    folder_cnt += 1
    print("folder_cnt:",folder_cnt)
    sys.stdout.flush()
    if(folder_cnt == 100):
        break

np.save("saved_small/X_rgb_100",X)
np.save("saved_small/Y_100",Y)

print("X.len:",len(X))
print("Y.len:",len(Y))
print("X[0].shape", X[0].shape)
print("Y[0].shape", Y[0].shape)
print("Y[0][0].type", type(Y[0][0]))
print("X[0][0].type", type(X[0][0]))