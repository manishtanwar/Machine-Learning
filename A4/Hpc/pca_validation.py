import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

Xpca = []
X = []
X_rgb = []
Y = []
folder_cnt = 0

for folder in sorted(glob.glob("train_dataset/*")):
    for img in sorted(glob.glob(folder + "/*.png")):
        im_rgb = Image.open(img)
        im = im_rgb.convert("L")
        data = np.asarray(im).flatten()
        data_rgb = np.asarray(im_rgb).flatten()
        if(folder_cnt < 50):
            Xpca.append(data)
        X.append(data)
        X_rgb.append(data_rgb)
    Y.append(np.genfromtxt(folder + "/rew.csv",delimiter=','))
    folder_cnt += 1
    print("folder_cnt:",folder_cnt)
    sys.stdout.flush()
    # if(folder_cnt == 1):
    #     break

# np.save("saved/Xpca",Xpca)
# np.save("saved/X",X)
np.save("saved/X_rgb",X_rgb)
np.save("saved/Y",Y)

print("Xpca.len:",len(Xpca))
print("X.len:",len(X))
print("Y.len:",len(Y))
print("X[0].shape", X[0].shape)
print("X_rgb[0].shape", X_rgb[0].shape)
print("Y[0].shape", Y[0].shape)
del X_rgb
del Y
sys.stdout.flush()

pca = PCA(n_components=50)
pca.fit(Xpca)

print("pca variance ratio : \n", pca.explained_variance_ratio_)
sys.stdout.flush()

X_reduced = pca.transform(X)

print("X_reduced.shape", X_reduced.shape)
np.save("saved/X_reduced", X_reduced)