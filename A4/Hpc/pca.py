import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# im = Image.open(sys.argv[1]).convert("L")
# data = np.asarray(im)
Xpca = []
X = []
Y = []
folder_cnt = 0
for folder in sorted(glob.glob("train_dataset/*")):
    for img in sorted(glob.glob(folder + "/*.png")):
        im = Image.open(img).convert("L")
        data = np.asarray(im).flatten()
        if(folder_cnt < 50):
            Xpca.append(data)
        X.append(data)
    Y.append(np.genfromtxt(folder + "/rew.csv",delimiter=','))
    folder_cnt += 1
    if(folder_cnt == 1):
        break

np.save("Xpca",Xpca)
np.save("X",X)
np.save("Y",Y)
print("Xpca.len:",len(Xpca))
print("X.len:",len(X))
print("Y.len:",len(Y))
print("X[0].shape", X[0].shape)
print("Y[0].shape", Y[0].shape)
# 849281.hn1.hpc.iitd.ac.in

pca = PCA(n_components=50)
pca.fit(Xpca)
print(pca.explained_variance_ratio_)

print(X[0].shape)

xn = pca.transform(X)
print(type(xn), xn[0].shape, xn.shape)
print(X[0].shape)