import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

Xval = []
Yval = np.genfromtxt("validation_rewards.csv",delimiter=',',dtype="uint8")

folder_cnt = 0
for folder in sorted(glob.glob("validation_dataset/*")):
    for img in sorted(glob.glob(folder + "/*.png")):
        im_rgb = Image.open(img)
        # im = im_rgb.convert("L")
        data_rgb = np.asarray(im_rgb).flatten()
        # X_rgb.append(data_rgb)

        # im = Image.open(img).convert("L")
        # data = np.asarray(im).flatten()
        Xval.append(data_rgb)
        # break
    folder_cnt += 1
    print("folder_cnt:",folder_cnt)
    sys.stdout.flush()
    if(folder_cnt == 50):
        break

Yval = Yval[0:folder_cnt]
print("Xval.len:",len(Xval))
print("Yval.len:",len(Yval))
print("Xval[0].shape", Xval[0].shape)
np.save("saved_small/X_val", Xval)
np.save("saved_small/Yval", Yval[0:folder_cnt])