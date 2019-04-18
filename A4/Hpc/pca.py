import sys
import glob
import numpy as np
from PIL import Image

# im = Image.open(sys.argv[1]).convert("L")
# data = np.asarray(im)
Xpca = []
Xglo = []

for folder in glob.glob("train_dataset/*"):
    for img in glob.glob(folder):
        im = Image.open(img)
        data = np.asarray(im)

    print(folder)