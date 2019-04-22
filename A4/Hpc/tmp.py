import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import random

im = Image.open("image.png")
data = np.asarray(im)
im.save("original.png")
w, h = im.size
print(im.size)
crop = im.crop((3,27,w-3,h))
print(crop.size)
crop.save("crop.png")