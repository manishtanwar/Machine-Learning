import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import random

w = 160
h = 210
im = Image.open("0.png").crop((3,27,w-3,h-21)).convert("L")
w, h = im.size
print(w,h)
a = np.asarray(im)
print(a.shape, type(a[0][0]), 'a')

im.save("0b.png")