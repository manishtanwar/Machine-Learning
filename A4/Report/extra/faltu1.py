import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
w = 160
h = 210
# im = Image.open(img).crop((3,27,w-3,h-21)).convert("L")
im_rgb = Image.open("1.png").crop((3,27,w-3,h-21)).convert("L")
im_rgb.save("3.png")
print(im_rgb.size)