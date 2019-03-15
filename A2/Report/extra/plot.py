import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
array = [[ 969,    0,    1,    0,    0,    3,    4,    1,    2,    0],
         [   0, 1121,    3,    2,    1,    2,    2,    0,    3,    1],
         [   4,    0, 1000,    4,    2,    0,    1,    6,   15,    0],
         [   0,    0,    8,  985,    0,    4,    0,    6,    5,    2],
         [   0,    0,    4,    0,  962,    0,    6,    0,    2,    8],
         [   2,    0,    3,    6,    1,  866,    7,    1,    5,    1],
         [   6,    3,    0,    0,    4,    4,  939,    0,    2,    0],
         [   1,    4,   19,    2,    4,    0,    0,  987,    2,    9],
         [   4,    0,    3,   10,    1,    5,    3,    3,  942,    3],
         [   4,    4,    3,    8,   13,    4,    0,    9,   12,  952]]

# array = [[14629,  3370,  1212,   566,   392],
#          [ 2947,  2916,  3243,  1322,   410],
#          [ 1430,  1438,  4719,  5979,   965],
#          [ 1111,   549,  2060, 18380,  7258],
#          [ 3089,   255,   448, 14930, 40100]]

# for i in range(5):
#     s = 0
#     for j in range(5):
#         s += array[i][j]
#     print(s)

df_cm = pd.DataFrame(array, index = [i for i in "0123456789"],columns = [i for i in "0123456789"])
# df_cm = pd.DataFrame(array, index = [i for i in "12345"], columns = [i for i in "12345"])

plt.figure(figsize = (10,7))
cmap = sn.cm.rocket_r

ax = sn.heatmap(df_cm, annot=True, cmap = cmap, fmt = 'g')
ax.set(xlabel='Predicted Class', ylabel='Actual Class')
plt.show()
