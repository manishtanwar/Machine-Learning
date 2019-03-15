import numpy as np
from numpy import linalg as la
# export PYTHONPATH="/home/sachin/libsvm-3.23/python:${PYTHONPATH}"



a = np.array([[1,2],[3,4],[5,6]])

b = np.array([[1],[2]])

x = np.array([1,2,3])
y = ([1,2,3])

print(la.norm(y))




y=x
y[0] = 10
print(y)
print(x)



combs = []
for i in range(10):
	for j in range(i+1,10):
		combs.append((i,j))
print(len(combs))


# lenas = [1]*5
# print(lenas)


x = [1,2,3]
y = [1,2,3]


print(np.argmax(x))