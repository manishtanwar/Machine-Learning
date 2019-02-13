import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import sys

def train(x_in,y):
	theta = np.zeros((2,1))
	theta_list = []

	# x_un : unnormalized data
	x_un = np.copy(x_in)
	mean = x_in.mean()
	std_dev = np.std(x_in)
	x_in = (x_in - x_in.mean())/np.std(x_in)
	rate = 0.5
	n = 2
	m = x_in.size

	X = np.ones((m,n))
	X[:,0] = x_in
	y = y[:,np.newaxis]

	iter = 0
	while True:
	    iter+=1
	    theta_list.append(theta[:,0])
	    # print(X.T.shape, y.shape, np.matmul(X.T,y).shape)
	    delta_J = (np.matmul(np.matmul(X.T,X), theta) - np.matmul(X.T,y))
	    # print(delta_J.shape)
	    theta = theta - (rate/m) * delta_J
	    max_change = abs(max(delta_J, key=abs))
	    if(max_change < 1e-3):
	        break
	print(iter)

	y_pred = np.matmul(X,theta)

	theta_req = np.array([theta[0]/std_dev, theta[1] - theta[0] * (mean / std_dev)])
	theta_list = np.array(theta_list)
	# theta_list[:,0], theta_list[:,1] = theta_list[:,0]/std_dev , theta_list[:,1] - theta_list[:,0] * (mean/std_dev)
	
	print(theta_req)

	X_un = np.ones((m,n))
	X_un[:,0] = x_un
	y_pred1 = np.matmul(X_un,theta_req)
	plt.plot(x_un,y,'ro',color='blue')
	plt.plot(x_un,y_pred1,color='red')
	# plt.show()
	plt.pause(1)
	print(theta_list.shape)
	return (theta_list,X,y)

x_in = np.genfromtxt('../ass1_data/linearX.csv',delimiter=',')
y_in = np.genfromtxt('../ass1_data/linearY.csv',delimiter=',')

m = x_in.shape[0]

# X_mat = np.ones((m,2))
# X_mat[:,0] = x_in
# Y_mat = np.zeros((m,1))
# Y_mat[:,0] = y_in

(theta_array, X_mat, Y_mat) = train(x_in, y_in)

np.set_printoptions(precision=3,suppress=True)

def errorFun(theta_0, theta_1):
	theta = np.array([theta_0 ,theta_1])
	theta = theta[:,np.newaxis]
	res = np.matmul((Y_mat - np.matmul(X_mat,theta)).T, (Y_mat - np.matmul(X_mat,theta))) / (2. * m)
	return res

def draw_mesh():
	# print(errorFun(1,1))
	# print("here")
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111,projection = '3d')

	# x = np.arange(-10.0, 10.0, 0.05)
	# y = np.arange(-10.0, 10.0, 0.05)
	x = np.linspace(-1.0, 1.0, 100)
	y = np.linspace(-1.0, 1.0, 100)
	X,Y = np.meshgrid(x, y)

	z_arr = np.array([errorFun(x,y) for x, y in zip(np.ravel(X), np.ravel(Y))])
	Z = z_arr.reshape(X.shape)

	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, color='blue')

	halt_time = 2
	print(theta_array.shape, theta_array.size)
	for i in range(theta_array.shape[0]):
		# ax.remove()
		p = ax.scatter(theta_array[i][0], theta_array[i][1], errorFun(theta_array[i][0], theta_array[i][1]), s = 10)
		ax.set_xlabel('a')
		ax.set_ylabel('b')
		ax.set_zlabel('c')
		ax.view_init(30, 30)
		plt.pause(halt_time)
		plt.draw()
		p.remove()
	# plt.draw()
	plt.show()
	
def draw_contours():
	
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	# x = np.arange(-10.0, 10.0, 0.05)
	# y = np.arange(-10.0, 10.0, 0.05)
	x = np.linspace(-5.0, 5.0, 200)
	y = np.linspace(-4.0, 6.0, 200)
	X,Y = np.meshgrid(x, y)

	z_arr = np.array([errorFun(x,y) for x, y in zip(np.ravel(X), np.ravel(Y))])
	Z = z_arr.reshape(X.shape)

	ax.contour(X, Y, Z)

	halt_time = 2
	print(theta_array.shape, theta_array.size)
	for i in range(theta_array.shape[0]):
		# ax.remove()
		p = ax.scatter(theta_array[i][0], theta_array[i][1], s = 10)
		ax.set_xlabel('a')
		ax.set_ylabel('b')
		plt.pause(halt_time)
		plt.draw()
		p.remove()
	# plt.draw()
	plt.show()

# draw_mesh()
draw_contours()