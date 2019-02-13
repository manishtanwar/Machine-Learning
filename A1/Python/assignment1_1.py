import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import sys

def train(X_in, Y, learning_rate):
	'''
    Input : X_in:   An array of shape (m,n-1) containing the training data provided
            Y:      An array of shape (m,) containing labels {0,1}
            learning rate : Learning rate parameter for gradient descent
        where
            m : Number of training examples
            n : Dimension of the input data(including the intercept term)
            n = 2 in this problem
    '''

	# mean and standered deviation of the input data x(i)'s
	mean = X_in.mean()
	std_dev = np.std(X_in)

    # adding new axis in X_in and Y
	Y = Y[:,np.newaxis]
	X_in = X_in[:, np.newaxis]

	n = X_in.shape[1] + 1
	m = X_in.shape[0]

	theta = np.zeros((2,1))
	# theta list for drawing plots
	theta_list = []

	# X : An array of shape (m,n) containing normalized data including the intercept term
	X = np.ones((m,n))
	X[:,0] = (X_in[:,0] - mean)/std_dev

	# X_un : An array of shape (m,n) containing original(un-normalized) data including the intercept term
	X_un = np.ones((m,n))
	X_un[:,0] = X_in[:,0]

	# termination condition variable
	EPS = 1e-6
	# number of iterations
	no_iter = 0
	
	while True:
		no_iter += 1
		theta_list.append(theta[:,0])
		# delta_J = (X.T * X * theta - X.T * Y)/ m 
		delta_J = (np.matmul(np.matmul(X.T,X), theta) - np.matmul(X.T,Y)) / m
		theta = theta - (learning_rate * delta_J)
		max_change = abs(max(learning_rate * delta_J, key=abs))
		# termination condition
		if(max_change < EPS):
			break

		# calculateing J(theta)
		J = np.matmul((Y - np.matmul(X,theta)).T, (Y - np.matmul(X,theta))) / (2. * m)
		# divergence condition (if J(theta) is too big implies that J is diverging)
		if(J[0][0] > (1e20)):
			print("Error: J is diverging")
			break

	# theta_req : theta for un-normalized data
	theta_req = np.array([theta[0]/std_dev, theta[1] - theta[0] * (mean / std_dev)])
	
	print("Theta parameter:")
	print(theta_req)

	theta_list = np.array(theta_list)
	# Points for plotting curve 
	y_pred = np.matmul(X_un,theta_req)

	figure1 = plt.figure(1)
	sub_plot = figure1.add_subplot(1,1,1)
	# Drawing the given data
	sub_plot.scatter(X_in, Y, color='blue', label = "Given Data", s = 20)
	plt.xlabel("x-axis")
	plt.ylabel("y-axis")

	# Plot curve
	sub_plot.plot(X_un[:,0], y_pred, color='red', label = "Predicted Curve")
	plt.legend()
	plt.show()
	plt.pause(1)
	return (theta_list,X,Y)

plt.ion()

# input
X = np.genfromtxt(sys.argv[1],delimiter=',')
Y = np.genfromtxt(sys.argv[2],delimiter=',')
learning_rate = float(sys.argv[3])
halt_time = float(sys.argv[4])

# setting print option to a fixed precision
np.set_printoptions(precision = 6, suppress = True)

(theta_array, X_mat, Y_mat) = train(X, Y, learning_rate)
m = X.shape[0]

def errorFun(theta_0, theta_1):
	theta = np.array([theta_0 ,theta_1])
	theta = theta[:,np.newaxis]
	res = np.matmul((Y_mat - np.matmul(X_mat,theta)).T, (Y_mat - np.matmul(X_mat,theta))) / (2. * m)
	return res

def draw_mesh(halt_time):
	# plt.ion()
	fig = plt.figure(2)
	# ax = fig.gca(projection = '3d')
	ax = fig.add_subplot(1,1,1,projection = '3d')

	# x = np.arange(-10.0, 10.0, 0.05)
	# y = np.arange(-10.0, 10.0, 0.05)
	x = np.linspace(-3.0, 3.0, 100)
	y = np.linspace(-3.0, 3.0, 100)
	X,Y = np.meshgrid(x, y)

	z_arr = np.array([errorFun(x,y) for x, y in zip(np.ravel(X), np.ravel(Y))])
	Z = z_arr.reshape(X.shape)

	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, color='blue')

	print(theta_array.shape, theta_array.size)
	for i in range(theta_array.shape[0]):
		ax.scatter(theta_array[i][0], theta_array[i][1], errorFun(theta_array[i][0], theta_array[i][1]), s = 10)
		ax.set_xlabel('theta_0')
		ax.set_ylabel('theta_1')
		ax.set_zlabel('J(theta)')
		ax.view_init(40, 50)
		plt.pause(halt_time)
		plt.draw()
		# p.remove()
	# plt.draw()
	# plt.show()
	
def draw_contours(halt_time):
	# plt.ion()
	fig = plt.figure(3)
	ax = fig.add_subplot(1,1,1)

	# x = np.arange(-10.0, 10.0, 0.05)
	# y = np.arange(-10.0, 10.0, 0.05)
	x = np.linspace(-5.0, 5.0, 200)
	y = np.linspace(-4.0, 6.0, 200)
	X,Y = np.meshgrid(x, y)

	z_arr = np.array([errorFun(x,y) for x, y in zip(np.ravel(X), np.ravel(Y))])
	Z = z_arr.reshape(X.shape)

	ax.contour(X, Y, Z)

	print(theta_array.shape, theta_array.size)
	for i in range(theta_array.shape[0]):
		# ax.remove()
		p = ax.scatter(theta_array[i][0], theta_array[i][1], s = 10)
		ax.set_xlabel('a')
		ax.set_ylabel('b')
		plt.pause(halt_time)
		plt.draw()
		# p.remove()
	# plt.draw()
	plt.show()

draw_mesh(halt_time)
draw_contours(halt_time)

