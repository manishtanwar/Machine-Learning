import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot

def train(x_in,y):
	theta = np.zeros((2,1))
	theta_list = []

	# x_un : unnormalized data
	x_un = np.copy(x_in)
	mean = x_in.mean()
	std_dev = np.std(x_in)
	x_in = (x_in - x_in.mean())/np.std(x_in)
	rate = 1.9
	n = 2
	m = x_in.size

	X = np.ones((m,n))
	X[:,0] = x_in
	y = y[:,np.newaxis]

	iter = 0
	while True:
	    iter+=1
	    theta_list.append(theta)
	    # print(X.T.shape, y.shape, np.matmul(X.T,y).shape)
	    delta_J = (np.matmul(np.matmul(X.T,X), theta) - np.matmul(X.T,y))
	    # print(delta_J.shape)
	    theta = theta - (rate/m) * delta_J
	    max_change = abs(max(delta_J, key=abs))
	    if(max_change < 1e-8):
	        break
	# print(iter)

	y_pred = np.matmul(X,theta)

	theta_req = np.array([theta[0]/std_dev, theta[1] - theta[0] * (mean / std_dev)])
	theta_list = np.array(theta_list)
	theta_list[:,0], theta_list[:,1] = theta_list[:,0]/std_dev , theta_list[:,1] - theta_list[:,0] * (mean/std_dev)
	
	print(theta_req)

	X_un = np.ones((m,n))
	X_un[:,0] = x_un
	y_pred1 = np.matmul(X_un,theta_req)
	# plt.plot(x_un,y,'ro',color='blue')
	# plt.plot(x_un,y_pred1,color='red')
	# plt.show()
	return theta_list

x_in = np.genfromtxt('../ass1_data/linearX.csv',delimiter=',')
y_in = np.genfromtxt('../ass1_data/linearY.csv',delimiter=',')

m = x_in.shape[0]
print(m)
X = np.ones((m,2))
X[:,0] = x_in
Y = np.zeros((m,1))
Y[:,0] = y_in

theta_array = train(x_in, y_in)

np.set_printoptions(precision=3,suppress=True)

def draw_mesh():
	def fun(a, b):
		theta = np.array([a,b])
		theta = theta[:,np.newaxis]
		return np.matmul((Y - np.matmul(X,theta)).T, (Y - np.matmul(X,theta))) / (2. * m)
	print(fun(1,1))
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# x = y = np.arange(-3.0, 3.0, 0.05)
	# X, Y = np.meshgrid(x, y)
	# zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	# Z = zs.reshape(X.shape)

	# ax.plot_surface(X, Y, Z)
	# plt.show()
draw_mesh()