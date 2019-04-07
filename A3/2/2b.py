import sys
import numpy as np
import inspect, re
import time
import sklearn.metrics

global_start_time = time.time()
np.random.seed(0)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision = 4, suppress = True)

train_file = sys.argv[1]
test_file = sys.argv[2]

c = 10
layers = [100, 50]
batch_size = 100
rate = 0.15
activation_fn = 'relu'
# activation_fn = 'sigmoid'

state = "DEBUG"
# state = "NOT_DEBUG"

EPS = 1e-5
m = 0
n = 0
l = len(layers)
W = {}
o = {}
b = {}
P = {}
seed = 0

def sigmoid(x):
	return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1+np.exp(x)))

def softmax(x):
	yy = np.exp(x - np.max(x, axis=0, keepdims = True))
	return yy / np.sum(yy, axis = 0, keepdims = True)

def relu(x):
	return np.maximum(np.zeros(x.shape),x)

def gradient(x):
	if(activation_fn == 'relu'):
		return np.where(x > (1e-8), 1, 0)
	else:
		return np.multiply(x,1.0-x)

def activation(x):
	if(activation_fn == 'relu'):
		return relu(x)
	else:
		return sigmoid(x)

def loss_function(y_pred, y_true):
	log = -np.log(y_pred[y_true[:,0], range(y_pred.shape[1])] + (1e-10))
	return np.sum(log) / y_pred.shape[1]

def forward_pass(x_batch, y_batch):
	o[0] = activation(np.matmul(W[0].T, x_batch.T) + b[0])
	for i in range(1,l):
		o[i] = activation(np.matmul(W[i].T, o[i-1]) + b[i])

	o[l] = softmax(np.matmul(W[l].T, o[l-1]) + b[l])
	return (o[l], loss_function(o[l], y_batch))

test_input = np.load("test.npy")
x_test = test_input[:,:-1].astype(float)
y_test = test_input[:,-1].astype(int)
y_test = y_test[:,np.newaxis]

def train(file):
	input = np.load(file)
	x = input[:,:-1].astype(float)
	(m, n) = x.shape
	y = input[:,-1].astype(int)
	y = y[:,np.newaxis]

	# np.random.seed(seed)
	sigma = 1.0
	W[0] = np.random.normal(0., sigma, (n,layers[0]))
	b[0] = np.random.normal(0., sigma, (layers[0],1))
	for i in range(1, l):
		W[i] = np.random.normal(0., sigma, (layers[i-1], layers[i]))
		b[i] = np.random.normal(0., sigma, (layers[i],1))
	W[l] = np.random.normal(0., sigma, (layers[l-1], c))
	b[l] = np.random.normal(0., sigma, (c, 1))

	index = np.arange(0,m)

	def back_propagate(x_batch, y_batch):
		m_batch = y_batch.shape[0]
		y_hot = np.zeros((c, y_batch.shape[0]))
		y_hot[y_batch[:,0], range(y_batch.shape[0])] = 1

		P[l] = -(y_hot - o[l])
		for i in range(l-1, -1, -1):
			P[i] = np.multiply(np.matmul(W[i+1], P[i+1]), (gradient(o[i])))

		# m_batch = 1
		for i in range(l,0,-1):
			W[i] = W[i] - (rate / m_batch) * np.matmul(o[i-1], (P[i].T))
			tmp = np.ones((1,P[i].shape[1]))
			b[i] = b[i] - (rate / m_batch) * ((np.matmul(tmp, P[i].T)).T)

		W[0] = W[0] - (rate / m_batch) * np.matmul(x_batch.T, P[0].T)
		tmp = np.ones((1,P[0].shape[1]))
		b[0] = b[0] - (rate / m_batch) * ((np.matmul(tmp, P[0].T)).T)

	iter = 0
	while(True):
		if(iter == 1000):
			break;
		np.random.shuffle(index)
		# print("iter :",iter)
		prev_loss = 0.
		cur_loss = 0.
		for batch_i in range(0, (m+batch_size-1)//batch_size):
			start = batch_i*batch_size
			end = (batch_i+1)*batch_size

			end = min(end, m)
			batch = index[start : end]

			x_batch = x[batch]
			y_batch = y[batch]
			(_, loss_here) = forward_pass(x_batch, y_batch)
			cur_loss = max(cur_loss, loss_here)
			
			back_propagate(x_batch, y_batch)

		if(iter%50 == 0):
			print("iteration no:",iter)
			if(state == 'DEBUG'):
				# Train Data
				(y_pred_train, _) = forward_pass(x, y)
				y_lb_train = np.argmax(y_pred_train, axis=0)
				print("Train Accuracy:", 100.0 * (np.sum(y[:,0] == y_lb_train)/y_lb_train.shape[0]))

				# Test Data
				(y_pred_test, _) = forward_pass(x_test, y_test)
				y_lb_test = np.argmax(y_pred_test, axis=0)
				print("Test Accuracy:", 100.0 * (np.sum(y_test[:,0] == y_lb_test)/y_lb_test.shape[0]))
				print()
			sys.stdout.flush()
		if(abs(cur_loss-prev_loss) < EPS):
			# print("converged")
			break;
		# print(abs(cur_loss-prev_loss))
		prev_loss = cur_loss
		iter += 1

train_start_time = time.time()
train("train_big.npy")
global_end_time = time.time()

print("Training Time:", global_end_time - train_start_time)

(y_pred_test, _) = forward_pass(x_test, y_test)
y_lb_test = np.argmax(y_pred_test, axis=0)
# print(y_lb_test.shape, y_test.shape)

confusion_mat = sklearn.metrics.confusion_matrix(y_test[:,0], y_lb_test)
print("Confusion Matrix:")
print(confusion_mat)
print("Test Accuracy:", 100.0 * (np.sum(y_test[:,0] == y_lb_test)/y_lb_test.shape[0]))