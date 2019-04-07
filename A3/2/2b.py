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

if(train_file[-4:] != '.npy'):
	train_file += '.npy'
if(test_file[-4:] != '.npy'):
	test_file += '.npy'

config_file = open(sys.argv[3],"r")
config_lines = config_file.readlines()

n = int(config_lines[0])
c = int(config_lines[1])
batch_size = int(config_lines[2])
l = int(config_lines[3])
layers = [int(x) for x in config_lines[4].split()]   
activation_fn = config_lines[5].split()[0]
Learning_Rate_type = config_lines[6].split()[0]

# print(n, c, batch_size, l, layers, activation_fn, Learning_Rate_type)

rate = 0.1
Tolerance = 1e-4

state = "DEBUG"
# state = "NOT_DEBUG"

EPS = 1e-7
m = 0

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

test_input = np.load(test_file)
x_test = test_input[:,:-1].astype(float)
y_test = test_input[:,-1].astype(int)
y_test = y_test[:,np.newaxis]

def train(file):
	global rate
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
	prev_loss = 0.
	global_prev_loss = 0.
	global_cur_loss = 0.
	failed_decreasing_cnt = int(0)

	while(True):
		# if(iter == 1000):
		# 	break;
		np.random.shuffle(index)
		# print("iter :",iter)
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

		(y_pred_train, global_cur_loss) = forward_pass(x, y)
		
		
		if(iter%50 == 0):
			if(state == 'DEBUG'):
				print("iteration no:", iter)
				# print("loss:", cur_loss)
				# print("loss difference:", abs(cur_loss-prev_loss))
				print("loss:", global_cur_loss)
				print("loss difference:", abs(global_cur_loss - global_prev_loss))
				# Train Data
				y_lb_train = np.argmax(y_pred_train, axis=0)
				print("Train Accuracy:", 100.0 * (np.sum(y[:,0] == y_lb_train)/y_lb_train.shape[0]))

				# Test Data
				(y_pred_test, _) = forward_pass(x_test, y_test)
				y_lb_test = np.argmax(y_pred_test, axis=0)
				print("Test Accuracy:", 100.0 * (np.sum(y_test[:,0] == y_lb_test)/y_lb_test.shape[0]))
				print()
			sys.stdout.flush()
		# if(abs(cur_loss-prev_loss) < EPS):
		# 	print("converged")
		# 	print("cur_loss:",cur_loss)
		# 	print("prev_loss:",prev_loss)
		# 	print("loss difference:", abs(cur_loss-prev_loss))
		# 	break;

		if(abs(global_cur_loss-global_prev_loss) < EPS):
			print("converged")
			print("cur_loss:",global_cur_loss)
			print("prev_loss:",global_prev_loss)
			print("loss difference:", abs(global_cur_loss-global_prev_loss))
			break;

		if(Learning_Rate_type == 'variable'):
			if(global_prev_loss - global_cur_loss < Tolerance):
				failed_decreasing_cnt += 1
				if(failed_decreasing_cnt == 2):
					rate /= 5
					failed_decreasing_cnt = 0
			else:
				failed_decreasing_cnt = 0
		
		global_prev_loss = global_cur_loss
		prev_loss = cur_loss
		iter += 1

	# Training Data
	(y_pred_train,_) = forward_pass(x, y)
	y_lb_train = np.argmax(y_pred_train, axis=0)
	print("Train Accuracy:", 100.0 * (np.sum(y[:,0] == y_lb_train)/y_lb_train.shape[0]))

train_start_time = time.time()
train(train_file)
global_end_time = time.time()

print("Training Time:", global_end_time - train_start_time)

# Testing Data
(y_pred_test, _) = forward_pass(x_test, y_test)
y_lb_test = np.argmax(y_pred_test, axis=0)
confusion_mat = sklearn.metrics.confusion_matrix(y_test[:,0], y_lb_test)
print("Test Accuracy:", 100.0 * (np.sum(y_test[:,0] == y_lb_test)/y_lb_test.shape[0]))
print("Confusion Matrix:")
print(confusion_mat)