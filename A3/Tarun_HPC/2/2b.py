import sys
import numpy as np
import inspect, re
np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(precision = 2, suppress = True)

train_file = sys.argv[1]
test_file = sys.argv[2]

c = 10
layers = [20]
# layers = [50, 100]
batch_size = 1
rate = 2.0

EPS = 1e-6
m = 0
n = 0
l = len(layers)
W = {}
o = {}
b = {}
P = {}
seed = 0

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

def softmax(x):
	y = np.exp(x - np.max(x))
	return y / np.sum(y)

def loss_function(y_pred, y_true):
	log = -np.log(y_pred[y_true[:,0], range(y_pred.shape[1])] + (1e-10))
	return np.sum(log) / y_pred.shape[1]

def train(file):
	input = np.load(file)
	x = input[:,:-1].astype(float)
	(m, n) = x.shape
	y = input[:,-1].astype(int)
	y = y[:,np.newaxis]

	W[0] = np.random.normal(0., 3., (n,layers[0]))
	b[0] = np.random.normal(0., 3., (layers[0],1))
	for i in range(1, l):
		W[i] = np.random.normal(0., 3., (layers[i-1], layers[i]))
		b[i] = np.random.normal(0., 3., (layers[i],1))
	b[l] = np.random.normal(0., 3., (c, 1))
	W[l] = np.random.normal(0., 3., (layers[l-1], c))

	index = np.arange(0,m)
	np.random.seed(seed)
	np.random.shuffle(index)

	def forward_pass(x_batch, y_batch):
		o[0] = sigmoid(W[0].T @ x_batch.T + b[0])
		for i in range(1,l):
			o[i] = sigmoid(W[i].T @ o[i-1] + b[i])

		bb = ((W[l].T @ o[l-1]) + b[l])
		o[l] = np.apply_along_axis(softmax, 0, ((W[l].T @ o[l-1]) + b[l]))
		return (o[l], loss_function(o[l], y_batch))

	def back_propagate(x_batch, y_batch):
		y_hot = np.zeros((c, y_batch.shape[0]))
		y_hot[y_batch[:,0], range(y_batch.shape[0])] = 1
		# print(y_hot)

		P[l] = -(y_hot - o[l])

		for i in range(l-1, -1, -1):
			# P[i] = np.multiply((W[i+1] @ P[i+1]), (np.sum(np.multiply(o[i],1-o[i]), axis=1)[:,np.newaxis]) )
			# print("shape : w[i+1]:",W[i+1].shape,"P[i+1]:",P[i+1].shape, "o[i]:", o[i].shape)
			P[i] = np.multiply((W[i+1] @ P[i+1]), (np.multiply(o[i],1-o[i])))
			
			# ---------- debug ---------------	
			if(i>0):
				assert(P[i].shape == (layers[i],1))
			# --------------------------------

			# W[i+1] = W[i+1] - rate * ((np.sum(o[i], axis=1)[:,np.newaxis]) @ (P[i+1].T))
			W[i+1] = W[i+1] - rate * o[i] @ (P[i+1].T) 
			b[i+1] = b[i+1] - rate * P[i+1]

		W[0] = W[0] - rate * (x_batch.T @ P[0].T)
		# W[0] = W[0] - rate * ((np.sum(x_batch.T, axis=1)[:,np.newaxis]) @ (P[0].T))
		b[0] = b[0] - rate * P[0]

	# print(y)
	iter = 0
	while(True):
		iter += 1
		if(iter == 1000):
			break;
		prev_loss = 0.
		cur_loss = 0.

		start = 0
		end = batch_size

		while(True):
			# print("start,end:",start,end)
			batch = index[start : end]
			x_batch = x[batch]
			y_batch = y[batch]
			
			# ---------- debug -------------
			# x_batch = x_batch[:,0:5]
			# ------------------------------

			(_, loss_here) = forward_pass(x_batch, y_batch)
			cur_loss = max(cur_loss, loss_here)
			back_propagate(x_batch, y_batch)
			
			start += batch_size
			end += batch_size
			
			if(start >= m):
				break;
			end = min(end,m)

		# print("LOSS : ", cur_loss)
		# break;
		# --------- debug -------------
		# sys.stdout.flush()
		
		if(iter%50 == 0):
			print("iter: ",iter)
			(y_pred, _) = forward_pass(x, y)
			# print(y_pred.shape, y.shape)
			y_lb = np.argmax(y_pred, axis=0)
			count = np.zeros(c, dtype=int)
			for i in range(y_lb.shape[0]):
				count[y_lb[i]] += 1
			count[y_lb]+=1
			print(count)
			y1 = y[:,0]
			# print(y_lb.shape, y1.shape)

			print(np.sum(y1 == y_lb), "Loss : ", loss_function(y_pred, y))
			sys.stdout.flush()
		# print(W,b,o)
		# print(y_pred)

		# -----------------------------
		# if(abs(cur_loss-prev_loss) < EPS):
		# 	break;
		prev_loss = cur_loss


train("train_big.npy")
# train("train.npy")
# train("example.npy")