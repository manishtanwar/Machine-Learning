import sys
import numpy as np
import inspect, re
np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(precision = 2, suppress = True)

train_file = sys.argv[1]
test_file = sys.argv[2]

c = 10
layers = [13, 26]
batch_size = 100
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

	W[0] = np.random.normal(0., 1., (n,layers[0]))
	b[0] = np.random.normal(0., 1., (layers[0],1))
	for i in range(1, l):
		W[i] = np.random.normal(0., 1., (layers[i-1], layers[i]))
		b[i] = np.random.normal(0., 1., (layers[i],1))
	b[l] = np.random.normal(0., 1., (c, 1))
	W[l] = np.random.normal(0., 1., (layers[l-1], c))

	index = np.arange(0,m)
	np.random.seed(seed)
	np.random.shuffle(index)

	def forward_pass(x_batch, y_batch):
		o[0] = sigmoid(W[0].T @ x_batch.T + b[0])
		for i in range(1,l):
			o[i] = sigmoid(W[i].T @ o[i-1] + b[i])

		o[l] = np.apply_along_axis(softmax, 0, ((W[l].T @ o[l-1]) + b[l]))
		return (o[l], loss_function(o[l], y_batch))

	def back_propagate(x_batch, y_batch):
		m_batch = y_batch.shape[0]
		y_hot = np.zeros((c, y_batch.shape[0]))
		y_hot[y_batch[:,0], range(y_batch.shape[0])] = 1

		P[l] = -(y_hot - o[l])
		# print("P[l] shape : ", P[l].shape)
		for i in range(l-1, -1, -1):
			P[i] = np.multiply((W[i+1] @ P[i+1]), (np.multiply(o[i],1.0-o[i])))
			# print("P[i] shape : ", P[i].shape)
			
			# ---------- debug ---------------	
			if(i>0):
				assert(P[i].shape == (layers[i],m_batch))
			# --------------------------------

		for i in range(l,0,-1):
			W[i] = W[i] - (rate / m_batch) * o[i-1] @ (P[i].T) 
			tmp = np.ones((1,P[i].shape[1]))
			b[i] = b[i] - (rate / m_batch) * ((tmp @ P[i].T).T)

		W[0] = W[0] - (rate / m_batch) * (x_batch.T @ P[0].T)
		tmp = np.ones((1,P[0].shape[1]))
		b[0] = b[0] - (rate / m_batch) * ((tmp @ P[0].T).T)

	# print(y)
	iter = 0
	while(True):
		iter += 1
		if(iter == 500):
			break;
		prev_loss = 0.
		cur_loss = 0.
		for batch_i in range(0, (m+batch_size-1)//batch_size):
			start = batch_i*batch_size
			# end = start + batch_size
			end = (batch_i+1)*batch_size

			end = min(end, m)
			batch = index[start : end]

			x_batch = x[batch]
			
			# ---------- debug -------------
			# x_batch = x_batch[:,0:5]
			# ------------------------------

			y_batch = y[batch]
			# print(y.dtype, y_batch.dtype)
			# print("start :",start, "end :",end)
			# print(x_batch.shape, y_batch)

			(_, loss_here) = forward_pass(x_batch, y_batch)
			cur_loss = max(cur_loss, loss_here)
			# print("LOSS : ", cur_loss)
			back_propagate(x_batch, y_batch)
		
		# print("LOSS : ", cur_loss)
		# break;
		# --------- debug -------------
		# sys.stdout.flush()
		
		if(iter%50 == 0):
			(y_pred, _) = forward_pass(x, y)
			# print(y_pred.shape, y.shape)
			y_lb = np.argmax(y_pred, axis=0)
			count = np.zeros(c, dtype=int)
			for i in range(y_lb.shape[0]):
				count[y_lb[i]] += 1
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