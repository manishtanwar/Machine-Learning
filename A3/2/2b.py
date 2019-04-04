import sys
import numpy as np
import inspect, re
np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(precision = 2, suppress = True)

train_file = sys.argv[1]
test_file = sys.argv[2]

c = 10
layers = [50, 100]
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

# start_time = time.time()
# end_time = time.time()
# print("Time taken:", end_time - start_time)

# def hot_encode(y):
# 	y_hot = np.zeros(c, dtype=np.int)
# 	y_hot[y] = 1
# 	return y_hot

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

def softmax1(x): # not stable
	y = np.exp(x)
	# print(x)
	return y / np.sum(y)

def softmax(x):
	y = np.exp(x - np.max(x))
	# print(x)
	return y / np.sum(y)

def loss_function(y_pred, y_true):
	# print(y_pred.shape, y_true.shape)
	# print(y_pred.T)
	log = -np.log(y_pred[y_true[:,0], range(y_pred.shape[1])] + (1e-10))
	# print(log)
	# print("shape : ", log.shape)
	return np.sum(log) / y_pred.shape[1]

def train(file):
	input = np.load(file)
	x = input[:,:-1].astype(float)
	# x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
	(m, n) = x.shape
	# print(x.shape)
	y = input[:,-1].astype(int)
	y = y[:,np.newaxis]
	# y_hot = np.apply_along_axis(hot_encode,1,y)

	# W[0] = np.random.rand(n,layers[0])
	# b[0] = np.random.rand(layers[0],1)
	# # P[0] = np.zeros((layers[0],1))
	# for i in range(1, l):
	# 	W[i] = np.random.rand(layers[i-1], layers[i])
	# 	b[i] = np.random.rand(layers[i],1)
	# 	# P[i] = np.zeros((layers[i],1))
	# b[l] = np.random.rand(c, 1)
	# P[l] = np.zeros((c, 1))
	# W[l] = np.random.rand(layers[l-1], c)

	#----------- debug ------------
	# n = 5
	#----------- debug ------------

	W[0] = np.random.normal(0., 1., (n,layers[0]))
	b[0] = np.random.normal(0., 1., (layers[0],1))
	# P[0] = np.zeros((layers[0],1))
	for i in range(1, l):
		W[i] = np.random.normal(0., 1., (layers[i-1], layers[i]))
		b[i] = np.random.normal(0., 1., (layers[i],1))
		# P[i] = np.zeros((layers[i],1))
	b[l] = np.random.normal(0., 1., (c, 1))
	P[l] = np.zeros((c, 1))
	W[l] = np.random.normal(0., 1., (layers[l-1], c))

	# for i in range(l+1):
	# 	print(W[i].shape)
	# print(b)
	# print(W)

	index = np.arange(0,m)
	np.random.seed(seed)
	np.random.shuffle(index)
	# print(x,y_hot)
	# print(index)

	def forward_pass(x_batch, y_batch):
		# print("HERE : ", W[0].shape, x_batch.shape, b[0].shape)
		# print(x_batch, W[0].T, b[0])

		o[0] = sigmoid(W[0].T @ x_batch.T + b[0])
		# print("o[0].T:\n", o[0].T)
		for i in range(1,l):
			# print("There : ", W[i].shape, o[i-1].shape, b[i].shape)
			o[i] = sigmoid(W[i].T @ o[i-1] + b[i])
			# print("o[i].T:\n", o[i].T)
		# print("l-1:")
		# print(o[l-1].T)
		# print(W[l].T.shape, o[l-1].shape, b[l].shape)
		# print((W[l].T @ o[l-1] + b[l]).shape)

		bb = ((W[l].T @ o[l-1]) + b[l])
		o[l] = np.apply_along_axis(softmax, 0, ((W[l].T @ o[l-1]) + b[l]))
		# print("l:")
		# print((W[l].T @ o[l-1] + b[l]).T)
		# print("y[l].T:\n", bb.T)
		# print("o[l].T:\n", o[l].T)
		# print("\n\n\n\n\n\n\n\n\n\n")
		return (o[l], loss_function(o[l], y_batch))

	def back_propagate(x_batch, y_batch):
		y_hot = np.zeros((c, y_batch.shape[0]))
		y_hot[y_batch[:,0], range(y_batch.shape[0])] = 1
		# print(y_hot)

		P[l] = -(np.sum(y_hot, axis=1) - np.sum(o[l], axis=1)) / y_batch.shape[0]
		P[l] = P[l][:,np.newaxis]

		for i in range(l-1, -1, -1):
			# P[i] = np.multiply((W[i+1] @ P[i+1]), (np.sum(np.multiply(o[i],1-o[i]), axis=1)[:,np.newaxis]) )
			P[i] = np.multiply((W[i+1] @ P[i+1]), (np.sum(np.multiply(o[i],1-o[i]), axis=1)[:,np.newaxis]) ) / y_batch.shape[0]
			
			# ---------- debug ---------------
			if(i>0):
				assert(P[i].shape == (layers[i],1))
			# --------------------------------

			# print(P[i+1].T.shape, P[i+1].shape, o[i].shape)
			# print(W[i+1][0])
			# W[i+1] = W[i+1] - rate * ((np.sum(o[i], axis=1)[:,np.newaxis]) @ (P[i+1].T))
			W[i+1] = W[i+1] - rate * ((np.sum(o[i], axis=1)[:,np.newaxis]) @ (P[i+1].T) / y_batch.shape[0])
			# print(W[i+1][0])
			b[i+1] = b[i+1] - rate * P[i+1]
			# print(W[i+1].shape,P[i].shape,b[i+1].shape)

		# print(P[0].T.shape, x_batch.shape)
		W[0] = W[0] - rate * ((np.sum(x_batch.T, axis=1)[:,np.newaxis]) @ (P[0].T) / y_batch.shape[0])
		# W[0] = W[0] - rate * ((np.sum(x_batch.T, axis=1)[:,np.newaxis]) @ (P[0].T))
		b[0] = b[0] - rate * P[0]

	# print(y)
	iter = 0
	while(True):
		iter += 1
		# if(iter == 2):
		# 	break;
		prev_loss = 0.
		cur_loss = 0.
		for batch in range(0, (m+batch_size-1)//batch_size):
			start = batch*batch_size
			# end = start + batch_size
			end = (batch+1)*batch_size

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
		
		print("LOSS : ", cur_loss)
		# break;
		# --------- debug -------------
		# sys.stdout.flush()
		
		# if(iter%50 == 0):
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