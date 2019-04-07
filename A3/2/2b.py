import sys
import numpy as np
import inspect, re
import time

global_start_time = time.time()

np.random.seed(0)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision = 4, suppress = True)

train_file = sys.argv[1]
test_file = sys.argv[2]

c = 10
layers = [100,50]
batch_size = 100
rate = 0.1

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

# def sigmoid1(x):
# 	return 1 / (1 + np.exp(-x))

# def softmax_old(x):
# 	y = np.exp(x - np.max(x))
# 	return y / np.sum(y)

# def softmax1(x):
# 	y = np.exp(x - np.amax(x,axis=0))
# 	return y / np.sum(y,axis=0)

def loss_function(y_pred, y_true):
	log = -np.log(y_pred[y_true[:,0], range(y_pred.shape[1])] + (1e-10))
	return np.sum(log) / y_pred.shape[1]

start_time_test_input = time.time()

test_input = np.load("test.npy")
xt = test_input[:,:-1].astype(float)
yt = test_input[:,-1].astype(int)
yt = yt[:,np.newaxis]

end_time_test_input = time.time()

def train(file):
	start_time_init = time.time()

	input = np.load(file)
	x = input[:,:-1].astype(float)
	(m, n) = x.shape
	y = input[:,-1].astype(int)
	y = y[:,np.newaxis]

	# np.random.seed(seed)
	W[0] = np.random.normal(0., 1., (n,layers[0]))
	b[0] = np.random.normal(0., 1., (layers[0],1))
	for i in range(1, l):
		W[i] = np.random.normal(0., 1., (layers[i-1], layers[i]))
		b[i] = np.random.normal(0., 1., (layers[i],1))
	W[l] = np.random.normal(0., 1., (layers[l-1], c))
	b[l] = np.random.normal(0., 1., (c, 1))

	index = np.arange(0,m)
	# np.random.shuffle(index)

	def forward_pass(x_batch, y_batch):
		o[0] = sigmoid(np.matmul(W[0].T, x_batch.T) + b[0])
		for i in range(1,l):
			o[i] = sigmoid(np.matmul(W[i].T, o[i-1]) + b[i])

		# o[l] = np.apply_along_axis(softmax_old, 0, (np.matmul(W[l].T, o[l-1]) + b[l]))
		o[l] = softmax(np.matmul(W[l].T, o[l-1]) + b[l])
		return (o[l], loss_function(o[l], y_batch))

	def back_propagate(x_batch, y_batch):
		m_batch = y_batch.shape[0]
		y_hot = np.zeros((c, y_batch.shape[0]))
		y_hot[y_batch[:,0], range(y_batch.shape[0])] = 1

		P[l] = -(y_hot - o[l])
		# print("P[l] shape : ", P[l].shape)
		for i in range(l-1, -1, -1):
			P[i] = np.multiply(np.matmul(W[i+1], P[i+1]), (np.multiply(o[i],1.0-o[i])))
			# print("P[i] shape : ", P[i].shape)
			
			# ---------- debug ---------------	
			if(i>0):
				assert(P[i].shape == (layers[i],m_batch))
			# --------------------------------

		m_batch = 1
		for i in range(l,0,-1):
			W[i] = W[i] - (rate / m_batch) * np.matmul(o[i-1], (P[i].T))
			tmp = np.ones((1,P[i].shape[1]))
			b[i] = b[i] - (rate / m_batch) * ((np.matmul(tmp, P[i].T)).T)

		W[0] = W[0] - (rate / m_batch) * np.matmul(x_batch.T, P[0].T)
		tmp = np.ones((1,P[0].shape[1]))
		b[0] = b[0] - (rate / m_batch) * ((np.matmul(tmp, P[0].T)).T)

	end_time_init = time.time()

	# print(W)
	# print(b)

	iter = 0
	while(True):
		if(iter == 501):
			break;
		np.random.shuffle(index)
		# print("iter :",iter)
		prev_loss = 0.
		cur_loss = 0.
		for batch_i in range(0, (m+batch_size-1)//batch_size):
			start_time_init_inner = time.time()
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
			end_time_init_inner = time.time()
			
			start_time_forward = time.time()
			(_, loss_here) = forward_pass(x_batch, y_batch)
			end_time_forward = time.time()
			cur_loss = max(cur_loss, loss_here)
			
			# print("LOSS : ", cur_loss)
			start_time_back = time.time()
			back_propagate(x_batch, y_batch)
			end_time_back = time.time()

			# print("Time in inner init", end_time_init_inner - start_time_init_inner)	
			# print("Time in forward", end_time_forward - start_time_forward)	
			# print("Time in backward", end_time_back - start_time_back)	
		
		# print("LOSS : ", cur_loss)
		# break;
		# --------- debug -------------
		# sys.stdout.flush()
		
		if(iter%50 == 0):
			if(iter==0):
				print("True Prediction")
				count = np.zeros(c, dtype=int)
				for i in range(y.shape[0]):
					count[y[i][0]] += 1
				print(count)
			print("iter: ",iter,'\n')
			(y_pred, _) = forward_pass(x, y)
			# print(y_pred.shape, y.shape)
			y_lb = np.argmax(y_pred, axis=0)
			count = np.zeros(c, dtype=int)
			for i in range(y_lb.shape[0]):
				count[y_lb[i]] += 1
			print(count)
			y1 = y[:,0]
			# print(y_lb.shape, y1.shape)

			print("Loss : ", loss_function(y_pred, y))
			print(np.sum(y1 == y_lb))
			print("Train Accuracy:", 100.0 * (np.sum(y1 == y_lb)/y1.shape[0]))

			(y_predt, _) = forward_pass(xt, yt)
			# print(y_pred.shape, y.shape)
			y_lbt = np.argmax(y_predt, axis=0)
			y1t = yt[:,0]
			print("Test Accuracy:", 100.0 * (np.sum(y1t == y_lbt)/y1t.shape[0]))

			sys.stdout.flush()

		# print(W,b,o)
		# print(y_pred)

		# -----------------------------
		if(abs(cur_loss-prev_loss) < EPS):
			print("converged")
			break;
		# print(abs(cur_loss-prev_loss))
		prev_loss = cur_loss
		iter += 1

	# print("Time in init", end_time_init - start_time_init)	


# start_time = time.time()
train("train_big.npy")
# end_time = time.time()
# print("Time taken:", end_time - start_time)

# print("Time in test input", end_time_test_input - start_time_test_input)

# train("train.npy")
# train("example.npy")

global_end_time = time.time()
print("Time taken:", global_end_time - global_start_time)