import sys
import numpy as np

train_file = sys.argv[1]
test_file = sys.argv[2]

c = 10
layers = [10,20,30,5]
batch_size = 1

m = 0
n = 0
l = len(layers)
W = {}
b = np.random.rand(l+1)
seed = 0

# start_time = time.time()
# end_time = time.time()
# print("Time taken:", end_time - start_time)

def hot_encode(y):
	y_hot = np.zeros(c)
	y_hot[y] = 1
	return y_hot

def train(file):
	input = np.load(file)
	x = input[:,:-1].astype(float)
	# x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
	(m, n) = x.shape
	y = input[:,-1].astype(int)
	y = y[:,np.newaxis]
	y_hot = np.apply_along_axis(hot_encode,1,y)

	W[0] = np.random.rand(n,layers[0])
	for i in range(1, l):
		W[i] = np.random.rand(layers[i-1], layers[i])
	W[l] = np.random.rand(layers[l-1], c)

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
		

	while(True):
		for batch in range(0, (m+batch_size-1)//batch_size):
			start = batch*batch_size
			# end = start + batch_size
			end = (batch+1)*batch_size

			end = min(end, m)
			batch = index[start : end]

			x_batch = x[batch]
			y_batch = y_hot[batch]
			print("start :",start, "end :",end)
			# print(x_batch, y_batch)

			forward_pass(x_batch, y_batch)

		break;


train("example.npy")