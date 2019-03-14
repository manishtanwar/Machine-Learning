import sys
import numpy as np
import cvxopt
from cvxopt import matrix
import csv
import time

train_file = sys.argv[1]
test_file = sys.argv[2]
binary_multi = sys.argv[3]
part_num = sys.argv[4]

# Constants
d1 = 3
d2 = (d1+1)%10
n = 28*28
C = 1.0
m = 0
EPS = 1e-8

def read_input(file):
	# input
	x = []
	y = []
	with open(file, 'r') as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
			label = int(row[n])
			if (label == d1 or label == d2):
				y.append(1 if label == d1 else -1)
				xl = []
				for i in range(n):
					xl.append(int(row[i]) / 255)
				x.append(xl)
	x = np.array(x)
	y = np.array(y)
	y = y[:,np.newaxis]
	return (x,y)

def gaussian_kernal(x):
	print("due")

def find_matrices(x,y,C,part_num):
	# print(x.shape, y.shape, C)
	A = matrix(y.T, tc='d')
	b = matrix(0.0)
	
	i_ar = np.identity(m)
	G = matrix(np.concatenate((i_ar,-i_ar), axis=0), tc='d')
	
	h_ar = np.zeros(2*m)
	h_ar[0:m] = C
	h = matrix(h_ar, tc='d')
	
	q = matrix(-np.ones(m), tc='d')
	
	if part_num == 0:
		P = matrix(np.multiply(x @ x.T, y @ y.T), tc='d')
	else:
		P = matrix(np.multiply(gaussian_kernal(x), y @ y.T), tc='d')
	
	# print(P.size, q.size, G.size, h.size, A.size, b.size)
	return (P,q,G,h,A,b)

def train(part_no,x,y):
	global m
	m = y.shape[0]
	(P,q,G,h,A,b) = find_matrices(x,y,C,part_no)
	cvxopt.solvers.options['show_progress'] = False
	sol = cvxopt.solvers.qp(P,q,G,h,A,b)
	alpha = sol['x']
	no_sv = 0
	one_sv = -1
	w = np.zeros((n,1))

	for i in range(m):
		if (alpha[i] > EPS):
			no_sv += 1
			one_sv = i
			x_i = x[i].T.reshape((n,1))
			w += alpha[i] * y[i] * x_i

	# print("# of SV:",no_sv)
	b = y[one_sv] - w.T @ x[one_sv]
	# print("W:",w)
	print("b:",b)

def test(x,y):
	print("Hey!")

def part_a():
	(x,y) = read_input(train_file)
	train(0,x,y)
	test(x,y)

def part_b():
	train(1)
	test()

# setting print option to a fixed precision
np.set_printoptions(precision = 6, suppress = True)

start_time = time.time()

if part_num == 'a':
	part_a()
elif part_num == 'b':
	part_b()
elif part_num == 'c':
	part_c()

end_time = time.time()
print("Time taken:", end_time - start_time)

# ./run.sh 2 <path_of_train_data> <path_of_test_data> <binary_or_multi_class> <part_num> 
# Here, 'binary_or_multi_class' is 0 for binary classification and 1 for multi-class. 
# 'part_num' is part number which can be a-c for binary classification and a-d for multi-class.

