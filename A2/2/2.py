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
gamma = 0.05

def read_input(file):
	# input
	x = []
	y = []
	with open(file, 'r') as csvfile:
		reader = csv.reader(csvfile)

		limit = 0
		for row in reader:
			limit += 1

			# -------- debug ----------
			# if limit == 1000:
			# 	break
			# -------------------------

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
	x_g = np.zeros((m,m))
	
	for i in range(m):
		for j in range(m):
			x_g[i][j] = np.exp(-gamma * np.linalg.norm(x[i] - x[j]))
	return x_g

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

def part_a():
	# ------------- training: ---------------
	(x,y) = read_input(train_file)
	global m
	m = y.shape[0]
	(P,q,G,h,A,b) = find_matrices(x,y,C,0)
	cvxopt.solvers.options['show_progress'] = False
	sol = cvxopt.solvers.qp(P,q,G,h,A,b)
	alpha = sol['x']
	sc_cnt = 0
	one_sv = -1
	w = np.zeros((n,1))

	# List of indices of support vectors
	Ind_SV = [index for index, ele in enumerate(alpha) if alpha[i] > EPS]
	sc_cnt = len(Ind_SV)

	for i in Ind_SV:
		x_i = x[i].T.reshape((n,1))
		w += alpha[i] * y[i] * x_i
		if (alpha[i] < C - EPS):
			one_sv = i
			
	if(one_sv == -1):
		print("Error!")

	print("# of SV:",sc_cnt)
	b = y[one_sv] - w.T @ x[one_sv]
	# print("W:",w)
	print("b:",b)

	# ------------- testing: ---------------
	correct_pred_cnt = 0
	(x_test,y_test) = read_input(test_file)
	# y_pred = 1 if w.T @ x_test.T + b > 0 else -1

	y_pred = np.zeros(y_test.shape)

	for i in range(y_test.shape[0]):
		y_pred[i][0] = 1 if w.T @ x_test[i] + b > 0 else -1
		if(y_pred[i][0] == y_test[i][0]):
			correct_pred_cnt += 1

	print(y_test.shape[0])
	print("accuracy:", correct_pred_cnt / y_test.shape[0])


def part_b():
	# ------------- training: ---------------
	(x,y) = read_input(train_file)
	global m
	m = y.shape[0]
	(P,q,G,h,A,b) = find_matrices(x,y,C,1)
	cvxopt.solvers.options['show_progress'] = False
	sol = cvxopt.solvers.qp(P,q,G,h,A,b)
	alpha = sol['x']
	sc_cnt = 0
	one_sv = -1

	# List of indices of support vectors
	Ind_SV = [index for index, ele in enumerate(alpha) if alpha[i] > EPS]

	# Finding one SV with 0 < alpha < C
	for i in range(m):
		if (alpha[i] > EPS and alpha[i] < C - EPS):
			one_sv = i
			break

	if(one_sv == -1):
		print("Error!")
	# ------------- testing: ---------------



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

