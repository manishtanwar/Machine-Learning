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

d1 = 3
d2 = (d1+1)%10
n = 28*28
C = 1.0
m = 0

def read_input():
	# input
	x = []
	y = []
	with open(train_file, 'r') as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
			if (int(row[n]) == d1 or int(row[n]) == d2):
				y.append(float(row[n]))
				xl = []
				for i in range(n):
					xl.append(float(row[i]) / 255)
				x.append(xl)
	x = np.array(x)
	y = np.array(y)
	return (x,y)


def find_matrices(x,y,C):
	A = matrix(y.T, tc='d')
	b = matrix(0.0)
	
	i_ar = np.identity(m)
	G = matrix(np.concatenate((i_ar,-i_ar), axis=0), tc='d')
	
	h_ar = np.zeros(2*m)
	h_ar[0:m] = C
	h = matrix(h_ar, tc='d')
	
	q = matrix(np.ones(m), tc='d')
	P = matrix(np.multiply(x @ x.T, y @ y.T), tc='d')

	return (P,q,G,h,A,b)

def part_a():
	(x,y) = read_input()
	y = y[:,np.newaxis]
	global m
	m = y.shape[0]
	(P,q,G,h,A,b) = find_matrices(x,y,C)
	cvxopt.solvers.options['show_progress'] = False
	sol = cvxopt.solvers.qp(P,q,G,h,A,b)
	alpha = sol['x']
	print(alpha)


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

