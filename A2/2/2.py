import sys
import numpy as np
import cvxopt
from cvxopt import matrix
import csv
import time
import sklearn.metrics

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
EPS = 1e-5
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
			# if limit == 100:
			# 	break
			# -------------------------

			label = int(row[n])
			if (label == d1 or label == d2):
				y.append(1 if label == d1 else -1)
				xl = []
				for i in range(n):
					xl.append(float(row[i]) / 255.0)
				x.append(xl)
	x = np.array(x)
	y = np.array(y)
	return (x,y)

def gauss_K(x,z):
	norm = np.linalg.norm(x-z)
	return np.exp(-gamma * norm * norm)

def gaussian_kernal(x):
	# return np.asarray([ [ gauss_K(i,j) for j in x ] for i in x ])
	return sklearn.metrics.pairwise.rbf_kernel(x, Y=None, gamma=gamma)

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


# ------------ part_a -----------------------

def part_a():
	# ------------- training: ---------------
	start_time = time.time()

	(x,y) = read_input(train_file)
	y = y[:,np.newaxis]
	global m
	m = y.shape[0]
	(P,q,G,h,A,b) = find_matrices(x,y,C,0)
	cvxopt.solvers.options['show_progress'] = False
	sol = cvxopt.solvers.qp(P,q,G,h,A,b)
	alpha = sol['x']
	sv_cnt = 0
	one_sv = -1
	w = np.zeros((n,1))
	# List of indices of support vectors
	Ind_SV = [index for index, ele in enumerate(alpha) if ele > EPS]
	sv_cnt = len(Ind_SV)

	for i in Ind_SV:
		x_i = x[i].T.reshape((n,1))
		w += alpha[i] * y[i] * x_i
		if (alpha[i] < C - EPS):
			one_sv = i
			
	if(one_sv == -1):
		print("Error!")

	print("# of SV:",sv_cnt)
	b = y[one_sv] - w.T @ x[one_sv]
	# print("W:",w)
	print("b:",b)

	end_time = time.time()
	print("Training time:", end_time - start_time)

	# ------------- testing: ---------------
	correct_pred_cnt = 0
	(x_test,y_test) = read_input(test_file)

	y_pred = np.zeros(y_test.shape)

	for i in range(y_test.shape[0]):
		y_pred[i] = 1 if w.T @ x_test[i] + b > 0 else -1
		if(y_pred[i] == y_test[i]):
			correct_pred_cnt += 1

	print("accuracy:", 100.0 * (correct_pred_cnt / y_test.shape[0]))


# ------------ part_b -----------------------

def part_b():
	# ------------- training: ---------------
	start_time = time.time()

	(x,y) = read_input(train_file)
	y = y[:,np.newaxis]
	global m
	m = y.shape[0]

	(P,q,G,h,A,b) = find_matrices(x,y,C,1)
	cvxopt.solvers.options['show_progress'] = False
	sol = cvxopt.solvers.qp(P,q,G,h,A,b)
	alpha = sol['x']
	alpha = np.array(alpha)
	
	# Saving alpha:
	# alnp = np.array(alpha)
	# np.save("save", alnp)
	
	# Loading alpha:
	# alpha = np.load("save.npy")

	sv_cnt = 0
	one_sv = -1
	print("m:",m)
	# print(alpha)

	# List of indices of support vectors
	Ind_SV = [index for index, ele in enumerate(alpha) if ele > EPS]
	sv_cnt = len(Ind_SV)

	print("# of SV:",sv_cnt)
	# Finding one SV with 0 < alpha < C
	for i in Ind_SV:
		if (alpha[i] < C - EPS):
			one_sv = i
			break

	if(one_sv == -1):
		print("Error!")

	b = float(y[one_sv])
	for i in Ind_SV:
		b -= alpha[i][0] * y[i][0] * gauss_K(x[i], x[one_sv])
	print("b:",b)

	# Finding x,y,alpha for indices of Support Vectors
	xSV = x[Ind_SV,:]
	ySV = y[Ind_SV,:]
	alphaSV = alpha[Ind_SV,:]

	end_time = time.time()
	print("Training Time:", end_time - start_time)

	# ------------- testing: ---------------
	correct_pred_cnt = 0
	(x_test,y_test) = read_input(test_file)

	# using sk-learn gaussian
	MAT = sklearn.metrics.pairwise.rbf_kernel(x_test, Y=xSV, gamma=gamma)
	vec = np.multiply(alphaSV[:,0], ySV[:,0])
	MAT[:,] = np.multiply(MAT[:,], vec)
	Amt = np.sum(MAT, axis = 1) + b
	y_pred = np.where(Amt > 0, 1, -1)
	correct_pred_cnt = sum(y_pred == y_test)

	# without sk-learn gaussian
	# print(alphaSV.shape, ySV.shape)
	# Amt = np.sum(np.asarray([ [ alphaSV[j][0] * ySV[j][0] * gauss_K(xSV[j], x_test[i]) for j in range(xSV.shape[0])] for i in range(x_test.shape[0])]), axis = 1) + b
	# print(Amt.shape)
	# y_pred = np.where(Amt > 0, 1, -1)
	# correct_pred_cnt = sum(y_pred == y_test)

	# print(y_test.shape[0], correct_pred_cnt)
	print("accuracy:", 100.0 * (correct_pred_cnt / y_test.shape[0]))


# ------------ part_c -----------------------
from svmutil import *

def part_c1():
	# ------------- LINEAR ------------------
	# ------------- training: ---------------
	start_time = time.time()

	(x,y) = read_input(train_file)
	global m
	m = y.shape[0]
	prob  = svm_problem(y, x)
	param = svm_parameter('-t 0 -c 1.0 -q')
	model = svm_train(prob, param)
	
	alpha = np.asarray(model.get_sv_coef())
	alpha = alpha[:,0]
	Ind_SV = np.asarray(model.get_sv_indices()) - 1

	sv_cnt = len(Ind_SV)
	print("# of SV:",sv_cnt)

	# -----------finding b and w---------------
	w = np.zeros((n,1))
	j = 0
	for i in Ind_SV:
		x_i = x[i].T.reshape((n,1))
		w += alpha[j] * y[i] * x_i
		if (alpha[j] < C - EPS):
			one_sv = i
		j+=1
			
	if(one_sv == -1):
		print("Error!")

	b = y[one_sv] - w.T @ x[one_sv]
	# print("W:",w)
	print("b:",b)

	end_time = time.time()
	print("Training Time:", end_time - start_time)

	# ------------- testing: ---------------
	(x_test,y_test) = read_input(test_file)
	p_label,_,_ = svm_predict(y_test, x_test, model, '-q')
	ACC,_,_ = evaluations(y_test, p_label)

	print("Accuracy:", ACC)


def part_c2():
	# ------------- GAUSSIAN ------------------
	# ------------- training: ---------------
	start_time = time.time()

	(x,y) = read_input(train_file)
	global m
	m = y.shape[0]
	prob  = svm_problem(y, x)
	param = svm_parameter('-t 2 -c 1.0 -g 0.05 -q')
	model = svm_train(prob, param)

	alpha = np.asarray(model.get_sv_coef())
	alpha = alpha[:,0]
	Ind_SV = np.asarray(model.get_sv_indices()) - 1

	sv_cnt = len(Ind_SV)
	print("# of SV:",sv_cnt)	

	# ------------- finding b: --------------

	for i in range(len(Ind_SV)):
		if (alpha[i] < C - EPS):
			one_sv = i
			break

	b = float(y[one_sv])
	j = 0
	for i in Ind_SV:
		b -= alpha[j] * y[i] * gauss_K(x[i], x[one_sv])
		j += 1
	print("b:",b)

	end_time = time.time()
	print("Training Time:", end_time - start_time)

	# ------------- testing: ---------------
	(x_test,y_test) = read_input(test_file)
	p_label,_,_ = svm_predict(y_test, x_test, model, '-q')
	ACC,_,_ = evaluations(y_test, p_label)

	print("Accuracy:", ACC)

# setting print option to a fixed precision
np.set_printoptions(precision = 6, suppress = True)



if part_num == 'a':
	start_time = time.time()
	part_a()
	end_time = time.time()
	print("Time taken:", end_time - start_time)

elif part_num == 'b':
	start_time = time.time()
	part_b()
	end_time = time.time()
	print("Time taken:", end_time - start_time)
	
elif part_num == 'c':
	print("LibSVM - LINEAR:")
	start_time = time.time()
	part_c1()
	end_time = time.time()
	print("Time taken:", end_time - start_time, "\n")
	
	print("LibSVM - GAUSSIAN:")
	start_time = time.time()
	part_c2()
	end_time = time.time()
	print("Time taken:", end_time - start_time)
	
# ./run.sh 2 <path_of_train_data> <path_of_test_data> <binary_or_multi_class> <part_num> 
# Here, 'binary_or_multi_class' is 0 for binary classification and 1 for multi-class. 
# 'part_num' is part number which can be a-c for binary classification and a-d for multi-class.