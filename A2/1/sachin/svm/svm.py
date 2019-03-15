import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import csv
import time
from numpy import linalg as la
import math
from svmutil import *

X = []
Y = []
X_test = []
Y_test = []

Xq1 = []
Yq1 = []
Xq1_test = []
Yq1_test = []

num = 5

start1 = time.time()

combs = []
for i in range(10):
	for j in range(i+1,10):
		combs.append(i,j)

with open("mnist/train.csv") as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		temp = []
		for i in range(784):
			temp.append(float(row[i])/255)
		Y.append(float(row[784]))
		X.append(temp)
		if(float(row[784])==num):
			Yq1.append(1)
			Xq1.append(temp)
		if(float(row[784])==num+1):
			Yq1.append(-1)
			Xq1.append(temp)

with open("mnist/test.csv") as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		temp = []
		for i in range(784):
			temp.append(float(row[i])/255)
		Y_test.append(float(row[784]))
		X_test.append(temp)
		if(float(row[784])==num):
			Yq1_test.append(1)
			Xq1_test.append(temp)
		if(float(row[784])==num+1):
			Yq1_test.append(-1)
			Xq1_test.append(temp)

end1 = time.time()
print("Input done, Time taken", end1-start1)
start2 = time.time()
alpha_count = len(Xq1)

def linear_kernel(x,y):
	return np.inner(x,y)

gamma = 0.05

def guassian_kernel(x,y):
	tempmat = np.zeros((alpha_count, alpha_count))
	for i in range(alpha_count):
		for j in range(i, alpha_count):
			normsq = (la.norm(np.array(Xq1[i])- np.array(Xq1[j])))
			tempmat[i][j] = np.exp((-1)*((normsq**2)*(gamma)))
			tempmat[j][i] = tempmat[i][j]
	return tempmat

def guas(x,z):
	normsq = (la.norm(np.array(x)- np.array(z)))
	return math.exp((-1)*((normsq**2)*(gamma)))
			

# alpha_count = 10
#######################################################
q1 = np.ones((alpha_count, 1))*-1
#######################################################
# P1 = np.zeros((alpha_count, alpha_count))
# for i in range(alpha_count):
# 	for j in range(i, alpha_count):
# 		P1[i][j]=Yq1[i]*Yq1[j]*np.inner(Xq1[i],Xq1[j])
# 		P1[j][i] = P1[i][j]
matXq1 = np.array(Xq1)
matYq1 = np.array([Yq1])
# P1 = linear_kernel(matXq1, matXq1)
P1 = guassian_kernel(matXq1, matXq1)
P1 = P1*((matYq1.transpose()).dot(matYq1))
# print(matY)

# print("inner product done")
#######################################################
A1 = np.zeros((1,alpha_count))
for i in range(alpha_count):
	A1[0][i] = Yq1[i] 

#######################################################
temp1 = np.identity(alpha_count)*(-1)
temp2 = np.identity(alpha_count)
G1 = np.concatenate((temp1, temp2), axis=0)

#######################################################
C = 1.0
temp1 = np.zeros((alpha_count,1))
temp2 = np.ones((alpha_count,1))*C
h1 = np.concatenate((temp1, temp2), axis=0)
#######################################################
# b1 = np.array([[0]])
b1 = np.zeros((1,1))
# print(b1.shape)

P = matrix(P1)
q = matrix(q1)
G = matrix(G1)
h = matrix(h1)
A = matrix(A1)
b = matrix(b1)


end2 = time.time()
print("Matrices Made, Time taken", end2-start2)

solvers.options["show_progress"] = False

start3 = time.time()
solution = solvers.qp(P,q,G,h,A,b)
# print(solution['status'])
# print(solution['x'])
alphas_q1 = np.array(solution['x'])
# print(solution['primal objective'])

end3 = time.time()
print("Solved, Time taken", end3-start3)

# print(len(Xq1))
# print(matXq1.shape)
# print(alphas_q1)
SV = []
for i in range(alphas_q1.shape[0]):
	if alphas_q1[i][0]<C and alphas_q1[i][0]>1e-5:
		SV.append(i)




Wq1 = (matXq1.transpose()).dot(alphas_q1*(matYq1.transpose()))
bq1 = Yq1[SV[0]] - (Wq1.transpose().dot(matXq1[SV[0]])) 
# bq2 = Yq1[SV[100]] - (Wq1.transpose().dot(matXq1[SV[100]])) 
# bq3 = Yq1[SV[200]] - (Wq1.transpose().dot(matXq1[SV[200]])) 
# bq3 = Yq1[3535] - matXq1[3535].dot(Wq1) 


matXq1_test = np.array(Xq1_test)
matYq1_test = np.array([Yq1_test]).transpose()

# minlist = []
# maxlist = []
# for i in range(alpha_count):
# 	if Yq1[i]==1:
# 		minlist.append(Wq1.transpose().dot(matXq1[i]))
# 	elif Yq1[i]==-1:
# 		maxlist.append(Wq1.transpose().dot(matXq1[i]))

# bstar = (max(maxlist)+ min(minlist))*(-0.5)
# print("B's", bstar, bq1) 


# print(Wq1)

# print("Samples of b: ", bq1, bq2, bq3)

predictionYq1 = matXq1_test.dot(Wq1)  + bq1

predictionYq1_gaus = []

# b_gaus = Yq1[SV[0]] - (Wq1.transpose().dot(matXq1[SV[0]]))

temp_x_guas = []
temp_alpha_y = []

for j in range(len(SV)):
	kernel = guas(Xq1[SV[j]], Xq1[SV[0]])
	temp_x_guas.append(kernel)
	temp_alpha_y.append(alphas_q1[SV[j]]*Yq1[SV[j]])

temp_row = np.array([temp_x_guas])
temp_col = np.array([temp_alpha_y]).transpose()
b_gaus = Yq1[SV[0]] - temp_row.dot(temp_col) 
print("b for guassian", b_gaus)


for i in range(len(Yq1_test)):
	x_guas = []
	alpha_y = []
	for j in range(len(SV)):
		kernel = guas(Xq1[SV[j]], Xq1_test[i])
		x_guas.append(kernel)
		alpha_y.append(alphas_q1[SV[j]]*Yq1[SV[j]])
	temp_row = np.array([x_guas])
	temp_col = np.array([alpha_y]).transpose()
	pred = temp_row.dot(temp_col) + b_gaus
	# print("Pred: ", pred)
	# print("minus: ", pred-b_gaus)
	predictionYq1_gaus.append(pred)


# count = 0
# for i in range(len(Yq1_test)):
# 	pred =0
# 	if(predictionYq1[i][0]>=0):
# 		pred = 1
# 	else:
# 		pred = -1
# 	if(pred==Yq1_test[i]):
# 		count+=1

count_gaus = 0
for i in range(len(Yq1_test)):
	pred =0
	if(predictionYq1_gaus[i]>=0):
		pred = 1
	else:
		pred = -1
	if(pred==Yq1_test[i]):
		count_gaus+=1

# print("Total correct: ", count)
# print("Total test: ", len(Yq1_test))
# print("Accuracy: ", count/len(Yq1_test)*100)
# print("No. of Support Vectors: ", len(SV))


print("Total correct: ", count_gaus)
print("Total test: ", len(Yq1_test))
print("Accuracy using Guassian Kernel: ", (count_gaus/len(Yq1_test))*100)
print("No. of Support Vectors: ", len(SV))

x_svm, y_svm = Xq1, Yq1

prob  = svm_problem(y_svm, x_svm)
param = svm_parameter('-t 2 -c 1 -b 0 -g 0.05 -q')
m = svm_train(prob, param, '-q')
p_label, p_acc, p_val = svm_predict(Yq1_test, Xq1_test, m, '-b 0 -q')
# print("Accuracy using LIBSVM: ", p_acc)
ACC, MSE, SCC = evaluations(Yq1_test, p_label)
print("Accuracy using LIBSVM: ", ACC)
alpha_libsvm = m.get_sv_coef()
SV_indices = m.get_sv_indices()
alpha_svm = []
j=0
for i in range(len(Yq1_test)):
	if(i==SV_indices[j]):
		alpha_svm.append(alpha_libsvm[j][0])
		j+=1
	else:
		alpha_svm.append(0)


# print(type(alpha_svm))
# print(type(SV_indices))
# print((alpha_svm))
# print((SV_indices))

