import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import csv
import time
from numpy import linalg as la
import math
import sklearn.metrics
from sklearn.metrics import f1_score, confusion_matrix 
import sys


train = sys.argv[1]
test = sys.argv[2]


train = "mnist/train.csv"
test = "mnist/test.csv"



X = [[],[],[],[],[],[],[],[],[],[]]
# Y = [[],[],[],[],[],[],[],[],[],[]]
X_train = []
Y_train = []
X_test = []
Y_test = []

num = 5

start1 = time.time()

combs = []
for i in range(10):
	for j in range(i+1,10):
		combs.append((i,j))

with open("mnist/train.csv") as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		temp = []
		for i in range(784):
			temp.append(float(row[i])/255)
		X[int(row[784])].append(temp)
		X_train.append(temp)
		Y_train.append(float(row[784]))
		# Y[float(row[784])].append(temp)
		# Y.append(float(row[784]))
		# X.append(temp)
		# if(float(row[784])==num):
			# Yq1.append(1)
			# Xq1.append(temp)
		# if(float(row[784])==num+1):
			# Yq1.append(-1)
			# Xq1.append(temp)

with open("mnist/test.csv") as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		temp = []
		for i in range(784):
			temp.append(float(row[i])/255)
		Y_test.append(float(row[784]))
		X_test.append(temp)
		# if(float(row[784])==num):
		# 	Yq1_test.append(1)
		# 	Xq1_test.append(temp)
		# if(float(row[784])==num+1):
		# 	Yq1_test.append(-1)
		# 	Xq1_test.append(temp)

end1 = time.time()
print("Input done, Time taken", end1-start1)
start2 = time.time()

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
			
alpha_combs = []
SV_combs = []
b_combs = []


start3 = time.time()
for k in range(len(combs)):
	num1 = combs[k][0]
	num2 = combs[k][1]
	# print(type(X[num1]))
	# print(len(X[num1]))
	# print(len(X[num2]))
	# print(type(X[num2]))
	Xq1 = X[num1] + X[num2]
	Yq1 = ([1]*(len(X[num1]))) + ([-1]*(len(X[num2])))
	# print(len(Xq1))
	# print(len(Yq1))
	# print((Xq1))
	# print(Yq1)

	alpha_count = len(Xq1)
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
	# P1 = guassian_kernel(matXq1, matXq1)
	P1 = sklearn.metrics.pairwise.rbf_kernel(Xq1, Xq1, gamma=0.05)
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

	solvers.options["show_progress"] = False

	solution = solvers.qp(P,q,G,h,A,b)
	# print(solution['status'])
	# print(solution['x'])
	alphas_q1 = np.array(solution['x'])
	# print(solution['primal objective'])
	alpha_combs.append(alphas_q1)

	SV = []
	for i in range(alphas_q1.shape[0]):
		if alphas_q1[i][0]<C and alphas_q1[i][0]>1e-5:
			SV.append(i)

	SV_combs.append(SV)
	
	temp_x_guas = []
	temp_alpha_y = []

	for j in range(len(SV)):
		kernel = guas(Xq1[SV[j]], Xq1[SV[0]])
		temp_x_guas.append(kernel)
		temp_alpha_y.append(alphas_q1[SV[j]]*Yq1[SV[j]])

	temp_row = np.array([temp_x_guas])
	temp_col = np.array([temp_alpha_y]).transpose()
	b_gaus = Yq1[SV[0]] - temp_row.dot(temp_col) 
	# print("b for guassian", b_gaus)
	b_combs.append(b_gaus)

	print("Classifier No: ", k, "Trained")



# print(combs[35])
# print(len(alpha_combs))
# print((alpha_combs[35][0]))
# print(len(SV_combs))
# print(len(SV_combs[35]))
# print(len(b_combs))
# print(b_combs[35])

end3 = time.time()
print("Training, Time taken using CVXOPT", int(end3-start3))

prediction_test = []

# prediction_train = []

# new_confu = confusion_matrix(actual_value, predicted_value)
# for i in range(len(Y_test)):
for i in range(1500):
	score = np.zeros(10)
	for k in range(len(combs)):
		# print("Classifier", k)
		num1 = combs[k][0]
		num2 = combs[k][1]
		Xq1 = X[num1] + X[num2]
		Yq1 = ([1]*(len(X[num1]))) + ([-1]*(len(X[num2])))
		x_guas = []
		alpha_y = []
		for j in range(len(SV_combs[k])):
			kernel = guas(Xq1[SV_combs[k][j]], X_test[i])
			x_guas.append(kernel)
			alpha_y.append(alpha_combs[k][SV_combs[k][j]]*Yq1[SV_combs[k][j]])
		temp_row = np.array([x_guas])
		temp_col = np.array([alpha_y]).transpose()
		pred = temp_row.dot(temp_col) + b_combs[k]
		if(pred>=0):
			winner = num1
		else:
			winner = num2
		score[winner]+=1

	predicted = np.argmax(score)	
	# print("Testing :", i, "Num=", Y_test[i])
	print("Testing-Test Data:", i, "Num=", Y_test[i], "Predicted: ", predicted)
	# print("Scores", score)
	prediction_test.append(predicted)


count_test = 0
# for i in range(len(Y_test)):
for i in range(1500):
	pred =0
	if(prediction_test[i]==Y_test[i]):
		count_test+=1


print("Total correct(test data): ", count_test)
# print("Total no.: ", len(Y_test))
print("Total no.: ", 1500)
print("Accuracy using Guassian Kernel(test data): ", (count_test/1500)*100)

confu_test = confusion_matrix(Y_test[0:1500], prediction_test)
print("Confusion Matrix for test data:")
print(confu_test)

# prediction_train = []


# # for i in range(len(Y_train)):
# for i in range(1500):
# 	score = np.zeros(10)
# 	for k in range(len(combs)):
# 		# print("Classifier", k)
# 		num1 = combs[k][0]
# 		num2 = combs[k][1]
# 		Xq1 = X[num1] + X[num2]
# 		Yq1 = ([1]*(len(X[num1]))) + ([-1]*(len(X[num2])))
# 		x_guas = []
# 		alpha_y = []
# 		for j in range(len(SV_combs[k])):
# 			kernel = guas(Xq1[SV_combs[k][j]], X_train[i])
# 			x_guas.append(kernel)
# 			alpha_y.append(alpha_combs[k][SV_combs[k][j]]*Yq1[SV_combs[k][j]])
# 		temp_row = np.array([x_guas])
# 		temp_col = np.array([alpha_y]).transpose()
# 		pred = temp_row.dot(temp_col) + b_combs[k]
# 		if(pred>=0):
# 			winner = num1
# 		else:
# 			winner = num2
# 		score[winner]+=1

# 	predicted = np.argmax(score)	
# 	# print("Testing :", i, "Num=", Y_test[i])
# 	print("Testing-Train Data :", i, "Num=", Y_train[i], "Predicted: ", predicted)
# 	# print("Scores", score)
# 	prediction_train.append(predicted)


# count_train = 0
# # for i in range(len(Y_train)):
# for i in range(1500):
# 	pred =0
# 	if(prediction_train[i]==Y_train[i]):
# 		count_train+=1


# print("Total correct(train data): ", count_train)
# # print("Total no.: ", len(Y_train))
# print("Total no.: ", 1500)
# print("Accuracy using Guassian Kernel(train data): ", (count_train/1500)*100)

# confu_train = confusion_matrix(Y_train[0:1500], prediction_train)
# print("Confusion Matrix for train data:")
# print(confu_train)