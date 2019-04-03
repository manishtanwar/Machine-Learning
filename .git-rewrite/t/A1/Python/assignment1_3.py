import numpy as np
import matplotlib.pyplot as plt
import sys
import math

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(X_in, Y):
    '''
    Input : X_in: An array of shape (m,n-1) containing the training data provided
            Y: An array of shape (m,) containing labels {0,1}
        where
            m : Number of training examples
            n : Dimension of the input data(including the intercept term)
            n = 3 in this problem
    '''

    n = X_in.shape[1] + 1
    m = X_in.shape[0]

    # X : An array of shape (m,n) containing noramlized data with(intercept term 1)
    X = np.ones((m,n))

    # mean and standerd deviation of each column of X
    mean = np.zeros(n)
    std = np.zeros(n)

    for i in range(n-1):
        mean[i+1] = X_in[:,i].mean()
        std[i+1]  = np.std(X_in[:,i])
        X[:,i+1] = (X_in[:,i] - mean[i+1])/std[i+1]

    # theta : parameter of the model
    theta = np.zeros((n,1))
    Y = Y[:,np.newaxis]

    '''
    theta_(t+1) = theta_(t) - inv(H)*grad(LL(theta_t))

    grad(LL(theta_t)) = X.T * (y - g(X*theta_t))
    let pi(i) = g(x(i)_T * theta_t)
    W = diag(pi(i) * (1-pi(i)))
    H = - X.T * W * X

    H : Hessian matrix of LL(theta)
    grad : Gradient of LL(theta)
    '''

    iter = 0
    # sig : array for storing sigmoid of (X*theta)
    sig = np.zeros((n,1))
    
    # termination condition variable
    EPS = 1e-8

    while(True):
        iter+=1

        # calculate sigmoid of (X*theta)
        sig = sigmoid(np.matmul(X,theta))

        W = np.diag(np.multiply(sig, 1-sig)[:,0])
        grad_LL = np.matmul(X.T, Y-sig)
        H = -np.matmul(X.T, np.matmul(W, X))

        change_amt = -np.matmul(np.linalg.pinv(H), grad_LL)
        
        theta = theta + change_amt

        # max_change in any dimension of theta
        max_change = abs(max(change_amt, key=abs))
        
        # termination condition
        if(max_change < EPS):
            break;

    # theta_req : theta for un-normalized data
    theta_req = np.zeros(n)
    theta_req[2] = theta[2] / std[2]
    theta_req[1] = theta[1] / std[1]
    theta_req[0] = theta[0] - (theta[2]*mean[2])/std[2] - (theta[1]*mean[1])/std[1]
    
    print("Theta parameter:")
    print(theta_req)
    
    # Separator eqn : ((X1_boundary-mean[1])/std[1])*theta[1] + ((X_in[:,2] - mean[2])/std[2])*theta[2] + theta[0] = 0;
    # Determining boundary points
    X2_boundary = (-(theta_req[1]/theta_req[2]) * X_in[:,1]) - (theta_req[0] / theta_req[2])
    
    # Plotting the separator
    plt.plot(X_in[:,1],X2_boundary,color='red')
    plt.show()

# input
X = np.genfromtxt(sys.argv[1],delimiter=',')
Y = np.genfromtxt(sys.argv[2],delimiter=',')

# List of indices with label 0 in Y
Ind_0 = [index for index, ele in enumerate(Y) if ele == 0]

# List of indices with label 1 in Y
Ind_1 = [index for index, ele in enumerate(Y) if ele == 1]

# Ploting input data with provided labels
plt.scatter(X[Ind_0,0], X[Ind_0,1], color = "green", marker = "o", s = 25, label = "y = 0")
plt.scatter(X[Ind_1,0], X[Ind_1,1], color = "blue", marker = "*", s = 30, label = "y = 1")
plt.xlabel("x1 - axis")
plt.ylabel("x2 - axis")
plt.legend()


# setting print option to a fixed precision 
np.set_printoptions(precision = 6, suppress = True)

train(X, Y)