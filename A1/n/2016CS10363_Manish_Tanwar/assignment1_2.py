import numpy as np
import matplotlib.pyplot as plt
import sys

def train(X_in, Y, tau):
    '''
    Input : X_in:   An array of shape (m,n-1) containing the training data provided
            Y:      An array of shape (m,) containing labels {0,1}
            tau:    Bandwidth Parameter
        where
            m : Number of training examples
            n : Dimension of the input data(including the intercept term)
            n = 2 in this problem
    '''
    
    # adding new axis in X_in and Y
    Y = Y[:, np.newaxis]
    X_in = X_in[:, np.newaxis]

    n = X_in.shape[1] + 1
    m = X_in.shape[0]

    # X : An array of shape (m,n) containing input data including the intercept term (X[:,0]=1)
    X = np.ones((m,n))
    X[:,1] = X_in[:,0]

    # Generating x data (1000 points) to draw the curve
    X_line = np.linspace(min(X_in[:,0]), max(X_in[:,0]), 1000)
    Y_pred = np.zeros(X_line.shape)

    for i in range(X_line.shape[0]):
        # W = diag(w(i)) where w(i) = weight of i_th training example
        # Theta = inv(X.T * W * X) * (X.T * W * Y)
        W = np.diag(np.exp(-(np.square(X_in - X_line[i]))/(2.*tau*tau))[:,0])
        theta = np.matmul(np.linalg.inv(np.matmul(X.T, np.matmul(W, X))), np.matmul(X.T, np.matmul(W, Y)))

        # Prediction value of X_line[i]
        Y_pred[i] = theta[1][0]*X_line[i] + theta[0][0]

    # Plotting Curve resulting from the fit 
    plt.plot(X_line, Y_pred, color='blue', label="Predicted Curve")

    plt.legend()
    plt.show()

# input
X = np.genfromtxt(sys.argv[1],delimiter=',')
Y = np.genfromtxt(sys.argv[2],delimiter=',')
tau = float(sys.argv[3])

# setting print option to a fixed precision 
np.set_printoptions(precision = 6, suppress = True)

# Drawing the given data
plt.scatter(X, Y, color='red', label = "Given Data", s = 20)
plt.xlabel("x - axis")
plt.ylabel("y - axis")

train(X, Y, tau)