import numpy as np
import matplotlib.pyplot as plt
import sys

def train_linear(X, Y):
    '''
    Input : X: An array of shape (m,n) containing the training data provided
            m : Number of training examples
            n : Dimension of the input data
            n = 2 in this problem

            Y: An array of shape (m,) containing labels {0,1}
    '''
    
    m,n = X.shape

    # one_cnt : number of data points with label 1
    one_cnt = np.sum(Y)

    # phi : Bernoulli Parameter corresponding to label 1
    phi = one_cnt / m
    
    # Adding new axis for converting Y's shape : (m,) to (m,1)
    Y = Y[:,np.newaxis]

    # m0 : mean of x(i)'s corresponding to y = 0 labels
    # m1 : mean of x(i)'s corresponding to y = 1 labels
    # sig : co-variance matrix
    m0 = np.zeros((1,n))
    m1 = np.zeros((1,n))
    
    m0 = np.matmul((1-Y).T, X) / (m - one_cnt)
    m1 = np.matmul(Y.T, X) / one_cnt

    # sig = (W.T * W)/m
    # where W = X - m_y(i) = X - Y*m1 - (1-Y)*m0
    W = X - np.matmul((1-Y), m0) - np.matmul(Y, m1)
    
    sig = 1/m * np.matmul(W.T, W)
    sig_inv = np.linalg.inv(sig)
    
    # coeff and intercepts for linear separator equations
    coeff = 2 * (m1-m0).dot(sig_inv)
    intercept = np.matmul(m1, np.matmul(sig_inv,m1.T)) - np.matmul(m0, np.matmul(sig_inv,m0.T))
    intercept += 2 * np.log((1-phi)/phi)

    # Determining Boundary Points
    X1_boundary = (-(coeff[0][0] * (X[:,0])) + intercept[0][0]) / coeff[0][1]
    
    print("mean0:")
    print(m0)
    print("mean1:")
    print(m1)
    print("sigma:")
    print(sig)

    # Ploting linear separator
    plt.plot(X[:,0],X1_boundary,color='red')
    plt.show()

def train_quadratic(X, Y):
    '''
    Input : X: An array of shape (m,n) containing the training data provided
            m : Number of training examples
            n : Dimension of the input data
            n = 2 in this problem

            Y: An array of shape (m,) containing labels {0,1}
    '''
    
    m,n = X.shape

    # one_cnt : number of data points with label 1
    one_cnt = np.sum(Y)

    # phi : Bernoulli Parameter corresponding to label 1
    phi = one_cnt / m
    
    # Adding new axis for converting Y's shape : (m,) to (m,1)
    Y = Y[:,np.newaxis]

    # m0 : mean of x(i)'s corresponding to y = 0 labels
    # m1 : mean of x(i)'s corresponding to y = 1 labels
    # sig : co-variance matrix
    m0 = np.zeros((1,n))
    m1 = np.zeros((1,n))
    
    m0 = np.matmul((1-Y).T, X) / (m - one_cnt)
    m1 = np.matmul(Y.T, X) / one_cnt

    M0 = np.zeros(X.shape)
    M1 = np.zeros(X.shape)
    M0[:,] = m0
    M1[:,] = m1

    # computing sigma_0
    # sigma_0 = (X-M0).T * diag(1-Y) * (X-M0)
    W = X - M0
    D = np.diag(1-Y[:,0])
    sigma_0 = (W.T @ D @ W) / (m-one_cnt)
    
    # computing sigma_1
    # sigma_1 = (X-M1).T * diag(Y) * (X-M1)
    W = X - M1
    D = np.diag(Y[:,0])
    sigma_1 = (W.T @ D @ W) / one_cnt
    
    inv_sigma_0 = np.linalg.inv(sigma_0)
    inv_sigma_1 = np.linalg.inv(sigma_1)
    
    # function to compute value of quadratic equation at (x,y) 
    def quad_eqn(x,y):
        X = np.array([x,y])
        res = 0.
        for i in range(n):
            for j in range(n):
                res += (X[i]-m1[0][i]) * (X[j]-m1[0][j]) * inv_sigma_1[i][j]
                res -= (X[i]-m0[0][i]) * (X[j]-m0[0][j]) * inv_sigma_0[i][j]
        c = (1-phi)/phi
        c = c*c
        res += np.log((np.linalg.det(sigma_1)/np.linalg.det(sigma_0)) * c)
        return res;
    
    def draw_quadratic_separator():
        # generate x and y for drawing the quadratice separator
        x = np.linspace(-10, 250, 1000)
        y = np.linspace(100, 800, 1000)
        x, y = np.meshgrid(x, y)

        # drawing quadratic separator by setting eqaution = 0
        plt.contour(x, y, quad_eqn(x,y), [0], colors = "red")
        plt.show()

    print("mean0:")
    print(m0)
    print("mean1:")
    print(m1)
    print("sigma0:")
    print(sigma_0)
    print("sigma1:")
    print(sigma_1)

    draw_quadratic_separator()


# setting print option to a fixed precision 
np.set_printoptions(precision=3,suppress=True)

# Input
X = np.genfromtxt(sys.argv[1])
Y_label = np.genfromtxt(sys.argv[2],delimiter=",",dtype=str)


# Convert lables to {0,1}
# Alaska : '0'
# Canada : '1'
Y = np.zeros(Y_label.shape[0])
for i in range(Y_label.shape[0]):
    Y[i] = 0 if Y_label[i] == "Alaska" else 1


# List of indices with label 0 in Y
Ind_0 = [index for index, ele in enumerate(Y) if ele == 0]

# List of indices with label 1 in Y
Ind_1 = [index for index, ele in enumerate(Y) if ele == 1]

# Ploting input data with provided labels
plt.scatter(X[Ind_0,0], X[Ind_0,1], color = "green", marker = "o", s = 25, label = "Alaska")
plt.scatter(X[Ind_1,0], X[Ind_1,1], color = "blue", marker = "*", s = 30, label = "Canada")
plt.xlabel("x1 - fresh water")
plt.ylabel("x2 - marine water")

plt.legend()

if sys.argv[3] == "0":
    train_linear(X, Y)
else:
    train_quadratic(X, Y)