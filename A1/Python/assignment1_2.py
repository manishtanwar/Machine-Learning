
# coding: utf-8

# In[396]:


import numpy as np


# In[397]:


def train(X_in, Y, tau):

    """
    Input :

    Ouput :

    """

    X_un = np.copy(X_in)
    # X_in = (X_in - X_in.mean())/np.std(X_in)

    n = 2
    m = X_in.size

    X = np.ones((m,n))
    X[:,0] = X_in
    Y_pred = np.zeros(m)
    np.set_printoptions(precision=3,suppress=True)
    X_T = np.transpose(X)

    print(X,X_in,Y)

    for i in range(m):
        X1 = np.delete(X, (i), axis=0)
        Y1 = np.delete(Y, (i), axis=0)
        X_in1 = np.delete(X_in, (i), axis=0);
        print(X1,X_in1,Y1)
        
        X_T1 = np.transpose(X1)

        W = np.diag(np.exp(-(np.square(X_in1-X_in[i]))/(2.*tau*tau)))
        # print(np.diag(W))
        # W[i][i] = 0.
        theta = np.matmul(np.linalg.inv(np.matmul(X_T1, np.matmul(W, X1))), np.matmul(X_T1, np.matmul(W, Y1)))
        # print(np.diag())
#         print(type(theta),type(Y_pred[i]),type(X))
#         print(theta.shape,X.shape)
        Y_pred[i] = np.matmul(X[i],theta)
    return Y_pred


# In[398]:


x_in = np.genfromtxt('../ass1_data/weightedX.csv',delimiter=',')
y_in = np.genfromtxt('../ass1_data/weightedY.csv',delimiter=',')
tau = 0.1
y_pred = train(x_in, y_in, tau)
# train(x_in, y_in, tau)
# print(y_pred)
# for i in range(x_in.size):
#     print(x_in[i],y_pred[i],y_in[i])

import matplotlib.pyplot as plt
# plt.plot(x_in,y_in,'ro',color='blue')
# plt.plot(x_in,y_pred,'ro',color='red')
# plt.scatter(x,y,color='red')
# plt.scatter(x,y1,color='blue')
plt.scatter(x_in,y_in,color='blue',marker="o",s=30)
plt.scatter(x_in,y_pred,color='red',marker="o",s=30)

plt.show()


# In[399]:


A1 = np.array([[1.,4.35,1.], [4.375, 1.,1.], [4., 1.,1.]])
print(A1)
B1 = np.linalg.inv(A1)
print(B1)
print(A1.dot(B1))

