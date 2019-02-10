
# coding: utf-8

# In[400]:


import numpy as np


# In[401]:


def train(X_in, Y, tau):

    """
    Input :

    Ouput :

    """

    X_un = np.copy(X_in)
#     X_in = (X_in - X_in.mean())/np.std(X_in)

    n = 2
    m = X_in.size

    X = np.ones((m,n))
    X[:,0] = X_in
    Y_pred = np.zeros(m)
    X_T = np.transpose(X)

    for i in range(m):
        W = np.diag(np.exp(-(np.square(X_in-X_in[i]))/(2.*tau*tau)))
        W[i][i] = 0.
        theta = np.matmul(np.linalg.inv(np.matmul(X_T, np.matmul(W, X))), np.matmul(X_T, np.matmul(W, Y)))
#         print(theta.shape)
#         print(type(theta),type(Y_pred[i]),type(X))
#         print(theta.shape,X.shape)
        Y_pred[i] = np.matmul(X[i],theta)
    return Y_pred


# In[402]:


x_in = np.genfromtxt('ass1_data/weightedX.csv',delimiter=',')
y_in = np.genfromtxt('ass1_data/weightedY.csv',delimiter=',')
tau = 0.1
y_pred = train(x_in, y_in, tau)
# train(x_in, y_in, tau)
# print(y_pred,y_in)
# for i in range(x_in.size):
#     print(x_in[i],y_pred[i],y_in[i])

import matplotlib.pyplot as plt
plt.plot(x_in,y_in,'ro',color='blue')
plt.plot(x_in,y_pred,'ro',color='red')
# plt.scatter(x,y,color='red')
# plt.scatter(x,y1,color='blue')
plt.show()


# In[403]:


A1 = np.array([[1.,4.35,1.], [4.375, 1.,1.], [4., 1.,1.]])
print(A1)
B1 = np.linalg.inv(A1)
print(B1)
print(A1.dot(B1))

