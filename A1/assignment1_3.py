
# coding: utf-8

# In[70]:


import numpy as np


# In[71]:


def train(X_in, Y):

    """
    Input :

    Ouput :

    """
    n = 2
    m = X_in.size
    np.set_printoptions(precision=3,suppress=True)
    X_un = np.copy(X_in)
    
    for i in range(2):
        X_in[:,i] = (X_in[:,i] - X_in[:,i].mean())/np.std(X_in[:,i])
#     print(X_un,X_in)

#     X = np.ones((m,n))
#     X[:,0] = X_in
#     Y_pred = np.zeros(m)
#     X_T = np.transpose(X)

#     for i in range(m):
#         W = np.diag(np.exp(-(np.square(X_in-X_in[i]))/(2.*tau*tau)))
#         W[i][i] = 0.
#         theta = np.matmul(np.linalg.inv(np.matmul(X_T, np.matmul(W, X))), np.matmul(X_T, np.matmul(W, Y)))
# #         print(theta.shape)
# #         print(type(theta),type(Y_pred[i]),type(X))
# #         print(theta.shape,X.shape)
#         Y_pred[i] = np.matmul(X[i],theta)
# #         if(X[i][0] > 2.5): 
            
# # #             print(X[i],theta,Y[i],Y_pred[i])
# # #             print(np.diag(W))
# #             print(X[i])
# #             A = np.matmul(np.matmul(X_T, W), X)
# #             print(A.shape)
# #             B = np.linalg.inv(A)
# #             np.set_printoptions(precision=3,suppress=True)
# #             print(A,B)
# #             np.set_printoptions(precision=3,suppress=True)
# #             C = np.matmul(A,B)
# #             print(C)
# #             print(np.matmul(A,B))
#     return Y_pred


# In[72]:


x_in = np.genfromtxt('ass1_data/logisticX.csv',delimiter=',')
y_in = np.genfromtxt('ass1_data/logisticY.csv',delimiter=',')

print(x_in.shape, y_in.shape)
train(x_in, y_in)
# y_pred = train(x_in, y_in)
# print(y_pred,y_in)
# for i in range(x_in.size):
#     print(x_in[i],y_pred[i],y_in[i])

# import matplotlib.pyplot as plt
# plt.plot(x_in,y_in,'ro',color='blue')
# plt.plot(x_in,y_pred,'ro',color='red')
# # plt.scatter(x,y,color='red')
# # plt.scatter(x,y1,color='blue')
# plt.show()

