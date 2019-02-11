# coding: utf-8
# In[319]:
import numpy as np
# In[320]:
x_in = np.genfromtxt('ass1_data/linearX.csv',delimiter=',')
y = np.genfromtxt('ass1_data/linearY.csv',delimiter=',')

# x_in = np.genfromtxt('ass1_data/weightedX.csv',delimiter=',')
# y = np.genfromtxt('ass1_data/weightedY.csv',delimiter=',')
# In[321]:
theta = np.zeros(2)
# In[322]:

# x_un : unnormalized data
x_un = np.copy(x_in)
x_in = (x_in - x_in.mean())/np.std(x_in)
rate = 0.5
n = 2
m = x_in.size

# In[323]:
# x_in = x_in[:,np.newaxis]
X = np.ones((100,2))
X[:,0] = x_in
# print(X)

# In[324]:
X_T = np.transpose(X)
iter = 0
while True:
    iter+=1
    delta_J = (np.matmul(np.matmul(X_T,X), theta) - np.matmul(X_T,y))
    theta = theta - (rate/m) * delta_J
#     print(delta_J)
    max_change = max(abs(delta_J[0]),abs(delta_J[1]))
#     print(max_change)
    if(max_change < 1e-8):
        break
print(iter)

# In[325]:
y_pred = np.matmul(X,theta)

# In[326]:
theta

# In[327]:
import matplotlib.pyplot as plt
plt.plot(x_un,y,'ro',color='blue')
plt.plot(x_un,y_pred,color='red')
# plt.scatter(x,y,color='red')
# plt.scatter(x,y1,color='blue')
plt.show()