3rd problem:
X_cus = np.copy(X_in)

for j in range(n):
    mean = 0.
    std = 0.
    for i in range(m):
        mean += X_cus[i][j]
        std += X_cus[i][j] * X_cus[i][j]
    mean /= m
    std /= m
    std -= mean*mean
    std = math.sqrt(std)
    for i in range(m):
        X_cus[i][j] = (X_cus[i][j]-mean)/std


#     x_axis = np.linspace(0,10,400)
#     y_axis = -(theta[0]/theta[1]) * x_axis - (theta[2]/theta[1])
#     plt.plot(x_axis, y_axis, '-r')
    
#     plt.scatter(X_in[:,0],X_in[:,1],c=y_pred)
#     plt.scatter(X_in[:,0],X_in[:,1],c=y_in)
#     plt.show()