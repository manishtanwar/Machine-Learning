if(iter%50 == 0):
	if(iter==0):
		print("True Prediction")
		count = np.zeros(c, dtype=int)
		for i in range(y.shape[0]):
			count[y[i][0]] += 1
		print(count)
	print("iter: ",iter,'\n')
	(y_pred, _) = forward_pass(x, y)
	# print(y_pred.shape, y.shape)
	y_lb = np.argmax(y_pred, axis=0)
	count = np.zeros(c, dtype=int)
	for i in range(y_lb.shape[0]):
		count[y_lb[i]] += 1
	print(count)
	y1 = y[:,0]
	# print(y_lb.shape, y1.shape)

	print("Loss : ", loss_function(y_pred, y))
	print(np.sum(y1 == y_lb))
	print("Train Accuracy:", 100.0 * (np.sum(y1 == y_lb)/y1.shape[0]))

	(y_predt, _) = forward_pass(xt, yt)
	# print(y_pred.shape, y.shape)
	y_lbt = np.argmax(y_predt, axis=0)
	y1t = yt[:,0]
	print("Test Accuracy:", 100.0 * (np.sum(y1t == y_lbt)/y1t.shape[0]))

	sys.stdout.flush()