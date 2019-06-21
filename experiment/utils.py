def scale_function(y, min_new, max_new):
    
    min_old = y.min()
    max_old = y.max()
    scaled_y = ((max_new-min_new)/(max_old-min_old) * (y-min_old) + min_new)
    return scaled_y


def get_series_extrap(X, y, train_size, test_size):
    
    trainX = X[:train_size]
    testX = X[train_size:test_size+train_size]
    trainy = y[:train_size]
    testy = y[train_size:test_size+train_size]
    return trainX, trainy, testX, testy