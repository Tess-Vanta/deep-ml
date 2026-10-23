import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(y)
    weights = np.asarray(weights, dtype=float)

    def mse_update(X_part, y_part):
        error = X_part @ weights - y_part
        return (2 / len(y_part)) * (X_part.T @ error)
    
    for _ in range(n_epochs):

        if method == 'batch':
            weights -= learning_rate * mse_update(X, y)
        
        elif method == 'stochastic':
            for i in range(m):
                weights -= learning_rate * mse_update(X[i:i+1], y[i:i+1])
        
        else: 
            for start in range(0, m, batch_size):
                end = start + batch_size
                weights -= learning_rate * mse_update(X[start:end], y[start:end])
    
    return weights
    
