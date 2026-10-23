# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your triaining strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    n_epochs, learning_rate = 1000, 0.01
    n, _ = X.shape
    for _ in range(1000):
        error = X @ W - y
        dw = (2 / n) * (X.T @ error)
        db = (2 / n) * np.sum(error)
        W -= learning_rate * dw
        b -= learning_rate * db
    return W, b