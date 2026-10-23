import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    x = np.asarray(x, dtype=float)
    numerical_grad = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus.flat[i] += epsilon
        x_minus.flat[i] -= epsilon

        numerical_grad.flat[i] += (f(x_plus) - f(x_minus)) / (2 * epsilon)
    
    top = np.linalg.norm(numerical_grad - analytical_grad)
    bottom = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)

    if not bottom:
        relative_error = 0.0
    else:
        relative_error = top / bottom

    return numerical_grad, relative_error  