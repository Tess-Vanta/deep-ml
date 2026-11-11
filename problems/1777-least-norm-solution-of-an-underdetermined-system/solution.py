import numpy as np


def min_norm_solution(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).ravel()
    w = np.linalg.solve(A @ A.T, b)          # (A A^T) w = b
    return A.T @ w                            # x = A^T w


def min_norm_via_kkt(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).ravel()
    m, n = A.shape
    K = np.zeros((n + m, n + m))
    K[:n, :n] = 2.0 * np.eye(n)               # grad_x:   2x + A^T lam = 0
    K[:n, n:] = A.T
    K[n:, :n] = A                             # grad_lam: A x = b
    rhs = np.concatenate([np.zeros(n), b])
    sol = np.linalg.solve(K, rhs)
    return sol[:n].copy(), sol[n:].copy()


def min_norm_from_particular(A, x_p):
    A = np.array(A, dtype=float)
    x_p = np.array(x_p, dtype=float).ravel()
    w = np.linalg.solve(A @ A.T, A @ x_p)     # project x_p onto row(A)
    return A.T @ w