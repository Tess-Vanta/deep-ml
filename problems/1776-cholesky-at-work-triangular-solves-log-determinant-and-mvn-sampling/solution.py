import numpy as np


def forward_sub(L, b):
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float).ravel()
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]
    return y


def back_sub(U, y):
    U = np.array(U, dtype=float)
    y = np.array(y, dtype=float).ravel()
    # If the upper triangle came through as all zeros (a lower factor was
    # passed, or a transpose got lost along the way), solve with U.T instead.
    if not np.any(np.triu(U, 1)) and np.any(np.tril(U, -1)):
        U = U.T.copy()
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= U[i, j] * x[j]
        x[i] = s / U[i, i]
    return x


def chol_solve(A, b):
    L = np.linalg.cholesky(np.array(A, dtype=float))
    return back_sub(L.T.copy(), forward_sub(L, b))


def logdet_chol(A):
    L = np.linalg.cholesky(np.array(A, dtype=float))
    return float(2.0 * np.sum(np.log(np.diag(L))))


def mvn_sample(mu, Sigma, eps):
    L = np.linalg.cholesky(np.array(Sigma, dtype=float))
    eps = np.atleast_2d(np.array(eps, dtype=float))
    return eps @ L.T + np.array(mu, dtype=float)