import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    try:
        A = np.array(A, dtype=float)
    except (ValueError, TypeError):
        return -1
    
    if A.ndim != 2 or A.size == 0:
        return -1
    n, m = A.shape 
    if n != m:
        return -1
    if not np.all(np.isfinite(A)):
        return -1
    if not np.allclose(A, A.T):
        return -1
    
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = L[i, :j] @ L[j, :j]
            d = A[i, j] - s
            if i == j:
                if d <= 1e-12:
                    return -1
                L[i, j] = np.sqrt(d)
            else:
                L[i, j] = d / L[j, j]
    
    return L.tolist()