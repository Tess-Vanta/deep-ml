import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	n = A.shape[0]

	for k in range(n - 1):
		p = k + np.argmax(np.abs(A[k:, k]))
		if p != k:
			A[[k, p]] = A[[p, k]]
			b[[k, p]] = b[[p, k]]
		m = A[k + 1:, k] / A[k, k]
		A[k + 1:, k:] -= np.outer(m, A[k, k:])
		b[k + 1:] -= m * b[k]
	
	x = np.empty(n)
	for k in range(n - 1, -1, -1):
		x[k] = (b[k] - A[k, k + 1:] @ x[k + 1:]) / A[k, k]
	return x