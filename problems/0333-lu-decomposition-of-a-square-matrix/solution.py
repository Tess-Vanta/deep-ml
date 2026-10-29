import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	# Your code here
	A = np.asarray(A, dtype=float)
	n = len(A)

	L, U = np.eye(n), np.zeros((n, n))

	for i in range(n):
		for j in range(i, n):
			U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]

		for j in range(i + 1, n):
			L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]
		
	return L, U