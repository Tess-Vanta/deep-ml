def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    r, c = len(a), len(a[0])
    res = [[0] * r for _ in range(c)]
    for i in range(r):
        for j in range(c):
            res[j][i] = a[i][j]
    return res

