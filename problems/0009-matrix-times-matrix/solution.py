def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    
    m, x = len(a), len(a[0])
    y, n = len(b), len(b[0])

    if x != y:
        return -1
    
    c = []
    for i in range(m):
        new_row = []
        for j in range(n):
            tot = 0
            for k in range(x):
                tot += a[i][k] * b[k][j]
            new_row.append(tot)
        c.append(new_row)
    
    return c