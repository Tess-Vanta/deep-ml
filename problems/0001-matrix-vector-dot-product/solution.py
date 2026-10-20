def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	m, x = len(a), len(a[0])
	y = len(b)
	if x != y: return -1
	res = []

	for row in a:
		tot = 0
		for i in range(y):
			tot += row[i] * b[i]
		res.append(tot)
	
	return res
	

