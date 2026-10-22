
def orthogonal_projection(v, L):
	n = len(L)

	vL, LL = 0, 0
	for i in range(n):
		vL += (v[i] * L[i])
		LL += (L[i] ** 2)
	
	proj = vL / LL
	res = []
	for i in range(n):
		res.append(proj * L[i])
	
	return res