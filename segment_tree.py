	def segfunc(x,y):
		return min(x,y)

	def init(init_val):
		for i in range(N):
			seg[i+num-1] = init_val[i]
		for i in range(num-2,-1,-1):
			seg[i] = segfunc(seg[2*i+1],seg[2*i+2])

	def update(k,x):
		k += num-1
		seg[k] = x
		while k:
			k = (k-1)//2
			seg[k] = segfunc(seg[2*k+1],seg[2*k+2])

	def query(p,q):
		if q <= p:
			return ide_ele
		p += num-1
		q += num-2
		ret = ide_ele
		while q-p > 1:
			if p&1 == 0:
				ret = segfunc(ret,seg[p])
			if q&1 == 1:
				ret = segfunc(ret,seg[q])
				q -= 1
			p = p//2
			q = (q-1)//2
		ret = segfunc(segfunc(ret,seg[p]),seg[q])
		return ret

	# 今回はsegfuncがminなので
	ide_ele = 10**15

	num = 2**(N-1).bit_length()
	seg = [ide_ele]*2*num
	init(a)