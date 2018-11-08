import editdistance as ed


class BKTree:
	"Burkhard Keller tree: used to find strings within tolerance (w.r.t. edit distance metric) to given query string"


	def __init__(self, txtList):
		"pass list of texts (words) to insert into tree"
		self.root = None
		for txt in txtList:
			self._insert(self.root, txt)


	def query(self, txt, tolerance):
		"query strings within given tolerance (w.r.t. edit distance metric)"
		return self._query(self.root, txt, tolerance)


	def _insert(self, node, txt):
		# insert root node
		if node is None:
			self.root = (txt, {})
			return

		# insert all other nodes
		d = ed.eval(node[0], txt)
		if d in node[1]:
			self._insert(node[1][d], txt)
		else:
			node[1][d] = (txt, {})


	def _query(self, node, txt, tolerance):
		# handle empty root node
		if node is None:
			return []

		# distance between query and current node
		d = ed.eval(node[0], txt)

		# add current node to result if within tolerance
		res = []
		if d <= tolerance:
			res.append(node[0])

		# iterate over children
		for (edge, child) in node[1].items():
			if d - tolerance <= edge and edge <= d + tolerance:
				res += self._query(child, txt, tolerance)

		return res


def testBKTree():
	"test BK tree on words from corpus"
	with open('../data/word/corpus.txt') as f:
		words = f.read().split()

	tolerance = 2
	t = BKTree(words)
	q = 'air'
	res1 = sorted(t.query(q, tolerance))
	res2 = sorted([w for w in words if ed.eval(q, w) <= tolerance])
	print(res1)
	print(res2)
	assert res1 == res2


if __name__ == '__main__':
	testBKTree()
