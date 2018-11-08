import BestPath
import Loss


def ctcLexiconSearch(mat, classes, bkTree, tolerance):
	"compute approximation with best path decoding, search most similar words in dictionary, calculate score for each of them, return best scoring one. See Shi, Bai and Yao."

	# use best path decoding to get an approximation
	approx = BestPath.ctcBestPath(mat, classes)

	# get similar words from dictionary within given tolerance
	words = bkTree.query(approx, tolerance)

	# if there are no similar words, return empty string
	if not words:
		return ''

	# else compute probabilities of all similar words and return best scoring one
	wordProbs = [(w, Loss.ctcLabelingProb(mat, w, classes)) for w in words]
	wordProbs.sort(key=lambda x: x[1], reverse=True)
	return wordProbs[0][0]
