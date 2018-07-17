import BestPath
import Loss
import LanguageModel
import editdistance


def ctcLexiconSearch(mat, classes, lm):
	"compute approximation with best path decoding, search most similar words in dictionary, calculate score for each of them, return best scoring one. See Shi, Bai and Yao."
	
	# use best path decoding to get an approximation
	approx = BestPath.ctcBestPath(mat, classes)
	
	# search words with minimal edit-distance to the approximation (speed-up possible by using BK-tree data-structure)
	keepBest = 10
	dist = [(w, editdistance.eval(approx, w)) for w in lm.getWordList()] # edit-distance of words to the recognized word from best path decoding
	dist = sorted(dist, key=lambda x: x[1])[:keepBest] # keep 10 best words w.r.t. edit-distance

	# for each word candidate, calculate probability and keep best-scoring word
	probs = [(entry[0], Loss.ctcLabelingProb(mat, entry[0], classes)) for entry in dist]
	probs = sorted(probs, key=lambda x: x[1], reverse=True)
	
	return probs[0][0]
