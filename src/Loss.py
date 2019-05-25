from __future__ import division
from __future__ import print_function
import math
import numpy as np
import Common


def recLabelingProb(t, s, mat, labelingWithBlanks, blank, cache):
	"recursively compute probability of labeling, save results of sub-problems in cache to avoid recalculating them"

	# check index of labeling
	if s < 0:
		return 0.0

	# sub-problem already computed
	if cache[t][s] != None:
		return cache[t][s]

	# initial values
	if t == 0:
		if s == 0:
			res = mat[0, blank]
		elif s == 1:
			res = mat[0, labelingWithBlanks[1]]
		else:
			res = 0.0

		cache[t][s] = res
		return res

	# recursion on s and t
	res = (recLabelingProb(t-1, s, mat, labelingWithBlanks, blank, cache) + recLabelingProb(t-1, s-1, mat, labelingWithBlanks, blank, cache)) * mat[t, labelingWithBlanks[s]]

	# in case of a blank or a repeated label, we only consider s and s-1 at t-1, so we're done
	if labelingWithBlanks[s] == blank or (s >= 2 and labelingWithBlanks[s-2] == labelingWithBlanks[s]):
		cache[t][s] = res
		return res

	# otherwise, in case of a non-blank and non-repeated label, we additionally add s-2 at t-1
	res += recLabelingProb(t-1, s-2, mat, labelingWithBlanks, blank, cache) * mat[t, labelingWithBlanks[s]]
	cache[t][s] = res
	return res


def emptyCache(maxT, labelingWithBlanks):
	"create empty cache"
	return [[None for _ in range(len(labelingWithBlanks))] for _ in range(maxT)]


def ctcLabelingProb(mat, gt, classes):
	"calculate probability p(gt|mat) of a given labeling gt and a matrix mat according to section 'The CTC Forward-Backward Algorithm' in Graves paper"
	maxT, _ = mat.shape # size of input matrix
	blank = len(classes) # index of blank label
	labelingWithBlanks = Common.extendByBlanks(Common.wordToLabelSeq(gt, classes), blank) # ground truth text as label string extended by blanks
	cache = emptyCache(maxT, labelingWithBlanks) # cache subresults to avoid recalculating  subproblems over and over again
	return recLabelingProb(maxT-1, len(labelingWithBlanks)-1, mat, labelingWithBlanks, blank, cache) + recLabelingProb(maxT-1, len(labelingWithBlanks)-2, mat, labelingWithBlanks, blank, cache)


def ctcLoss(mat, gt, classes):
	"calculate CTC loss"
	try:
		return -math.log(ctcLabelingProb(mat, gt, classes))
	except:
		return float('inf')


def testLoss():
	"test loss"
	classes = 'ab'
	mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
	print('Test loss calculation')
	expected = 0.64
	actual = ctcLabelingProb(mat, 'a', classes)
	print('Expected: ' + str(expected))
	print('Actual: ' + str(actual))
	print('OK' if expected == actual else 'ERROR')


if __name__ == '__main__':
	testLoss()
