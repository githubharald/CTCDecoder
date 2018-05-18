from __future__ import division
from __future__ import print_function
import numpy as np


def ctcPrefixSearch(mat, classes):
	"implements CTC Prefix Search Decoding Algorithm as shown by Graves (Dissertation, p63-66)"
	blankIdx = len(classes)
	maxT, maxC = mat.shape

	# g_n and g_b: gamma in paper
	g_n = []
	g_b = []

	# p(y|x) and p(y...|x), where y is a prefix (not p as in paper to avoid confusion with probability)
	prob = {}
	prob_ext = {}

	# Init: 1-6
	for t in range(maxT):
		g_n.append({'' : 0})
		last = g_b[t - 1][''] if t > 0 else 1
		g_b.append({'' : last * mat[t, blankIdx]})

	# init for empty prefix
	prob[''] = g_b[maxT - 1]['']
	prob_ext[''] = 1 - prob['']
	l_star = y_star = ''
	Y = set([''])

	# Algorithm: 8-31
	while prob_ext[y_star] > prob[l_star]:
		probRemaining = prob_ext[y_star]

		# for all labels
		for k in range(maxC - 1):
			y = y_star + classes[k]
			g_n[0][y] = mat[0, k] if len(y_star) == 0 else 0
			g_b[0][y] = 0
			prefixProb = g_n[0][y]

			# for all time steps
			for t in range(1, maxT):
				newLabelProb = g_b[t - 1][y_star] + (0 if y_star != '' and y_star[-1] == classes[k] else g_n[t-1][y_star])
				g_n[t][y] = mat[t, k] * (newLabelProb + g_n[t - 1][y])
				g_b[t][y] = mat[t, blankIdx] * (g_b[t - 1][y] + g_n[t - 1][y])
				prefixProb += mat[t, k] * newLabelProb

			prob[y] = g_n[maxT - 1][y] + g_b[maxT - 1][y]
			prob_ext[y] = prefixProb - prob[y]
			probRemaining -= prob_ext[y]

			if prob[y] > prob[l_star]:
				l_star = y
			if prob_ext[y] > prob[l_star]:
				Y.add(y)
			if probRemaining <= prob[l_star]:
				break

		# 30
		Y.remove(y_star)

		# 31
		bestY = None
		bestProbExt = 0
		for y in Y:
			if prob_ext[y] > bestProbExt:
				bestProbExt = prob_ext[y]
				bestY = y
		y_star = bestY

		# terminate if no more prefix exists
		if bestY == None:
			break

	# Termination: 33-34
	return l_star


def ctcPrefixSearchHeuristicSplit(mat, classes):
	"speed up prefix computation by splitting sequence into subsequences as described by Graves (Dissertation, p66)"
	blankIdx = len(classes)
	maxT, _ = mat.shape

	# split sequence into 3 subsequences, splitting points should be roughly placed at 1/3 and 2/3
	splitTargets = [int(maxT * 1 / 3), int(maxT * 2 / 3)]
	best = [{'target' : s, 'bestDist' : maxT, 'bestIdx' : s} for s in splitTargets]

	# find good splitting points (blanks above threshold)
	thres = 0.9
	for t in range(maxT):
		for b in best:
			if mat[t, blankIdx] > thres and abs(t - b['target']) < b['bestDist']:
				b['bestDist'] = abs(t - b['target'])
				b['bestIdx'] = t
				break

	# splitting points plus begin and end of sequence
	ranges = [0] + [b['bestIdx'] for b in best] + [maxT]

	# do prefix search for each subsequence and concatenate results
	res = ''
	for i in range(len(ranges) - 1):
		beg = ranges[i]
		end = ranges[i + 1]
		res += ctcPrefixSearch(mat[beg : end, :], classes)

	return res


def testPrefixSearch():
	classes = 'ab'
	mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
	print('Test prefix search decoding')
	expected = 'a'
	actual = ctcPrefixSearch(mat, classes)
	print('Expected: "' + expected + '"')
	print('Actual: "' + actual + '"')
	print('OK' if expected == actual else 'ERROR')


if __name__ == '__main__':
	testPrefixSearch()
