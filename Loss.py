from __future__ import division
from __future__ import print_function
import numpy as np
import Common
import math


def alpha_prime(t, s, mat, lp, blank, cache):
	"alpha(t-1,s)+alpha(t-1,s-1)"
	return alpha(t-1, s, mat, lp, blank, cache)+alpha(t-1, s-1, mat, lp, blank, cache)


def alpha(t, s, mat, lp, blank, cache):
	"calculate alpha value. t=time, s=label index, mat=matrix, lp=label with blanks, blank=index of blank, cache=table holding already calculated (sub-)results"
	
	# 0 for s<0
	if s<0:
		return 0.0
	
	# already calculated?
	if cache[t][s]!=None:
		return cache[t][s]
	
	# initial values
	if t==0:
		if s==0:
			res=mat[0, blank]
			cache[t][s]=res
			return res
		elif s==1:
			res=mat[0, lp[1]]
			cache[t][s]=res
			return res
		else:
			res=0.0
			cache[t][s]=res
			return res
	
	# recursion on s and t
	if lp[s]==blank or lp[s-2]==lp[s]:
		res=alpha_prime(t, s, mat, lp, blank, cache)*mat[t,lp[s]]
		cache[t][s]=res
		return res
	else:
		res=(alpha_prime(t, s, mat, lp, blank, cache)+alpha(t-1, s-2, mat, lp, blank, cache))*mat[t,lp[s]]
		cache[t][s]=res
		return res


def emptyCache(maxT,lp):
	"create empty cache"
	return [[None for _ in range(len(lp))] for _ in range(maxT)]


def ctcLabelingProb(mat, gt, classes):
	"calculate probability p(gt|mat) of a given labeling gt and a matrix mat according to section 'The CTC Forward-Backward Algorithm' in Graves paper"
	maxT,maxC=mat.shape # size of input matrix
	blank=len(classes) # index of blank label
	lp=Common.extendByBlanks(Common.wordToLabelSeq(gt, classes), blank) # ground truth text as label string extended by blanks
	cache=emptyCache(maxT, lp) # cache subresults to avoid recalculating the same subproblems 
	return alpha(maxT-1, len(lp)-1, mat, lp, blank, cache) + alpha(maxT-1, len(lp)-2, mat, lp, blank, cache)

	
def ctcLoss(mat, gt, classes):
	"calculate CTC loss"
	return -math.log(ctcLabelingProb(mat, gt, classes))
	

if __name__=='__main__':
	classes="ab"
	mat=np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
	print('Test loss calculation')
	print(ctcLoss(mat, 'aa', classes))
	
	