from __future__ import division
from __future__ import print_function


def extendByBlanks(seq, b):
	"extends a label seq. by adding blanks at the beginning, end and in between each label"
	res = [b]
	for s in seq:
		res.append(s)
		res.append(b)
	return res


def wordToLabelSeq(w, classes):
	"map a word to a sequence of labels (indices)"
	res = [classes.index(c) for c in w]
	return res
