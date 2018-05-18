from __future__ import division
from __future__ import print_function
import numpy as np


def ctcBestPath(mat, classes):
	"implements best path decoding as shown by Graves (Dissertation, p63)"

	# dim0=t, dim1=c
	maxT, maxC = mat.shape
	label = ''
	blankIdx = len(classes)
	lastMaxIdx = maxC # init with invalid label

	for t in range(maxT):
		maxIdx = np.argmax(mat[t, :])

		if maxIdx != lastMaxIdx and maxIdx != blankIdx:
			label += classes[maxIdx]

		lastMaxIdx = maxIdx

	return label


def testBestPath():
	"test decoder"
	classes = 'ab'
	mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
	print('Test best path decoding')
	expected = ''
	actual = ctcBestPath(mat, classes)
	print('Expected: "' + expected + '"')
	print('Actual: "' + actual + '"')
	print('OK' if expected == actual else 'ERROR')


if __name__ == '__main__':
	testBestPath()
