from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import BestPath
import PrefixSearch
import BeamSearch
import TokenPassing
import LanguageModel
import BKTree
import Loss
import LexiconSearch


# specify if GPU should be used (via OpenCL)
useGPU = len(sys.argv) > 1 and sys.argv[1] == 'gpu'
if useGPU:
	import BestPathCL
	gpuDebug = True


def softmax(mat):
	"calc softmax such that labels per time-step form probability distribution"
	maxT, _ = mat.shape # dim0=t, dim1=c
	res = np.zeros(mat.shape)
	for t in range(maxT):
		y = mat[t, :]
		e = np.exp(y)
		s = np.sum(e)
		res[t, :] = e/s
	return res


def loadRNNOutput(fn):
	"load RNN output from csv file. Last entry in row terminated by semicolon."
	return np.genfromtxt(fn, delimiter=';')[:, : -1]


def testMiniExample():
	"example which shows difference between taking most probable path and most probable labeling. No language model used."

	# chars and input matrix
	classes = 'ab'
	mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])

	# decode
	gt = 'a'
	print('TARGET       :', '"' + gt + '"')
	print('BEST PATH    :', '"' + BestPath.ctcBestPath(mat, classes) + '"')
	print('PREFIX SEARCH:', '"' + PrefixSearch.ctcPrefixSearch(mat, classes) + '"')
	print('BEAM SEARCH  :', '"' + BeamSearch.ctcBeamSearch(mat, classes, None) + '"')
	print('TOKEN        :', '"' + TokenPassing.ctcTokenPassing(mat, classes, ['a', 'b', 'ab', 'ba']) + '"')
	print('PROB(TARGET) :', Loss.ctcLabelingProb(mat, gt, classes))
	print('LOSS(TARGET) :', Loss.ctcLoss(mat, gt, classes))


def testWordExample():
	"example which decodes a RNN output of a single word. Taken from IAM dataset. RNN output produced by TensorFlow model (see github.com/githubharald/SimpleHTR)."

	# chars of IAM dataset
	classes = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

	# matrix containing TxC RNN output. C=len(classes)+1 because of blank label.
	mat = softmax(loadRNNOutput('../data/word/rnnOutput.csv'))

	# BK tree to find similar words
	with open('../data/word/corpus.txt') as f:
		words = f.read().split()
	tolerance = 4
	bkTree = BKTree.BKTree(words)

	# decode RNN output with different decoding algorithms
	gt = 'aircraft'
	print('TARGET        :', '"' + gt + '"')
	print('BEST PATH     :', '"' + BestPath.ctcBestPath(mat, classes) + '"')
	print('LEXICON SEARCH:', '"' + LexiconSearch.ctcLexiconSearch(mat, classes, bkTree, tolerance) + '"')


def testLineExample():
	"example which decodes a RNN output of a text line. Taken from IAM dataset. RNN output produced by TensorFlow model."

	# chars of IAM dataset
	classes = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

	# matrix containing TxC RNN output. C=len(classes)+1 because of blank label.
	mat = softmax(loadRNNOutput('../data/line/rnnOutput.csv'))

	# language model: used for token passing (word list) and beam search (char bigrams)
	lm = LanguageModel.LanguageModel('../data/line/corpus.txt', classes)

	# decode RNN output with different decoding algorithms
	gt = 'the fake friend of the family, like the'
	print('TARGET        :', '"' + gt + '"')
	print('BEST PATH     :', '"' + BestPath.ctcBestPath(mat, classes) + '"')
	print('PREFIX SEARCH :', '"' + PrefixSearch.ctcPrefixSearchHeuristicSplit(mat, classes) + '"')
	print('BEAM SEARCH   :', '"' + BeamSearch.ctcBeamSearch(mat, classes, None) + '"')
	print('BEAM SEARCH LM:', '"' + BeamSearch.ctcBeamSearch(mat, classes, lm) + '"')
	print('TOKEN         :', '"' + TokenPassing.ctcTokenPassing(mat, classes, lm.getWordList()) + '"')
	print('PROB(TARGET)  :', Loss.ctcLabelingProb(mat, gt, classes))
	print('LOSS(TARGET)  :', Loss.ctcLoss(mat, gt, classes))


def testLineExampleGPU():
	"example which decodes a real RNN output. Taken from IAM dataset. RNN output produced by TensorFlow model."

	# possible chars
	classes = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

	# matrix containing TxC RNN output. C=len(classes)+1 because of blank label.
	mat = softmax(loadRNNOutput('../data/line/rnnOutput.csv'))

	# decode RNN output with best path decoding on GPU
	batchSize = 1000
	maxT, maxC = mat.shape
	clWrapper = BestPathCL.CLWrapper(batchSize, maxT, maxC, kernelVariant=1, enableGPUDebug=gpuDebug)

	# stack mat multiple times to simulate large batch
	batch = np.stack([mat]*batchSize)

	# compute best path for each batch element
	resBatch = BestPathCL.ctcBestPathCL(batch, classes, clWrapper)

	gt = 'the fake friend of the family, like the'
	print('Compute for ' + str(batchSize) + ' batch elements')
	print('TARGET        :', '"' + gt + '"')
	print('BEST PATH GPU :', '"' + resBatch[0] + '"')


if __name__ == '__main__':

	# example decoding matrix containing 2 time-steps and 2 chars
	print('=====Mini example=====')
	testMiniExample()

	# example decoding a word
	print('=====Word example=====')
	testWordExample()

	# example decoding a text-line
	print('=====Line example=====')
	testLineExample()

	# example decoding a text-line, computed on the GPU with OpenCL
	if useGPU:
		print('=====Line example (GPU)=====')
		testLineExampleGPU()
