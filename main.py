from __future__ import division
from __future__ import print_function
import numpy as np
import BestPath
import PrefixSearch
import BeamSearch
import TokenPassing
import LanguageModel
import Loss


def softmax(mat):
	"calc softmax such that labels per time-step form probability distribution"
	# dim0=t, dim1=c
	maxT,maxC=mat.shape
	res=np.zeros(mat.shape)
	for t in range(maxT):
			y=mat[t,:]
			e=np.exp(y)
			s=np.sum(e)
			res[t,:]=e/s

	return res


def loadRNNOutput(fn):
	"load matrix from csv file. Last entry in row terminated by semicolon."
	return np.genfromtxt(fn, delimiter=';')[:,:-1]


def testRealExample():
	"example which decodes a real RNN output. Taken from IAM dataset. RNN output produced by TensorFlow model."
	
	# possible chars
	classes=' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

	# matrix containing TxC RNN output. C=len(classes)+1 because of blank label.
	mat=softmax(loadRNNOutput('data/rnnOutput.csv'))

	# language model: used for token passing (word list) and beam search (char bigrams)
	lm=LanguageModel.LanguageModel('data/corpus.txt', classes)
	
	# decode RNN output with different decoding algorithms
	gt='the fake friend of the family, like the'
	print('TARGET        :','"'+gt+'"')
	print('BEST PATH     :', '"'+BestPath.ctcBestPath(mat, classes)+'"')
	print('PREFIX SEARCH :', '"'+PrefixSearch.ctcPrefixSearchHeuristicSplit(mat, classes)+'"')
	print('BEAM SEARCH   :', '"'+BeamSearch.ctcBeamSearch(mat, classes, None)+'"')
	print('BEAM SEARCH LM:', '"'+BeamSearch.ctcBeamSearch(mat, classes, lm)+'"')
	print('TOKEN         :', '"'+TokenPassing.ctcTokenPassing(mat, classes, lm.getWordList())+'"')
	print('PROB(TARGET)  :', Loss.ctcLabelingProb(mat, gt, classes))
	print('LOSS(TARGET)  :', Loss.ctcLoss(mat, gt, classes))


def testMiniExample():
	"example which shows difference between taking most probable path and most probable labeling. No language model used."
	
	# possible chars and input matrix
	classes="ab"
	mat=np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
	
	# decode
	gt='a'
	print('TARGET       :','"'+gt+'"')
	print('BEST PATH    :', '"'+BestPath.ctcBestPath(mat, classes)+'"')
	print('PREFIX SEARCH:', '"'+PrefixSearch.ctcPrefixSearch(mat, classes)+'"')
	print('BEAM SEARCH  :', '"'+BeamSearch.ctcBeamSearch(mat, classes, None)+'"')
	print('TOKEN        :', '"'+TokenPassing.ctcTokenPassing(mat, classes, ['a','b','ab','ba'])+'"')
	print('PROB(TARGET) :', Loss.ctcLabelingProb(mat, gt, classes))
	print('LOSS(TARGET) :', Loss.ctcLoss(mat, gt, classes))


if __name__=='__main__':
	print('=====Mini example=====')
	testMiniExample()
	
	print('=====Real example=====')
	testRealExample()

