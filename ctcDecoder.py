from __future__ import division
from __future__ import print_function
import numpy as np
import codecs
import math
import re


class LanguageModel:
	"simple language model for CTC: word list for token passing, char bigrams for beam search"
	def __init__(self, fn, classes):
		"read text from file to generate language model"
		self.classes=classes
		self.initWordList(fn)
		self.initCharBigrams(fn)


	def initWordList(self, fn):
		"internal init of word list"
		txt=open(fn).read().lower()
		words=re.findall('\w+', txt)
		self.words=list(filter(lambda x:x.isalpha(), words))


	def initCharBigrams(self, fn):
		"internal init of character bigrams"
		self.bigram={}
		self.numSamples={}
		txt=codecs.open(fn, 'r', 'utf8').read()
		
		# init bigrams with 0 values
		for c in self.classes:
			self.bigram[c]={}
			self.numSamples[c]=len(self.classes)
			for d in self.classes:
					self.bigram[c][d]=0
		
		# go through text and create each char bigrams
		for i in range(len(txt)-1):
			first=txt[i]
			second=txt[i+1]
			
			# ignore unknown chars
			if first not in self.bigram or second not in self.bigram[first]:
					continue
			
			self.bigram[first][second]+=1
			self.numSamples[first]+=1


	def getCharBigram(self, first, second):
		"probability of seeing character 'first' next to 'second'"
		first=first if len(first) else ' ' # map start to word beginning
		second=second if len(second) else ' ' # map end to word end
		
		if self.numSamples[first]==0:
				return 0
		return self.bigram[first][second]/self.numSamples[first]


	def dump(self, fn):
		"dumps character bigrams to csv file"
		res='CHARBIGRAMS;\n'
		for k1 in self.classes:
			for k2 in self.classes:
				l1=self.classes.index(k1)
				l2=self.classes.index(k2)
				res+=str(l1)+';'+str(l2)+';'+str(self.getCharBigram(k1, k2))+';\n'
		open(fn,'w+').write(res)


	def getWordList(self):
		"get list of unique words"
		return self.words


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


def ctcBestPath(mat, classes):
	"implements best path decoding as shown by Graves (Dissertation, p63)"
	
	label=''
	blankIdx=len(classes)
	lastMaxIdx=-1

	# dim0=t, dim1=c
	maxT,maxC=mat.shape
	for t in range(maxT):
			maxIdx=np.argmax(mat[t,:])
			if maxIdx!=lastMaxIdx and maxIdx!=blankIdx:
					label+=classes[maxIdx]
			
			lastMaxIdx=maxIdx

	return label


class Token:
	"token for token passing algorithm. Each token contains a score and a history of visited words."
	def __init__(self, score=float("-inf"), history=[]):
		self.score=score
		self.history=history
	
	def __str__(self):
		res='class Token: '+str(self.score)+'; '
		for w in self.history:
			res+=w+'; '
		return res
		
		
class TokenList:
	"this class simplifies getting/setting tokens"
	def __init__(self):
		self.tokens={}
		
	def set(self, w, s, t, tok):
		self.tokens[(w, s, t)]=tok
		
	def get(self, w, s, t):
		return self.tokens[(w, s, t)]
		
	def dump(self, s, t):
		for (k,v) in self.tokens.items():
			if k[1]==s and k[2]==t:
				print(k, v)


def getPrime(w, b):
	"w' is w with blank at begin, end and between labels"
	res=str().join([b+c for c in w])+b
	return res


def classIdx(c, classes, blankChar):
	"transform char to index"
	if c==blankChar:
		return len(classes)
	return classes.index(c)


def outputIndices(toks, words, s, t):
	"argmax_w tok(w,s,t)"
	res=[]
	for (wIdx,_) in enumerate(words):
		res.append(toks.get(wIdx, s, t))
		
	idx=[i[0] for i in sorted(enumerate(res), key=lambda x:x[1].score)]
	return idx


def ctcTokenPassing(mat, classes, words, blankChar):
	"implements CTC Token Passing Algorithm as shown by Graves (Dissertation, p67-69)"
	blankIdX=len(classes)
	maxT,maxC=mat.shape

	# special s index for beginning and end of word
	beg=0
	end=-1
	
	# w' in paper: word with blanks in front, back and between labels: for -> _f_o_r_
	primeWords=[getPrime(w, blankChar) for w in words]
	
	# data structure holding all tokens
	toks=TokenList()
	
	# Initialisation: 1-9
	for (wIdx, w) in enumerate(words):
		w=words[wIdx]
		wPrime=primeWords[wIdx]
		
		#set all toks(w,s,t) to init state
		for s in range(len(wPrime)):
			for t in range(maxT):
				toks.set(wIdx, s+1, t+1, Token())
				toks.set(wIdx, beg, t, Token())
				toks.set(wIdx, end, t, Token())
		
		toks.set(wIdx, 1, 1, Token(math.log(mat[1-1,blankIdX]), [wIdx]))
		cIdx=classIdx(w[1-1], classes, blankChar)
		toks.set(wIdx, 2, 1, Token(math.log(mat[1-1,cIdx]), [wIdx]))
	
		if len(w)==1:
			toks.set(wIdx, end, 1, toks.get(wIdx, 2, 1))
		

	# Algorithm: 11-24
	t=2
	while t<=maxT:

		sortedWordIdx=outputIndices(toks, words, end, t-1)
		
		for wIdx in sortedWordIdx:
			wPrime=primeWords[wIdx]
			w=words[wIdx]
			
			# 15-17
			# if bigrams should be used, these lines have to be adapted. 
			bestOutputTok=toks.get(sortedWordIdx[-1], end, t-1)
			toks.set(wIdx, beg, t, Token(bestOutputTok.score, bestOutputTok.history+[wIdx]))
		
			# 18-24
			s=1
			while s<=len(wPrime):
				P=[toks.get(wIdx, s, t-1), toks.get(wIdx, s-1, t-1)]
				if wPrime[s-1]!=blankChar and s>2 and wPrime[s-2-1]!=wPrime[s-1]:
					tok=toks.get(wIdx, s-2, t-1)
					P.append(Token(tok.score, tok.history))

				maxTok=sorted(P, key=lambda x: x.score)[-1]
				cIdx=classIdx(wPrime[s-1], classes, blankChar)
				
				score=maxTok.score+math.log(mat[t-1, cIdx])
				history=maxTok.history
				
				toks.set(wIdx, s, t, Token(score,history))
				s+=1

			canditates=[toks.get(wIdx, len(wPrime), t), toks.get(wIdx, len(wPrime)-1, t)]

			maxTok=sorted([toks.get(wIdx, len(wPrime), t), toks.get(wIdx, len(wPrime)-1, t)], key=lambda x: x.score, reverse=True)[0]
			toks.set(wIdx, end, t, maxTok)
			
		t+=1

	# Termination: 26-28
	bestWIdx=outputIndices(toks, words, end, maxT)[-1]
	return str(' ').join([words[i] for i in toks.get(bestWIdx, end, maxT).history])


class BeamEntry:
	"information about one single beam at specific time-step"
	def __init__(self):
		self.prTotal=0 # blank and non-blank
		self.prNonBlank=0 # blank
		self.prBlank=0 # non-blank
		self.y=() # labellings at current time-step


class BeamState:
	"information about beams at specific time-step"
	def __init__(self):
		self.entries={}

	def norm(self):
		"length-normalise probabilities to avoid penalising long labellings"
		for (k,v) in self.entries.items():
			labellingLen=len(self.entries[k].y)
			self.entries[k].prTotal=self.entries[k].prTotal*(1.0/(labellingLen if labellingLen else 1))

	def sort(self):
		"return beams sorted by probability"
		u=[v for (k,v) in self.entries.items()]
		s=sorted(u, reverse=True, key=lambda x:x.prTotal)
		return [x.y for x in s]


def calcExtPr(k, y, t, mat, beamState, lm, classes):
	"probability for extending labelling y to y+k"
	
	# language model (char bigrams)
	bigramProb=1
	if lm:
		c1=classes[y[-1] if len(y) else classes.index(' ')]
		c2=classes[k]
		lmFactor=0.1 # controls influence of language model
		bigramProb=lm.getCharBigram(c1,c2)**lmFactor

	# optical model (RNN)
	if len(y) and y[-1]==k:
		return mat[t, k]*bigramProb*beamState.entries[y].prBlank
	else:
		return mat[t, k]*bigramProb*beamState.entries[y].prTotal


def addLabelling(beamState, y):
	"adds labelling if it does not exist yet"
	if y not in beamState.entries:
		beamState.entries[y]=BeamEntry()


def ctcBeamSearch(mat, classes, lm):
	"beam search similar to algorithm described by Hwang"
	"Hwang - Character-Level Incremental Speech Recognition with Recurrent Neural Networks"
	
	blankIdX=len(classes)
	maxT,maxC=mat.shape
	beamWidth=25
	
	# Initialise beam state
	last=BeamState()
	y=()
	last.entries[y]=BeamEntry()
	last.entries[y].prBlank=1
	last.entries[y].prTotal=1
	
	# go over all time-steps
	for t in range(maxT):
		curr=BeamState()
		
		# get best labellings
		BHat=last.sort()[0:beamWidth]
		
		# go over best labellings
		for y in BHat:
			prNonBlank=0
			# if nonempty labelling
			if len(y)>0:
				# seq prob so far and prob of seeing last label again
				prNonBlank=last.entries[y].prNonBlank*mat[t, y[-1]]
			
			# calc probabilities
			prBlank=(last.entries[y].prTotal)*mat[t, blankIdX]
			
			# save result
			addLabelling(curr, y)
			curr.entries[y].y=y
			curr.entries[y].prNonBlank+=prNonBlank
			curr.entries[y].prBlank+=prBlank
			curr.entries[y].prTotal+=prBlank+prNonBlank
			
			# extend current labelling
			for k in range(maxC-1):
				newY=y+(k,)
				prNonBlank=calcExtPr(k, y, t, mat, last, lm, classes)
				
				# save result
				addLabelling(curr, newY)
				curr.entries[newY].y=newY
				curr.entries[newY].prNonBlank+=prNonBlank
				curr.entries[newY].prTotal+=prNonBlank
		
		# set new beam state
		last=curr
		
	# normalise probabilities according to labelling length
	last.norm() 
	
	 # sort by probability
	bestLabelling=last.sort()[0] # get most probable labelling
	
	# map labels to chars
	res=''
	for l in bestLabelling:
		res+=classes[l]
		
	return res


def testRealExample():
	"example which decodes a real RNN output. A language model is used."
	classes=' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	blankChar='_'

	rnnOutput=loadRNNOutput('data/rnnOutput.csv')
	rnnOutputSM=softmax(rnnOutput)

	lm=LanguageModel('data/corpus.txt', classes)
	
	print('BEST PATH  :', '"'+ctcBestPath(rnnOutputSM, classes)+'"')
	print('BEAM SEARCH:', '"'+ctcBeamSearch(rnnOutputSM, classes, lm)+'"')
	print('TOKEN      :', '"'+ctcTokenPassing(rnnOutputSM, classes, lm.getWordList(), blankChar)+'"')


def testMiniExample():
	"example which shows difference between taking most probable path and most probable labelling. No language model used."
	classes="ab"
	rnnOutputSM=np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
	
	print('BEST PATH  :', '"'+ctcBestPath(rnnOutputSM, classes)+'"')
	print('BEAM SEARCH:', '"'+ctcBeamSearch(rnnOutputSM, classes, None)+'"')
	

if __name__=='__main__':
	print('=====Mini example=====')
	testMiniExample()
	
	print('=====Real example=====')
	testRealExample()

