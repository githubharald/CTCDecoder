from __future__ import division
from __future__ import print_function
import numpy as np


class BeamEntry:
	"information about one single beam at specific time-step"
	def __init__(self):
		self.prTotal=0 # blank and non-blank
		self.prNonBlank=0 # non-blank
		self.prBlank=0 # blank
		self.y=() # labelling at current time-step


class BeamState:
	"information about beams at specific time-step"
	def __init__(self):
		self.entries={}

	def norm(self):
		"length-normalise probabilities to avoid penalising long labellings"
		for (k,v) in self.entries.items():
			labellingLen=len(self.entries[k].y)
			self.entries[k].prTotal=self.entries[k].prTotal**(1.0/(labellingLen if labellingLen else 1))

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
		lmFactor=0.01 # controls influence of language model
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
	
	blankIdx=len(classes)
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
			prBlank=(last.entries[y].prTotal)*mat[t, blankIdx]
			
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


if __name__=='__main__':
	classes="ab"
	mat=np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
	print('Test beam search')
	expected='a'
	actual=ctcBeamSearch(mat, classes, None)
	print('Expected: "'+expected+'"')
	print('Actual: "'+actual+'"')
	print('OK' if expected==actual else 'ERROR')
