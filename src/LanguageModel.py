from __future__ import division
from __future__ import print_function
import codecs
import re


class LanguageModel:
	"simple language model: word list for token passing, char bigrams for beam search"
	def __init__(self, fn, classes):
		"read text from file to generate language model"
		self.initWordList(fn)
		self.initCharBigrams(fn, classes)


	def initWordList(self, fn):
		"internal init of word list"
		txt = open(fn).read().lower()
		words = re.findall(r'\w+', txt)
		self.words = list(filter(lambda x: x.isalpha(), words))


	def initCharBigrams(self, fn, classes):
		"internal init of character bigrams"
		self.bigram = {}
		self.numSamples = {}
		txt = codecs.open(fn, 'r', 'utf8').read()

		# init bigrams with 0 values
		for c in classes:
			self.bigram[c] = {}
			self.numSamples[c] = len(classes)
			for d in classes:
				self.bigram[c][d] = 0

		# go through text and create each char bigrams
		for i in range(len(txt)-1):
			first = txt[i]
			second = txt[i+1]

			# ignore unknown chars
			if first not in self.bigram or second not in self.bigram[first]:
				continue

			self.bigram[first][second] += 1
			self.numSamples[first] += 1


	def getCharBigram(self, first, second):
		"probability of seeing character 'first' next to 'second'"
		first = first if first else ' ' # map start to word beginning
		second = second if second else ' ' # map end to word end

		if self.numSamples[first] == 0:
			return 0
		return self.bigram[first][second] / self.numSamples[first]


	def getWordList(self):
		"get list of unique words"
		return self.words
