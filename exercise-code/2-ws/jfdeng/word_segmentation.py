# -*- coding: utf-8 -*-

"""
Written by jf.deng

Running codes in two ways:
1. First, train a unigram LM, then, do decoding for word segmentation.
2. Load a unigram LM directly, then, do decoding.

Please use **Python2**
"""

import fileinput
import math

SOS = u"<s>"
EOS = u"</s>"
Alpha = .95
N = 1.6 * 10. ** 210
# N = 1.6 * 10. ** 6
INF = 10. ** 15

class LM(object):
	"""Unigram language model"""
	def __init__(self):
		# unigram - the frequency of words in train set
		# unigram_model - the unigram LM
		# sentence - probability of each sentence in test set
		self.unigram = {}
		self.unigram_model = {}
	def loadData(self, path):
		for line in fileinput.input(path):
			line = line.strip().decode("utf-8")
			line = line.split()
			line.insert(0, SOS)
			line.append(EOS)
			yield line
	def loadModel(self, path, ch=None):
		for line in fileinput.input(path):
			try:
				line = line.decode("utf-8")
			except UnicodeError, e:
				print e
			w, p = line.strip().split(ch)
			self.unigram_model[w] = float(p)
	def train(self, path):
		for line in self.loadData(path):
			for w in line:
				self.unigram[w] = self.unigram.get(w, 0) + 1.
		total_wf = sum(self.unigram.values())
		for w in self.unigram:
			# 对于在训练集中出现的词，直接在训练的时候做平滑
			self.unigram_model[w] = Alpha * (self.unigram[w] / total_wf) + (1 - Alpha) / N
	def test(self, path):
		self.sentence = {}
		for line in self.loadData(path):
			pro = 0.
			for w in line:
				if w not in self.unigram_model:
					self.unigram_model[w] = Alpha * 0 + (1 - Alpha) / N
				pro += math.log(self.unigram_model[w], 2)
			self.sentence[" ".join(line)] = pro
	def writeUnigramModel(self, filename):
		self.writeDict(self.unigram_model, filename)
	def writeTestResults(self, filename):
		self.writeDict(self.sentence, filename)
	def writeDict(self, dict, filename):
		with open(filename, "w") as ff:
			for w_or_s, p in sorted(dict.items(), key=lambda x: x[-1], reverse=True):
				ff.write(w_or_s.encode("utf-8")+", "+str(p)+"\n")
###

class Tokenizer(object):
	def __init__(self, lm_model):
		self.lm_model = lm_model
	def loadFile(self, path):
		for line in fileinput.input(path):
			try:
				line = line.decode("utf-8")
			except UnicodeError, e:
				raise
			yield line.strip()
	def ws(self, path, filename, ch="/"):
		ff = open(filename, "w")
		for sentence in self.loadFile(path):
			# Implement Viterbi algorithm
			# Forward
			best_score = [0.] * (len(sentence)+1)
			best_edge = [None] * (len(sentence)+1)
			# 在我们寻找节点j的best_score时
			# 节点0至节点j-1的best_score全部都是已知的
			for j in range(1, len(sentence)+1):
				best_score[j] = INF
				for i in range(0, j):
					# 不允许将整个句子当成一个词
					if len(sentence) == j and 0 == i:
						continue
					word = sentence[i:j]
					# 针对LM中未包含的词做平滑
					if word not in self.lm_model:
						self.lm_model[word] = Alpha * 0 + (1 - Alpha) / N
					pro = self.lm_model[word]
					score = best_score[i] + -1 * math.log(pro, 2)
					if score < best_score[j]:
						best_score[j] = score
						best_edge[j] = (i, j)
			# Backward
			best_path = []
			next_edge = best_edge[-1]
			while next_edge is not None:
				best_path.append(next_edge)
				next_edge = best_edge[next_edge[0]]
			best_path.reverse()
			ff.write(ch.join([sentence[i:j] for i, j in best_path]).encode("utf-8")+"\n")
		ff.close()
###

if __name__ == "__main__":
	lm = LM()
	# 1. train a unigram LM firstly
	lm.train(path=r"people-daily.txt")
	lm.writeUnigramModel(filename="zh-unigram-model.txt")
	# 2. load a unigram LM directly
	lm.loadModel(path="zh-unigram-model.txt", ch=", ")
	tk = Tokenizer(lm.unigram_model)
	tk.ws(path=r"zh-ws-test.txt", filename=r"zh-ws-test-r.txt")
###