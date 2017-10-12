'''
1. construct uni/bi-gram count frequency (ps: each sentence add "<s>" and "</s>")
   1.1 compute unigram frequency
   1.2 compute bigram frequency

2. interface for querying bi-gram
'''
import argparse
import math
from collections import defaultdict

# global variables
N = 100000

parser = argparse.ArgumentParser()
parser.add_argument("-trainPath", type=str, default="../../data/wiki-en-train.word")
parser.add_argument("-testPath", type=str, default="../../data/wiki-en-test.word")
parser.add_argument("-cut_freq", type=int, default="0")

parse = parser.parse_args()

vocab = defaultdict(lambda: 0)

class InterpolatedLanguageModel:
  def __init__(self, filePath, n, cut_freq=0):
    assert filePath is not None, "filePath should be given!"
    self.grams = defaultdict(lambda: 0)
    self.unique = {}
    # self.bigram = defaultdict(lambda: 0)
    self.filePath = filePath
    self.n = n
    self.cut_freq = cut_freq

  # n: means compute 1 to `n` gram freq
  def trainModel(self):
    filePath = self.filePath
    n = self.n
    self.uniTotalFreq = 0
    with open(filePath, 'r', encoding="utf-8") as f:
      for line in f.readlines():
        words = line.split()
        words.append("</s>")
        words.insert(0, "<s>")
        sentLen = len(words)
        self.uniTotalFreq += sentLen
        print words
        for spanLen in range(1, n+1):
          gramLen = spanLen
          for i in range(sentLen - gramLen + 1):
            # compute Witten-bell smoothing params
            if gramLen == 2:
              if not words[i] in self.unique:
                self.unique[words[i]] = {}
              if not words[i + 1] in self.unique[words[i]]:
                self.unique[words[i]][words[i+1]] = 1
              else:
                self.unique[words[i]][words[i+1]] += 1
            self.grams[" ".join(words[i : i + gramLen])] += 1

  # given a `gram`, compute Count(gram)/Count(gram[0:-1])
  def queryFreq(self, gram):
    words = gram.split()
    length = len(words)
    # unigram
    if length == 1:
      if self.grams[gram] > self.cut_freq:
        return self.grams[gram] / self.uniTotalFreq
      else:
        return 1. / N

    # bigram: Witten-bell smoothing
    if length == 2:
      if words[0] not in self.unique:
        return 1. / (N * N)
      else:
        u_w = len(self.unique[words[0]])
      lambda_w = 1. - 1. * u_w / (u_w + self.grams[words[0]])
      print 'lambda_w;', lambda_w, 'u_w:', u_w
      if self.grams[words[1]] > self.cut_freq:
        print 'words[1]:', words[1], 'freq:', self.grams[words[1]]
        return (lambda_w * self.grams[gram] / self.grams[words[0]] + (1 - lambda_w) * self.grams[words[0]] / self.uniTotalFreq) * self.queryFreq(words[0])
      else:
        return 1. / (N * N) * self.queryFreq(words[0])

    # ngram: n > 1
    history = " ".join(words[:-1])
    if self.grams[history] > self.cut_freq:
      return 1. * self.grams[gram] / self.grams[history]
    else:
      return 1. / N

  def computeEntropy(self, testFilePath):
    assert testFilePath is not None
    entropy = .0
    ngram_count = 0
    with open(testFilePath, 'r', encoding="utf-8") as f:
      for line in f.readlines():
        words = line.split()
        words.insert(0, "<s>")
        words.append("</s>")
        length = len(words)
        entropy_per_sentence = .0
        log_sum = .0
        multiply = 1.
        # for word in line.split():
        for i in range(self.n-1, length):
          ngram_count += 1
          gram = " ". join(words[i - self.n + 1 : i + 1])
          prob = self.queryFreq(gram)
          print 'gram:', gram
          print 'prob', prob
          log_sum += math.log(prob, 2)
          # multiply *= prob
        entropy_per_sentence = - multiply * log_sum
        entropy += entropy_per_sentence
    return entropy / ngram_count

  def computeCoverage(self, testFilePath):
    assert testFilePath is not None
    known_word_count = 0
    word_count = 0
    with open(testFilePath, 'r', encoding="utf-8") as f:
      for line in f.readlines():
        words = line.split()
        word_count += len(words)
        for word in words:
          if word in self.vocab:
            known_word_count += 1
    return known_word_count * 1. / word_count

if __name__ == "__main__":
  trainPath = parse.trainPath
  n = 2 # n-gram
  cut_freq = parse.cut_freq

  ngram_lm = InterpolatedLanguageModel(trainPath, n, cut_freq)
  # train the model: just compute freq.
  ngram_lm.trainModel()
  # print ngram_lm.grams, len(ngram_lm.grams)
  # test
  testPath = parse.testPath
  entropy = ngram_lm.computeEntropy(testPath)
  # coverage = unigram_lm.computeCoverage(testPath)
  # print 'Entropy:', entropy, 'Coverage:', coverage
  print 'Entropy:', entropy
