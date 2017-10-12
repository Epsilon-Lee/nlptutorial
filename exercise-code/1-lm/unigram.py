'''
1. construct vocab. count frequency
2. interface for querying uni-gram
'''
import argparse
import math
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("-trainPath", type=str, default="../../data/wiki-en-train.word")
parser.add_argument("-testPath", type=str, default="../../data/wiki-en-test.word")
parser.add_argument("-cut_freq", type=int, default="0")

parse = parser.parse_args()

vocab = defaultdict(lambda: 0)

class LanguageModel:
    def __init__(self, filePath, cut_freq=0):
        assert filePath is not None, "filePath should be given!"
        self.vocab = defaultdict(lambda: 0)
        self.filePath = filePath
        self.cut_freq = cut_freq

    def createVocab(self):
        filePath = self.filePath
        with open(filePath, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                words = line.split()
                words.append("</s>")
                for word in words:
                    self.vocab[word] += 1
        self.totolFreq = 0
    for word, freq in self.vocab.iteritems():
            if freq > cut_freq:
                self.totolFreq += freq

    def queryFreq(self, word):
        if self.vocab[word] > self.cut_freq:
            prob = self.vocab[word] * 1.0 / self.totolFreq
            return self.vocab[word], prob
        else:
            # return self.vocab[word], .0
      return self.vocab[word], 1./len(self.vocab)

    def computeEntropy(self, testFilePath):
        assert testFilePath is not None
        entropy = .0
        word_count = 0
        with open(testFilePath, 'r') as f:
            for line in f.readlines():
                entropy_per_sentence = .0
                log_sum = .0
                multiply = 1.

                words = line.split()
                words.append("</s>")

                for word in words:
                    word_count += 1
                    _, prob = self.queryFreq(word)
                    ans = prob * 0.95 + 0.05 / len(self.vocab)
                    log_sum += math.log(ans, 2)
                    # multiply *= prob
                entropy_per_sentence = -multiply * log_sum
                entropy += entropy_per_sentence
        return entropy / word_count

    def computeCoverage(self, testFilePath):
        assert testFilePath is not None
        known_word_count = 0
        word_count = 0
        with open(testFilePath, 'r') as f:
            for line in f.readlines():
                words = line.split()
                word_count += len(words)
                for word in words:
                    if word in self.vocab:
                        known_word_count += 1
        return known_word_count * 1. / word_count

if __name__ == "__main__":
    trainPath = parse.trainPath
    cut_freq = parse.cut_freq
    unigram_lm = LanguageModel(trainPath, cut_freq)
    # train the model: just compute freq.
    unigram_lm.createVocab()

    # test
    testPath = parse.testPath
    entropy = unigram_lm.computeEntropy(testPath)
    coverage = unigram_lm.computeCoverage(testPath)
  print 'Entropy:', entropy, 'Coverage:', 1 - coverage