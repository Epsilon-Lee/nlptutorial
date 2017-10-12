import os
import sys
import math
import codecs
import argparse

from util import *
from trie import *

class NGramLanguageModel:
    def __init__(self, N):
        self.N = N
        self.trieTrees = [Trie() for _ in range(N)]

    def line_to_words(self, line):
        words = line.split()
        if (self.N > 1):
            tmp = [SOS] * (self.N - 1)
            tmp.extend(words)
            words = tmp
        words.append(EOS)

        # words = [w for w in words if len(w) > 0]

        return words

    def train_from_file(self, train_file):
        with codecs.open(filename=train_file, mode='r', encoding="utf-8") as reader:
            for line in reader:
                words = self.line_to_words(line)

                for n in range(self.N):
                    for i in range(0, len(words) - n):
                        self.trieTrees[n].add(words[i: i + n + 1])

        self.vocab_size = len(self.trieTrees[0].root.next)

    def test_on_seq(self, sequence, lam):
        pre_seq = []
        entropy = 0
        pre_point = self.trieTrees[-1].root
        last_p = 1 / self.vocab_size

        for i in range(self.N):
            x = sequence[i]

            if (lam is None):
                pre_point = self.trieTrees[i].find(pre_seq)
                l1 = 1 - len(pre_point.next) / (len(pre_point.next) + pre_point.count) if (pre_point is not None) else 0
            else:
                l1 = lam[i]

            ml = pre_point.next[x].count / pre_point.count if (pre_point is not None and x in pre_point.next) else 0
            last_p = ml * l1 + last_p * (1 - l1)
            entropy += -math.log(last_p, 2)

            pre_seq.append(sequence[i])

        return entropy

    def test_on_file(self, test_file, lam):
        entropy = 0
        total = 0

        with codecs.open(filename=test_file, mode='r', encoding="utf-8") as reader:
            for line in reader:
                words = self.line_to_words(line)

                n = self.N - 1
                for i in range(self.N - n - 1, len(words) - n):
                    entropy += self.test_on_seq(words[i: i + n + 1], lam)
                    total += 1

        return entropy / total

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--trainPath", type=str, default="../../../test/01-train-input.txt")
    # parser.add_argument("--testPath", type=str, default="../../../test/01-test-input.txt")
    parser.add_argument("--trainPath", type=str, default="../../../data/wiki-en-train.word")
    parser.add_argument("--testPath", type=str, default="../../../data/wiki-en-test.word")

    parser.add_argument("--n", type=int, default="2") # n-gram n=1 unigram, n=2 bigram
    parser.add_argument("--lam", type=str, default=None, help="two numbers separated by a hyphen")
    parser.add_argument("--cut_freq", type=int, default="0")

    parse = parser.parse_args()

    train_file = parse.trainPath
    test_file = parse.testPath
    cut_freq = parse.cut_freq
    lams = [float(_) for _ in parse.lam.split('-')] if (parse.lam is not None) else None
    N = parse.n

    lm = NGramLanguageModel(N)
    lm.train_from_file(train_file)
    # lm.vocab_size = 1e6
    E = lm.test_on_file(test_file, lams)
    print("E = {}".format(E))