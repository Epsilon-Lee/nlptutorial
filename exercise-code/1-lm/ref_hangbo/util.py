import codecs

UNK = "<unk>"
EOS = "</s>" # end of sequence
SOS = "<s>" # start of sequence [\233]

def loadfile(filename):
    dictcounter = {}
    sum = 0
    with codecs.open(filename=filename, mode='r') as reader:
        for line in reader:
            words = line.strip().split(' ')

            words.append(EOS)
            for word in words:
                if (len(word) == 0):
                    continue

                sum += 1

                if (word not in dictcounter):
                    dictcounter[word] = 1
                else:
                    dictcounter[word] += 1

    for w in dictcounter:
        dictcounter[w] /= sum

    return dictcounter