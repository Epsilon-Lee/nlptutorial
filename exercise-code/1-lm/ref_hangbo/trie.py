class Node:
    def __init__(self, word):
        self.count = 0
        self.word = word
        self.next = {}

    def add(self, word):
        self.count += 1
        if (word in self.next):
            return self.next[word]
        else:
            node = Node(word)
            self.next[word] = node
            return node

class Trie:
    def __init__(self):
        self.root = Node(None)

    def add(self, sequence):
        p = self.root
        for x in sequence:
            p = p.add(x)
        p.count += 1

    def find(self, sequence):
        p = self.root
        for x in sequence:
            if (x in p.next):
                p = p.next[x]
            else:
                return None

        return p