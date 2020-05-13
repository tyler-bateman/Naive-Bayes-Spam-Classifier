from nltk.lm import Vocabulary
import os
import pickle

NONSPAM_DIR = "data/nonspam-train"
SPAM_DIR = "data/spam-train"

def buildVocab(trainFolders):
    words = []
    for folder in trainFolders:
        for filename in os.listdir(os.getcwd() + '/' + folder):
            with open( '{}/{}/{}'.format(os.getcwd(), folder, filename), 'r') as doc:
                words.extend(doc.read().split())
    vocab = Vocabulary(words, unk_cutoff = 1)
    print(len(vocab), 'unique words')

    while len(vocab) > 2600:
        vocab._cutoff += 1

    print('dictionary size:', len(vocab))
    return vocab

#Precondition: fv is is dict which has the same key set as vocab
def processDoc(d, vocab, fv):
    for feature in vocab.lookup(d.read().split()):
        if feature != '<UNK>':
            fv[feature] += vocab[feature]
    return fv

def getFeatureVector(vocab, spam):
    dir = SPAM_DIR if spam else NONSPAM_DIR
    fv = dict.fromkeys((t for t in vocab), 0)
    for fname in os.listdir(os.getcwd() + '/' + dir):
        with open( '{}/{}/{}'.format(os.getcwd(), dir, fname), 'r') as doc:
            processDoc(doc, vocab, fv)


v = buildVocab((NONSPAM_DIR, SPAM_DIR))

getFeatureVector(v, True)
