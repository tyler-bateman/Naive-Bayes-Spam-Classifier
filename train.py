from nltk.lm import Vocabulary
import os
import pickle

NONSPAM_DIR = "data/nonspam-train"
SPAM_DIR = "data/spam-train"

#Precondition: numDocs must be less than or equal to the number of training docs in the train folders
def buildVocab(trainFolders, numDocs):
    words = []
    for folder in trainFolders:
        files =  os.listdir(os.getcwd() + '/' + folder)
        for i in range(numDocs // 2):
            with open('{}/{}/{}'.format(os.getcwd(), folder, files[i]), 'r') as doc:
                words.extend(doc.read().split())
    vocab = Vocabulary(words, unk_cutoff = 1)
    print(len(vocab), 'unique words')

    while len(vocab) > 3000:
        vocab._cutoff += 1

    print('dictionary size:', len(vocab))
    return vocab

#Precondition: fv is is dict which has the same key set as vocab
def processDoc(d, vocab, fv):
    for feature in vocab.lookup(d.read().split()):
        if feature != '<UNK>':
            fv[feature] += vocab[feature]
    return fv

#Precondition: numDocs <= number of training files
def getFeatureVector(vocab, spam, numDocs):
    dir = SPAM_DIR if spam else NONSPAM_DIR
    fv = dict.fromkeys((t for t in vocab), 0)
    files = os.listdir(os.getcwd() + '/' + dir)
    for i in range(numDocs // 2):
        with open('{}/{}/{}'.format(os.getcwd(), dir, files[i]), 'r') as doc:
            processDoc(doc, vocab, fv)
    return fv

def save(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)

def train(numDocs):
    v = buildVocab((NONSPAM_DIR, SPAM_DIR), numDocs)
    s_vect = getFeatureVector(v, True, numDocs)
    ns_vect = getFeatureVector(v, False, numDocs)
    save(v, 'obj/vocab_{}.p'.format(str(numDocs)))
    save(ns_vect, 'obj/nonspam_cts_{}.p'.format(str(numDocs)))
    save(s_vect, 'obj/spam_cts_{}.p'.format(str(numDocs)))

def main():
    for n in (50, 100, 400, 700):
        train(n)
    print('done')

if __name__ == '__main__':
    main()
