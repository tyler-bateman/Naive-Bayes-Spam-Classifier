from math import log, exp
from nltk.lm import Vocabulary
import pickle
import os

V = Vocabulary(pickle.load(open('obj/vocab.p', 'rb')))
SPAM_CTS = dict(pickle.load(open('obj/spam_cts.p', 'rb')))
NONSPAM_CTS = dict(pickle.load(open('obj/nonspam_cts.p', 'rb')))

print(type(SPAM_CTS))


def bagOfWords(d):
    bow = dict.fromkeys((t for t in V), 0)
    for word in d.read().split():
        if word in bow:
            bow[word] += 1
    return bow


# Classifies the given document as spam or not spam
# Returns True if it believes it to be spam,
# False otherwise
def isSpam(d):
    global V, SPAM_CTS, NONSPAM_CTS
    sProb = 0
    nsProb = 0

    bow = bagOfWords(d)
    #The training data had an equal number of spam and nonspam documents so the
    #initial probability of a given classification can be ignored
    for word in bow:
        sProb += log(SPAM_CTS[word] + 1) - log(sum(SPAM_CTS[t] for t in SPAM_CTS) + len(V))
        nsProb += log(NONSPAM_CTS[word] + 1) - log(sum(NONSPAM_CTS[t] for t in NONSPAM_CTS) + len(V))

    return exp(sProb) > exp(nsProb)

def classifyDocs(dir):
    sCount = 0;
    nsCount = 0;
    for fname in os.listdir(os.getcwd() + '/' + dir):
        print('Classifying', fname)
        with open('{}/{}/{}'.format(os.getcwd(), dir, fname), 'r') as doc:
            if isSpam(doc):
                print('spam')
                sCount += 1
            else:
                print('not spam')
                nsCount += 1
    return (sCount, nsCount)

def main():
    (spam, nonspam) = classifyDocs('data/spam-test')
    print('True positives:', spam)
    print('False positives:', nonspam)
    (spam, nonspam) = classifyDocs('data/nonspam-test')
    print('True negatives:', nonspam)
    print('False negatives:', spam)

if __name__=='__main__':
    main()
