from math import log, exp
from nltk.lm import Vocabulary
import pickle
import os

V = Vocabulary(pickle.load(open('obj/vocab.p', 'rb')))
SPAM_CTS = dict(pickle.load(open('obj/spam_cts.p', 'rb')))
NONSPAM_CTS = dict(pickle.load(open('obj/nonspam_cts.p', 'rb')))



def bagOfWords(d):
    bow = dict.fromkeys((t for t in V), 0)
    for word in d.read().split():
        if word in bow:
            bow[word] += 1
            #print(word)
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
    spamTot = log(sum(SPAM_CTS[t] for t in SPAM_CTS) + len(V))
    nonspamTot = log(sum(NONSPAM_CTS[t] for t in NONSPAM_CTS) + len(V))
    for word, count in bow.items():
        if count > 0:
            nsProb += log(count * (NONSPAM_CTS[word] + 1)) - nonspamTot
            sProb += log(count * (SPAM_CTS[word] + 1)) - spamTot

    return exp(sProb) > exp(nsProb)

def classifyDocs(dir):
    sCount = 0;
    nsCount = 0;
    for fname in os.listdir(os.getcwd() + '/' + dir):
        with open('{}/{}/{}'.format(os.getcwd(), dir, fname), 'r') as doc:
            if isSpam(doc):
                sCount += 1
            else:
                nsCount += 1
    return (sCount, nsCount)

def recall(tp, tn, fn):
    return tp/(tp + fn)

def precision(tp, tn, fp):
    return tp/(tp + fp)

def f1score(p, r):
    return 2 * p * r / (p + r)


def generateReport(trainSize, output):
    global V, SPAM_CTS, NONSPAM_CTS
    V = Vocabulary(pickle.load(open('obj/vocab_{}.p'.format(str(trainSize)), 'rb')))
    SPAM_CTS = dict(pickle.load(open('obj/spam_cts_{}.p'.format(str(trainSize)), 'rb')))
    NONSPAM_CTS = dict(pickle.load(open('obj/nonspam_cts_{}.p'.format(str(trainSize)), 'rb')))

    (tp, fn) = classifyDocs('data/spam-test')
    (fp, tn) = classifyDocs('data/nonspam-test')
    r = recall(tp, tn, fn)
    p = precision(tp, tn, fp)
    f1 = f1score(p, r)
    output.write('\nResults for model trained on {} documents:\n'.format(trainSize))
    output.write('True positives: ' + str(tp) + '\n')
    output.write('False negatives: ' + str(fn) + '\n')
    output.write('True negatives: ' + str(tn) + '\n')
    output.write('False positives: ' + str(fp) + '\n')
    output.write('Precision: ' + str(p) + '\n')
    output.write('Recall: ' + str(r) + '\n')
    output.write('F score: ' + str(f1) + '\n')

def main():
    output = open('report/output.txt', 'w')
    for n in (50, 100, 400, 700):
        generateReport(n, output)


if __name__=='__main__':
    main()
