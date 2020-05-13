from math import log, exp
from nltk.lm import Vocabulary
import pickle


V = pickle.load(open('dir/vocab.p', 'rb'))
SPAM_CTS = pickle.load(open('dir/spam_cts.p', 'rb'))
NONSPAM_CTS = pickle.load(open('dir/nonspam_cts.p', 'rb'))


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
    sProb = 0
    nsProb = 0

    bow = bagOfWords(d)
    #The training data had an equal number of spam and nonspam documents so the
    #initial probability of a given classification can be ignored
    for word in bow:
        sProb += log(SPAM_CTS[word] + 1) - log(sum(SPAM_CTS[t] for t in SPAM_CTS) + len(V))
        nsProb += log(NONSPAM_CTS[word] + 1) - log(sum(SPAM_CTS[t] for t in SPAM_CTS) + len(V))

    return exp(sProb) > exp(nsProb)
