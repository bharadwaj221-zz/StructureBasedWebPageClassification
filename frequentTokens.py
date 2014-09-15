import sys
import nltk
import re
import string
from scikits.learn.feature_extraction.text import CountVectorizer
from scikits.learn import svm
import numpy
from sklearn.preprocessing import Normalizer
from collections import defaultdict

punctuationRemove = re.compile('[%s]' % re.escape(string.punctuation))
digitsRemove = re.compile('[%s]' % re.escape(string.digits))

class DummyAnalyzer(object):
    @staticmethod
    def analyze(s):
        return s

vectorizer = CountVectorizer(analyzer=DummyAnalyzer, dtype=numpy.float64)

frequency = defaultdict(float)
total = 0.0
for line in sys.stdin:
    currFilename = line.strip()
    with open(currFilename) as currFile:
        html = currFile.read()
        text = nltk.clean_html(html)
        tokens = nltk.word_tokenize(text)
        tokens = [i.encode('string-escape') for i in tokens]
        lowerCaseTokens = [i.lower() for i in tokens]
        punctuationRemoved = [punctuationRemove.sub("", i) for i in lowerCaseTokens]
        digitsRemoved = [digitsRemove.sub("", i) for i in punctuationRemoved]
        emptyRemoved = [i for i in digitsRemoved if len(i) > 0]
        localFrequency = defaultdict(float)
        for i in emptyRemoved:
            localFrequency[i] += 1
        for token, count in localFrequency.items():
            frequency[token] += 1
        total += 1

for token, count in sorted(frequency.items(), reverse = True, key = lambda x: x[1]):
    sys.stdout.write(token + "\t" + str(count/total) + "\n")

