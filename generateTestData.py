# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 18:39:12 2014

@author: bharadwaj
"""

import sys
import os
import nltk
import re
import string
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import svm
import numpy




punctuationRemove = re.compile('[%s]' % re.escape(string.punctuation))
digitsRemove = re.compile('[%s]' % re.escape(string.digits))

class DummyAnalyzer(object):
    @staticmethod
    def analyze(s):
        return s

asps = []
f1=open('IrrelevantPageList.txt','w')

for root, dirs, files in os.walk(r'PLDataset/testPages/'):
    for file in files:
        print file
        f1.write(file)
        f1.write('\n')
         
f1.close();
             
punctuationRemove = re.compile('[%s]' % re.escape(string.punctuation))
digitsRemove = re.compile('[%s]' % re.escape(string.digits))

class DummyAnalyzer(object):
    @staticmethod
    def analyze(s):
        return s



vectorizer = CountVectorizer(stop_words='english',dtype=numpy.float64)
file1=open('IrrelevantPageList.txt');
sys.stdin=file1;
dataset = []
for line in sys.stdin:
   
    currFilename = line.strip()
  
    with open(r'PLDataset/testPages/'+currFilename) as currFile:
        html = currFile.read()
        text = nltk.clean_html(html)
        tokens = nltk.word_tokenize(text)
        tokens = [i.encode('string-escape') for i in tokens]
        lowerCaseTokens = [i.lower() for i in tokens]
        punctuationRemoved = [punctuationRemove.sub("", i) for i in lowerCaseTokens]
        digitsRemoved = [digitsRemove.sub("", i) for i in punctuationRemoved]
        emptyRemoved = [i for i in digitsRemoved if len(i) > 0]
        doc=' '.join(emptyRemoved)
        dataset.append(doc)

        
        

dataCOO = vectorizer.fit_transform(dataset)

dataCSR = dataCOO.tocsr()

normalizedData = normalize(dataCSR)
     
irrTestSet=normalizedData.todense()
numpy.save('irrTestSet',irrTestSet);
                  