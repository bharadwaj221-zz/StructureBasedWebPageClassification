# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:49:44 2014

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
f1=open('ProductPageList.txt','w')
f2=open('ListingPageList.txt','w')
for root, dirs, files in os.walk(r'PLDataset/fetchedPages/'):
    for file in files:
    	if file.endswith('.html'):
         [a,b]=file.split('.')
         if int(a)>1433:
             f2.write(file);
             f2.write('\n')
         else:
             f1.write(file)
             f1.write('\n')
             
             
             
    
                 
         
         
         
f1.close()
f2.close()


vectorizer = CountVectorizer(stop_words='english',dtype=numpy.float64)
file1=open('ListingPageList.txt');
sys.stdin=file1;
dataset = []
for line in sys.stdin:
   
    currFilename = line.strip()
  
    with open(r'PLDataset/fetchedPages/'+currFilename) as currFile:
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
        
        #dataset.append(emptyRemoved)

dataCOO = vectorizer.fit_transform(dataset)

dataCSR = dataCOO.tocsr()

normalizedData = normalize(dataCSR)
     
normalizedData.todense()
trainSet, testSet=train_test_split(normalizedData,test_size=0.1)

clf2 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
print 'training started..'
clf2.fit(trainSet)
y=clf2.predict(testSet)




