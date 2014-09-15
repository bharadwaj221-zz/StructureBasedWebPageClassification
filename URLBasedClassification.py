# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 16:56:47 2014

@author: bharadwaj
"""

import numpy
from urlparse import urlparse
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score

URLs=[]
productURLs=[]
list=open('PLODataset/productPages/fetcher.log','r')
for line in list:
       words=line.split()
       if words[0] == 'SUCCESS':
           #print words[0],words[1]
           url=words[1]
           r=urlparse(url)
           cleanURL=''
          
               
           for e in r.path+' '+r.query:
               if e.isalpha():
                   cleanURL+=e
               else:
                   cleanURL+=(' ')
                   
           productURLs.append(cleanURL)     
           URLs.append(cleanURL) 
           
           
#listingURLs=[]
#list=open('PLODataset/listingPages/fetcher.log','r')
#for line in list:
#       words=line.split()
#       if words[0] == 'SUCCESS':
#           #print words[0],words[1]
#           url=words[1]
#           r=urlparse(url)
#           cleanURL=''
#          
#               
#           for e in r.path+' '+r.query:
#               if e.isalpha():
#                   cleanURL+=e
#               else:
#                   cleanURL+=(' ')
#                   
#           listingURLs.append(cleanURL) 
#           URLs.append(cleanURL) 

           
otherURLs=[]
list=open('PLODataset/neitherProductOrListingPages/fetcher.log','r')
for line in list:
       words=line.split()
       if words[0] == 'SUCCESS':
           #print words[0],words[1]
           url=words[1]
           r=urlparse(url)
           cleanURL=''
          
               
           for e in r.path+' '+r.query:
               if e.isalpha():
                   cleanURL+=e
               else:
                   cleanURL+=(' ')
                   
           otherURLs.append(cleanURL)
           URLs.append(cleanURL) 
#
#pickle.dump(productURLs,open('productURLs','wb'))
#pickle.dump(listingURLs,open('listingURLs','wb'))
#pickle.dump(otherURLs,open('otherURLs','wb'))
#pickle.dump(URLs,open('allURLs','wb'))

URLs=pickle.load(open('allURLs','r'))
vec=TfidfVectorizer(input='content',ngram_range=(10,11),stop_words='english')                   
features=vec.fit_transform(URLs)
output=numpy.loadtxt('COMPLETE_output.txt')
best=SelectKBest(chi2,500)
Xr=best.fit_transform(features,output)

pickle.dump(best,open( "urlbest200", "wb" ))
#numpy.savetxt('urlfeatures.txt',Xr)

print 'completed 100'

c1=1
c2=3



y1=output[output==c1];
y1=y1/max(y1)
y2=output[output==c2];
y2=y2*0


y=numpy.append(y1,y2,axis=0);











X=Xr




cv = StratifiedKFold(y, n_folds=5)
classifier=LogisticRegression()

for i, (train, test) in enumerate(cv):
    print i
    classifier.fit(X[train], y[train])
    pickle.dump( classifier, open( "URL_LvsO"+str(i+1), "wb" ) )    
    
    labTrain = classifier.predict(X[train])
    trainAccuracy = accuracy_score(y[train],labTrain)
    
    labTest = classifier.predict(X[test])
    cm= classifier.decision_function(X[test])
    testAccuracy = accuracy_score(y[test],labTest)
    
    print trainAccuracy, testAccuracy
    # print trainAccuracy, testAccuracy
  
  



           
