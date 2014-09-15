# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:02:24 2014

@author: bharadwaj
"""

import sklearn
import numpy
import string
import pickle
from sklearn import svm

def findValues(y,t,label):
    tp=0
    tn=0    
    fp=0
    fn=0
    for i in range(0,len(y)):
        if y[i] == t[i]:
            if y[i] == label:
                tp=tp+1
            else:
                tn=tn+1
                
        else:
            if y[i] == label:
                fp=fp+1
            else:
                fn=fn+1
    return tp,tn,fp,fn
            
  

testSet=numpy.load('testSet.npy')
trainSet=numpy.load('trainSet.npy')
target1=numpy.ones(183-50);
target2=-1*numpy.ones(50);
t=numpy.concatenate((target1,target2))
numpy.save('target1',target1)
numpy.save('target2',target2)


#for i in range(1,2) :
#    print i
#    p1=float(i)/10
#    print p1
#    clf = svm.OneClassSVM(nu=p1, kernel="rbf", gamma=0.2)
#    clf.fit(trainSet)
#    y=clf.predict(testSet)
#    vals1=sklearn.metrics.precision_recall_fscore_support(t,y,pos_label=1,average='micro')  
#    vals2=sklearn.metrics.precision_recall_fscore_support(t,y,pos_label=-1,average='micro') 
#    print vals1
#    print vals2
#    numpy.save('model_results'+str(i),(vals1,vals2))
    
results=[]
for i in range(5,6) :
    print i
    p2=float(i)/10
    print p2
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)
    clf.fit(trainSet)
    y=clf.predict(testSet)
    vals1=sklearn.metrics.precision_recall_fscore_support(t,y,pos_label=1,average='micro')  
    vals2=sklearn.metrics.precision_recall_fscore_support(t,y,pos_label=-1,average='micro') 
    results.append((p2,vals1,vals2))
    
    print(findValues(y,t,1))
    

# results2=[]
# for i in range(41,59) :
#     print i
#     p2=float(i)/100
#     print p2
#     clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=p2)
#     clf.fit(trainSet)
#     y=clf.predict(testSet)
#     vals1=sklearn.metrics.precision_recall_fscore_support(t,y,pos_label=1,average='micro')  
#     vals2=sklearn.metrics.precision_recall_fscore_support(t,y,pos_label=-1,average='micro') 
#     results2.append((p2,vals1,vals2))
#     
#     print(findValues(y,t,1))
           
pickle.dump( results, open( "results", "wb" ) )    


 