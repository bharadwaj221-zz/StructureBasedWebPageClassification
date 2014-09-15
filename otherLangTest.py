# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 21:47:53 2014

@author: bharadwaj

"""
import pickle
import numpy
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler

c1=1
c2=2
j=1
#model=pickle.load(open( "RandForModel_"+str(c1)+'vs'+str(c2)+'_'+str(j), "r" ) )    

testFeatures=numpy.loadtxt('COMPLETE_TEST_features500.txt');
testOutput=numpy.loadtxt('COMPLETE_TEST_output.txt')

testx1=testFeatures[testOutput==c1];
testx2=testFeatures[testOutput==c2];

testy1=testOutput[testOutput==c1];
testy1=testy1/max(testy1)
testy2=testOutput[testOutput==c2];
testy2=testy2*0

testX=numpy.append(testx1,testx2,axis=0);
testy=numpy.append(testy1,testy2,axis=0);


p=classifier.predict(scaler.transform(testX))
tac=accuracy_score(testy,p)