# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:42:40 2014

@author: bharadwaj
"""

from urlparse import urlparse
from nltk.util import ngrams
#url='https://apps.facebook.com/candycrush/?fb_source=bookmark&ref=bookmarks&count=0&fb_bmpos=2_0'
#r=urlparse(url)
#s=''
#
#for e in r.query:
#    if e.isalpha():
#       s+=e
#    else:
#        s+=(' ')
#            
#gram=[];
#N=ngrams(s,3)
#for g in N:
#    str='';
#    for i in g:
#        str+=i
#    gram.append(str);
#    print gram
   
import numpy

X=numpy.load('X_complete.npy')
