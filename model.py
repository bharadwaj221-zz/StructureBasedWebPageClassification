import sys
import os
import random
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

#asps = []
#f1=open('ProductPageList.txt','w')
#f2=open('ListingPageList.txt','w')
#for root, dirs, files in os.walk(r'PLDataset/fetchedPages/'):
#    for file in files:
#    	if file.endswith('.html'):
#         [a,b]=file.split('.')
#         if int(a)>1433:
#             f2.write(file);
#             f2.write('\n')
#         else:
#             f1.write(file)
#             f1.write('\n')
#             
             
             
    
                 
         
         



vectorizer = CountVectorizer(stop_words='english',dtype=numpy.float64)
file1=open('ProdPageList.txt');

sys.stdin=file1;
dataset = []
irrDataset = []
for line in sys.stdin:
    flag=0
    if line.startswith('testPages'):
        flag=1
    currFilename = line.strip()
    print currFilename
    with open(r'PLDataset/fetchedPages/'+currFilename) as currFile:
        html = currFile.read()
        #print html
        text = nltk.clean_html(html)
        tokens = nltk.word_tokenize(text)
        tokens = [i.encode('string-escape') for i in tokens]
        lowerCaseTokens = [i.lower() for i in tokens]
        punctuationRemoved = [punctuationRemove.sub("", i) for i in lowerCaseTokens]
        digitsRemoved = [digitsRemove.sub("", i) for i in punctuationRemoved]
        emptyRemoved = [i for i in digitsRemoved if len(i) > 0]
        doc=' '.join(emptyRemoved)
        if flag == 0:        
            dataset.append(doc)
        else:
            irrDataset.append(doc)
        
#shuffle the data , attach test data and take out the last few training samples as test
random.shuffle(dataset)
print('TRAIN DATA:'+str(len(dataset)))
for d in irrDataset:
    dataset.append(d)
print('FULL DATA:'+str(len(dataset)))
dataCOO = vectorizer.fit_transform(dataset)


dataCSR = dataCOO.tocsr()


normalizedData = normalize(dataCSR)



     
denseData=normalizedData.todense()
data=numpy.split(denseData,[1200])
trainSet=data[0];
testSet=data[1];

#==============================================================================
# trainSet, testSet=train_test_split(normalizedData,test_size=0.1)
# testSet=numpy.concatenate((testSet,irrNormalizedData))
# print(testSet)
# print 'training started..'
# clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.2)
# 
# clf.fit(trainSet)
# y=clf.predict(testSet)
numpy.save('trainSet',trainSet)
numpy.save('testSet',testSet)
# #numpy.save('denseData_P',denseData)
#==============================================================================




