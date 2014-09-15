#import sys
#import lxml.html
#import collections
#import nltk
#import re
#import string
#import os
#import itertools
#import pickle
#
#import numpy
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.feature_extraction import DictVectorizer
#from sklearn import preprocessing
#from sklearn import svm
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import StratifiedKFold
#from sklearn.metrics import roc_curve, auc, accuracy_score
#from sklearn.feature_selection import SelectKBest,chi2
#
#punctuationRemove = re.compile('[%s]' % re.escape(string.punctuation))
#digitsRemove = re.compile('[%s]' % re.escape(string.digits))
#
#def constant_factory(value):
#    return itertools.repeat(value).next
#
#def parentTag(element):
#    if element.getparent() is not None:
#        return element.getparent().tag
#    return "none"
#
#def depth(element):
#    if element.getparent() is None:
#        return 0
#    return depth(element.getparent()) + 1
#
#def childCount(element):
#    return float(len(element.getchildren()))
#
#def countDescendants(node):
#    count = 0.0
#    for i in node.iterdescendants():
#        count += 1
#    return count
#
#def getTextCount(node):
#    try:
#        text = node.text_content().encode("ascii","ignore")
#        tokens = nltk.word_tokenize(text)
#        tokens = [i.encode('string-escape') for i in tokens]
#        lowerCaseTokens = [i.lower() for i in tokens]
#        punctuationRemoved = [punctuationRemove.sub("", i) for i in lowerCaseTokens]
#        digitsRemoved = [digitsRemove.sub("", i) for i in punctuationRemoved]
#        emptyRemoved = [i for i in digitsRemoved if len(i) > 0]
#        return float(len(emptyRemoved))
#    except:
#        return 0.0
#
#def getTagDistribution(htmlFilename):
#    print htmlFilename
#    with open(htmlFilename) as inputFile: 
#        html = inputFile.read().replace("\n","")
#    givenRoot = lxml.html.fromstring(html)
#    tagCounts = collections.defaultdict(float)
#    totalDepth = collections.defaultdict(list)
#    totalChildren = collections.defaultdict(list)
#    totalText = collections.defaultdict(list)
#    totalDescendants = collections.defaultdict(list)
#    totalTags = 0.0
#    maxDepth = 0
#    classOccurrence = 0.0
#    for childReference in givenRoot.iter():
#        if type(childReference.tag) is str:
#            tagCounts[childReference.tag] += 1
#            currDepth = depth(childReference)
#            currChildCount = childCount(childReference)
#            maxDepth = max(maxDepth, currDepth)
#            currText = getTextCount(childReference)
#            currDescendants = countDescendants(childReference)
#            totalDepth[childReference.tag].append(currDepth)
#            totalChildren[childReference.tag].append(currChildCount)
#            totalText[childReference.tag].append(currText)
#            totalDescendants[childReference.tag].append(currDescendants)
#            totalTags += 1
#            if "class" in childReference.attrib.keys():
#                classOccurrence += 1
#    tagCounts["CLASS_OCCURRENCE"] = classOccurrence
#    tagCounts["MAX_DEPTH"] = maxDepth
#    for tag in totalDepth:
#        meanDepth = sum(totalDepth[tag]) / tagCounts[tag]
#        varianceDepth = 1.0/len(totalDepth[tag]) * sum([(meanDepth - v)**2 for v in totalDepth[tag]])
#        maxDepth = float(max(totalDepth[tag]))
#        minDepth = float(min(totalDepth[tag]))
#        tagCounts["MEAN_DEPTH_"+tag] = meanDepth
#        tagCounts["VARIANCE_DEPTH_"+tag] = varianceDepth
#        tagCounts["MIN_DEPTH_"+tag] = maxDepth
#        tagCounts["MAX_DEPTH_"+tag] = minDepth
#
#        meanChildren = sum(totalChildren[tag]) / tagCounts[tag]
#        varianceChildren = 1.0/len(totalChildren[tag]) * sum([(meanChildren - v)**2 for v in totalChildren[tag]])
#        maxChildren = float(max(totalChildren[tag]))
#        minChildren = float(min(totalChildren[tag]))
#        tagCounts["MEAN_CHILDREN_"+tag] = meanChildren
#        tagCounts["VARIANCE_CHILDREN_"+tag] = varianceChildren
#        tagCounts["MIN_CHILDREN_"+tag] = maxChildren
#        tagCounts["MAX_CHILDREN_"+tag] = minChildren
#
#        meanText = sum(totalText[tag]) / tagCounts[tag]
#        varianceText = 1.0/len(totalText[tag]) * sum([(meanText - v)**2 for v in totalText[tag]])
#        maxText = float(max(totalText[tag]))
#        minText = float(min(totalText[tag]))
#        tagCounts["MEAN_TEXT_"+tag] = meanText
#        tagCounts["VARIANCE_TEXT_"+tag] = varianceText
#        tagCounts["MIN_TEXT_"+tag] = minText
#        tagCounts["MAX_TEXT_"+tag] = maxText
#
#        meanDescendants = sum(totalDescendants[tag]) / tagCounts[tag]
#        varianceDescendants = 1.0/len(totalDescendants[tag]) * sum([(meanDescendants - v)**2 for v in totalDescendants[tag]])
#        maxDescendants = float(max(totalDescendants[tag]))
#        minDescendants = float(min(totalDescendants[tag]))
#        tagCounts["MEAN_DESCENDANTS_"+tag] = meanDescendants
#        tagCounts["VARIANCE_DESCENDANTS_"+tag] = varianceDescendants
#        tagCounts["MIN_DESCENDANTS_"+tag] = minDescendants
#        tagCounts["MAX_DESCENDANTS_"+tag] = maxDescendants
#
#    tagCounts["TAGCOUNT"] = totalTags
#    text = nltk.clean_html(html)
#    tokens = nltk.word_tokenize(text)
#    tokens = [i.encode('string-escape') for i in tokens]
#    lowerCaseTokens = [i.lower() for i in tokens]
#    punctuationRemoved = [punctuationRemove.sub("", i) for i in lowerCaseTokens]
#    digitsRemoved = [digitsRemove.sub("", i) for i in punctuationRemoved]
#    emptyRemoved = [i for i in digitsRemoved if len(i) > 0]
#    tagCounts["TOKENCOUNT"] = float(len(emptyRemoved))
#    return tagCounts
#
#productExamples = [getTagDistribution(sys.argv[1] + "/" + i) for i in os.listdir(sys.argv[1])]
#listingExamples = [getTagDistribution(sys.argv[2] + "/" + i) for i in os.listdir(sys.argv[2])]
#otherExamples = [getTagDistribution(sys.argv[3] + "/" + i) for i in os.listdir(sys.argv[3])]
#
#productTests = [getTagDistribution("PLODataset/otherLangPages/Product/" + i) for i in os.listdir("PLODataset/otherLangPages/Product")]
#listingTests = [getTagDistribution("PLODataset/otherLangPages/Listing/" + i) for i in os.listdir("PLODataset/otherLangPages/Listing")]
#otherTests = [getTagDistribution("PLODataset/otherLangPages/Other/" + i) for i in os.listdir("PLODataset/otherLangPages/Other")]
#trainingData = []
#trainingData.extend(productExamples)
#trainingData.extend(listingExamples)
#trainingData.extend(otherExamples)
#
#testData = []
#testData.extend(productTests)
#testData.extend(listingTests)
#testData.extend(otherTests)


#DV = DictVectorizer(sparse=False)
#X = DV.fit_transform(trainingData)
#print shape(X)
#testX=DV.transform(testData)
#print shape(testX)
#pickle.dump(DV,open( "DV", "wb" ))
#print 'saved DV'
#y = numpy.r_[numpy.ones(len(productExamples)), 2*numpy.ones(len(listingExamples)), 3*numpy.ones(len(otherExamples))]
#testY = numpy.r_[numpy.ones(len(productTests)), 2*numpy.ones(len(listingTests)), 3*numpy.ones(len(otherTests))]
#

best=SelectKBest(chi2,100)
Xr=best.fit_transform(X,y)
testXr=best.transform(testX)
pickle.dump(best,open( "best100", "wb" ))
numpy.savetxt('COMPLETE_features100.txt', Xr)
numpy.savetxt('COMPLETE_TEST_features100.txt', testXr)
print 'completed 100'

best=SelectKBest(chi2,300)
Xr=best.fit_transform(X,y)
testXr=best.transform(testX)
pickle.dump(best,open( "best300", "wb" ))
numpy.savetxt('COMPLETE_features300.txt', Xr)
numpy.savetxt('COMPLETE_TEST_features300.txt', testXr)
print 'completed 300'


best=SelectKBest(chi2,500)
Xr=best.fit_transform(X,y)
testXr=best.transform(testX)
pickle.dump(best,open( "best500", "wb" ))
numpy.savetxt('COMPLETE_features500.txt', Xr)
numpy.savetxt('COMPLETE_TEST_features500.txt', testXr)
print 'completed 500'

best=SelectKBest(chi2,1000)
Xr=best.fit_transform(X,y)
testXr=best.transform(testX)
pickle.dump(best,open( "best1000", "wb" ))
numpy.savetxt('COMPLETE_features1000.txt', Xr)
numpy.savetxt('COMPLETE_TEST_features1000.txt', testXr)
print 'completed 100'


numpy.savetxt('COMPLETE_output.txt', y)
numpy.savetxt('COMPLETE_TEST_output.txt', testY)

numpy.savetxt('COMPLETE_ACTUAL_features.txt',X)
numpy.savetxt('COMPLETE_ACTUAL_testfeatures.txt',testX)

cv = StratifiedKFold(y, n_folds=10)


#X=numpy.load('X3');
#y=numpy.load('Y3');
# classifier = svm.SVC(kernel='rbf', probability=True, random_state=0, C = 1, gamma = 10)
# classifier = svm.LinearSVC()
# classifier = RandomForestClassifier(n_estimators=1000, max_features='log2')
#classifier = RandomForestClassifier(n_estimators=1000, max_features=0.0005)
#classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=1000, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=0.0005, verbose=0)
##classifier = RandomForestClassifier(n_estimators=1000, max_features=0.0005)
## classifier = LogisticRegression()
#j=1
#for i, (train, test) in enumerate(cv):
#    scaler = preprocessing.StandardScaler().fit(X[train])
#    classifier.fit(scaler.transform(X[train]), y[train])
#    pickle.dump( classifier, open( "GradBoostModel_LvsO"+str(j), "wb" ) )    
#    j=j+1
#    labTrain = classifier.predict(scaler.transform(X[train]))
#    trainAccuracy = accuracy_score(y[train],labTrain)
#    proTrain = classifier.predict_proba(scaler.transform(X[train]))
#    fprTrain, tprTrain, thresholdsTrain = roc_curve(y[train], proTrain[:, 1])
#    AUCTrain = auc(fprTrain, tprTrain)
#    errorsTrain = numpy.absolute(y[train] - proTrain[:, 1])
#    trainErrorMean = numpy.mean(errorsTrain)
#    trainErrorStd = numpy.std(errorsTrain)
#
#    labTest = classifier.predict(scaler.transform(X[test]))
#    testAccuracy = accuracy_score(y[test],labTest)
#    proTest = classifier.predict_proba(scaler.transform(X[test]))
#    fprTest, tprTest, thresholdsTest = roc_curve(y[test], proTest[:, 1])
#    AUCTest = auc(fprTest, tprTest)
#    errorsTest = numpy.absolute(y[test] - proTest[:, 1])
#    testErrorMean = numpy.mean(errorsTest)
#    testErrorStd = numpy.std(errorsTest)
#
#    print AUCTrain, trainErrorMean, trainErrorStd, trainAccuracy, AUCTest, testErrorMean, testErrorStd, testAccuracy
#    # print trainAccuracy, testAccuracy
