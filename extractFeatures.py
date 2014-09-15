import sys
import lxml.html
import collections
import nltk
import re
import string
import os
import itertools


import numpy

from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score

punctuationRemove = re.compile('[%s]' % re.escape(string.punctuation))
digitsRemove = re.compile('[%s]' % re.escape(string.digits))

def constant_factory(value):
    return itertools.repeat(value).next

def parentTag(element):
    if element.getparent() is not None:
        return element.getparent().tag
    return "none"

def depth(element):
    if element.getparent() is None:
        return 0
    return depth(element.getparent()) + 1

def childCount(element):
    return float(len(element.getchildren()))

def countDescendants(node):
    count = 0.0
    for i in node.iterdescendants():
        count += 1
    return count

def getTextCount(node):
    try:
        text = node.text_content().encode("ascii","ignore")
        tokens = nltk.word_tokenize(text)
        tokens = [i.encode('string-escape') for i in tokens]
        lowerCaseTokens = [i.lower() for i in tokens]
        punctuationRemoved = [punctuationRemove.sub("", i) for i in lowerCaseTokens]
        digitsRemoved = [digitsRemove.sub("", i) for i in punctuationRemoved]
        emptyRemoved = [i for i in digitsRemoved if len(i) > 0]
        return float(len(emptyRemoved))
    except:
        return 0.0

def getTagDistribution(htmlFilename):
    with open(htmlFilename) as inputFile: 
        html = inputFile.read().replace("\n","")
    givenRoot = lxml.html.fromstring(html)
    print givenRoot
    tagCounts = collections.defaultdict(float)
    totalDepth = collections.defaultdict(list)
    totalChildren = collections.defaultdict(list)
    totalText = collections.defaultdict(list)
    totalDescendants = collections.defaultdict(list)
    totalTags = 0.0
    maxDepth = 0
    classOccurrence = 0.0
    for childReference in givenRoot.iter():
        if type(childReference.tag) is str:
            tagCounts[childReference.tag] += 1
            currDepth = depth(childReference)
            currChildCount = childCount(childReference)
            maxDepth = max(maxDepth, currDepth)
            currText = getTextCount(childReference)
            currDescendants = countDescendants(childReference)
            totalDepth[childReference.tag].append(currDepth)
            totalChildren[childReference.tag].append(currChildCount)
            totalText[childReference.tag].append(currText)
            totalDescendants[childReference.tag].append(currDescendants)
            totalTags += 1
            if "class" in childReference.attrib.keys():
                classOccurrence += 1
    tagCounts["CLASS_OCCURRENCE"] = classOccurrence
    tagCounts["MAX_DEPTH"] = maxDepth
    for tag in totalDepth:
        meanDepth = sum(totalDepth[tag]) / tagCounts[tag]
        varianceDepth = 1.0/len(totalDepth[tag]) * sum([(meanDepth - v)**2 for v in totalDepth[tag]])
        maxDepth = float(max(totalDepth[tag]))
        minDepth = float(min(totalDepth[tag]))
        tagCounts["MEAN_DEPTH_"+tag] = meanDepth
        tagCounts["VARIANCE_DEPTH_"+tag] = varianceDepth
        tagCounts["MIN_DEPTH_"+tag] = maxDepth
        tagCounts["MAX_DEPTH_"+tag] = minDepth

        meanChildren = sum(totalChildren[tag]) / tagCounts[tag]
        varianceChildren = 1.0/len(totalChildren[tag]) * sum([(meanChildren - v)**2 for v in totalChildren[tag]])
        maxChildren = float(max(totalChildren[tag]))
        minChildren = float(min(totalChildren[tag]))
        tagCounts["MEAN_CHILDREN_"+tag] = meanChildren
        tagCounts["VARIANCE_CHILDREN_"+tag] = varianceChildren
        tagCounts["MIN_CHILDREN_"+tag] = maxChildren
        tagCounts["MAX_CHILDREN_"+tag] = minChildren

        meanText = sum(totalText[tag]) / tagCounts[tag]
        varianceText = 1.0/len(totalText[tag]) * sum([(meanText - v)**2 for v in totalText[tag]])
        maxText = float(max(totalText[tag]))
        minText = float(min(totalText[tag]))
        tagCounts["MEAN_TEXT_"+tag] = meanText
        tagCounts["VARIANCE_TEXT_"+tag] = varianceText
        tagCounts["MIN_TEXT_"+tag] = minText
        tagCounts["MAX_TEXT_"+tag] = maxText

        meanDescendants = sum(totalDescendants[tag]) / tagCounts[tag]
        varianceDescendants = 1.0/len(totalDescendants[tag]) * sum([(meanDescendants - v)**2 for v in totalDescendants[tag]])
        maxDescendants = float(max(totalDescendants[tag]))
        minDescendants = float(min(totalDescendants[tag]))
        tagCounts["MEAN_DESCENDANTS_"+tag] = meanDescendants
        tagCounts["VARIANCE_DESCENDANTS_"+tag] = varianceDescendants
        tagCounts["MIN_DESCENDANTS_"+tag] = minDescendants
        tagCounts["MAX_DESCENDANTS_"+tag] = maxDescendants

    tagCounts["TAGCOUNT"] = totalTags
    text = nltk.clean_html(html)
    tokens = nltk.word_tokenize(text)
    tokens = [i.encode('string-escape') for i in tokens]
    lowerCaseTokens = [i.lower() for i in tokens]
    punctuationRemoved = [punctuationRemove.sub("", i) for i in lowerCaseTokens]
    digitsRemoved = [digitsRemove.sub("", i) for i in punctuationRemoved]
    emptyRemoved = [i for i in digitsRemoved if len(i) > 0]
    tagCounts["TOKENCOUNT"] = float(len(emptyRemoved))
    return tagCounts
    
    getTagDistribution('PLODataset\\neitherProductOrListingPages\\pages\\1.html')

#positiveExamples = [getTagDistribution(sys.argv[1] + "/" + i) for i in os.listdir(sys.argv[1])]
#negativeExamples = [getTagDistribution(sys.argv[2] + "/" + i) for i in os.listdir(sys.argv[2])]
#
#trainingData = []
#trainingData.extend(positiveExamples)
#trainingData.extend(negativeExamples)
#
#DV = DictVectorizer(sparse=False)
#X = DV.fit_transform(trainingData)
#y = numpy.r_[numpy.ones(len(positiveExamples)), numpy.zeros(len(negativeExamples))]
#
#numpy.savetxt('features.txt', X)
#numpy.savetxt('output.txt', y)
