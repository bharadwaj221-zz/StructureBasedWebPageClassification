# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:32:34 2014

@author: bharadwaj
"""

import lxml.html
import urllib2
import collections
import hashlib
globalTagset = set([])

def getTagDistribution(givenRoot, globalTagset):
    tagCounts = collections.defaultdict(int)
    for childReference in givenRoot.iter():
        if type(childReference.tag) is str:
            tagCounts[childReference.tag] += 1
            globalTagset.add(childReference.tag)
    return tagCounts
    
html='PLDataset/fetchedPages/0.html'   
doc = lxml.html.fromstring(html)
#doc.make_links_absolute(sys.argv[1])
print doc
print globalTagset
tagDistribution = getTagDistribution(doc, globalTagset)
print tagDistribution

childUrls = getURLs(doc)
            for childUrl in childUrls:
                if "/" + seedDomain + "/" in childUrl:
                    if not childUrl in revisitTest:
                        revisitTest.add(childUrl)
                        urlQueue.append(childUrl)
        sys.stderr.write("Limit:" + str(limit) + "\tProcessed " + url + "\n")
    except Exception as e:
        sys.stderr.write("Limit:" + str(limit) + "\tFailed " + url + " with " + str(e) + "\n")

with open(sys.argv[3], 'w') as outputFile:
    outputFile.write("url")
    for tag in globalTagset:
        outputFile.write("\t" + tag)
    outputFile.write("\n")
    for url, tagDistribution in tagDistributions.items():
        outputFile.write("\"" + url + "\"")
        for tag in globalTagset:
            outputFile.write("\t" + str(tagDistribution[tag]))
        outputFile.write("\n")
            
