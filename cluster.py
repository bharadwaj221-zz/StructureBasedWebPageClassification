import sys
import lxml.html
import urllib2
import collections
import hashlib

def computeChecksum(givenString):
    accumulator = hashlib.md5()
    accumulator.update(givenString)
    return accumulator.hexdigest()

def getTagDistribution(givenRoot, globalTagset):
    tagCounts = collections.defaultdict(int)
    for childReference in givenRoot.iter():
        if type(childReference.tag) is str:
            tagCounts[childReference.tag] += 1
            globalTagset.add(childReference.tag)
    return tagCounts

def getURLs(givenRoot):
    return [i.attrib["href"] for i in givenRoot.xpath("//*[@href]")]

dirtTest = set([])
revisitTest = set([])
globalTagset = set([])
urlQueue = collections.deque()
tagDistributions = {}

seedURL = sys.argv[1]
seedDomain = seedURL.split('/')[2]
limit = int(sys.argv[2])

urlQueue.append(seedURL)

while urlQueue and limit > 0:
    try:
        limit -= 1
        url = urlQueue.pop()
        f=open('PLDataset/fetchedPages/0.html')
        
        response = urllib2.urlopen(url)
        html = response.read()
        html=f.read()
        #pass html file path directly
        doc = lxml.html.fromstring(html)
        checksum = computeChecksum(html)
        if not checksum in dirtTest:
            dirtTest.add(checksum)
            #doc.make_links_absolute(sys.argv[1])
            print doc
            print globalTagset
            tagDistribution = getTagDistribution(doc, globalTagset)
            tagDistributions[url] = tagDistribution
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
            




