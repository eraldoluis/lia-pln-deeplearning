# -*- coding: utf-8 -*-
# encoding=utf8
import sys
import re
import json
import numpy as np

def keepIfFound(inFile, outFile, hashtags, averageNum = None, skip1stLine = False, tweetSplitIndex = 1, replaceBy = "__HASHTAG__"):
    """
    Given an array of hashtags to be found, this script will search through a dataset of tweets
    and keep tweets containing at least one of these hashtags as well as using these hashtags
    as classes for the tweets
    """

    with open(inFile) as fileSource, open(outFile, 'w+') as fileResult:
        countLines = 0
        countLinesKept = 0

        hashtags = [tag[0].lower().encode('utf-8') for tag in hashtags]

        countWrites = {}

        #First line of result file shows the array of hashtags searched
        fileResult.write(hashtags.__str__()+"\n")

        if(skip1stLine):
            fileSource.readline()

        for line in fileSource:
            countLines += 1
            resultLine = line[:-1].split("\t", 1)[tweetSplitIndex]

            tagsFound = []
            allowWrite = False

            for token in resultLine.split():
                if token.startswith("#"):
                    token = token.lower()
                    for tag in hashtags:
                        if(token == tag):
                            resultLine = re.compile(token, re.IGNORECASE).sub(replaceBy, resultLine)
                            if(token not in tagsFound):
                                tagsFound.append(tag)

            for tag in tagsFound:
                if averageNum is None or (tag not in countWrites or countWrites[tag] < averageNum):
                    allowWrite = True

            if (len(tagsFound) > 0 and allowWrite):
                tagsText = ""
                for tag in tagsFound:
                    tagsText += tag + " "
                    if(tag in countWrites):
                        countWrites[tag] = countWrites[tag]+1
                    else:
                        countWrites[tag] = 1
                fileResult.write(resultLine + "\t" + tagsText + "\n")
                countLinesKept += 1


    print "Result: " + str(countLinesKept) + " out of " + str(countLines) + " tweets were kept\nResult file: " + str(sys.argv[2])

    # Prints number of tweets kept by hashtag
    for i in hashtags:
        if i in countWrites:
            print i + " = " + str(countWrites[i])

def loadHashtags(frequencyJsonFile, leastFrequency = None, upToPosition = None):
    with open(frequencyJsonFile) as file:
        objs = json.load(file)

    if(upToPosition is not None):
        tags = np.asarray(objs["mostFreq"])[:upToPosition, :]
    else:
        tags = np.asarray(objs["mostFreq"])

    if(leastFrequency is not None):
        temp = []

        for t in tags:
            if(int(t[1]) >= leastFrequency):
                temp.append(t)
            else:
                break
        tags = np.asarray(temp)

    return tags

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')

    keepIfFound(inFile=sys.argv[1],
                outFile=sys.argv[2],
                hashtags=loadHashtags(sys.argv[3], leastFrequency=1000),
                averageNum=1000,
                skip1stLine=True
    )