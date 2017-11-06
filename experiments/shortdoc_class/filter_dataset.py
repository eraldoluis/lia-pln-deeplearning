# -*- coding: utf-8 -*-
import sys
import re

#   Given an array of words (hashtags), this script will search through a dataset and keep the lines
#   in which at least one of the words in the array was found, in the following format:
#   <line>\t<words found separeted by space>

def keepIfFound(inFile, outFile, hashtags):
    with open(inFile) as fileSource, open(outFile, 'w+') as fileResult:
        countLines = 0
        countLinesKept = 0

        hashtags = [tag.lower() for tag in hashtags]

        #First line of result file shows the array of hashtags searshed
        fileResult.write(hashtags.__str__()+"\n")

        for line in fileSource:
            countLines += 1
            resultLine = line[:-1].split("\t", 1)[1]

            tagsFound = []

            for token in resultLine.split():
                if token.startswith("#"):
                    token = token.lower()
                    for tag in hashtags:
                        if(token == tag and token not in tagsFound):
                            tagsFound.append(tag)
            if (len(tagsFound) > 0):
                tagsText = ""
                for tag in tagsFound:
                    tagsText += tag + " "
                fileResult.write(resultLine + "\t" + tagsText + "\n")
                countLinesKept += 1

    print "Result: " + str(countLinesKept) + " out of " + str(countLines) + " lines were kept\nResult file: " + str(sys.argv[2])


if __name__ == '__main__':
    keepIfFound(sys.argv[1],
                sys.argv[2],
                ["#1", "#umrei", "#justin4mmva",
                 "#shawn4mmva", "#nate", "#skype"])
