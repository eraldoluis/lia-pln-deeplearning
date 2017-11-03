import sys
import re

#   Given an array of words (hashtags), this script will search through a dataset and keep the lines
#   in which at least one of the words in the array was found, in the following format:
#   <line>\t<words found separeted by space>

def keepIfFound(inFile, outFile, hashtags):
    with open(inFile) as fileSource, open(outFile, 'w+') as fileResult:
        linesKept = 0
        for line in fileSource:
            #the following line removes the "id" of the tweet in the dataset
            resultLine = line[:-1].split("\t",1)[1]
            tagsFound = ""

            for tag in hashtags:
                if(re.search(tag, resultLine) != None):
                    tagsFound += " " + tag
            if(tagsFound != ""):
                fileResult.write(resultLine+"\t"+tagsFound+"\n")
                linesKept += 1
    print "DONE!\nLines kept: " + linesKept


if __name__ == '__main__':
    keepIfFound(sys.argv[1],
                sys.argv[2],
                ["#Hashtag1", "#Hashtag2", "..."])
