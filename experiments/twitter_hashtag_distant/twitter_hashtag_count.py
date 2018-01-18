from sys import stdin, stdout
import csv
import json

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from codecs import open
import sys

def getTextFromCsv():
    # reader = csv.reader(open("/home/eraldo/lia/src/lia-pln-datasets-models/lucas_sa_twitter/sentiment_hashtags.csv"))
    reader = csv.reader(stdin)
    header = reader.next()
    idxText = header.index("text")
    for line in reader:
        yield line[idxText]


def getFromTextFile(filePath):
    if filePath != "-":
        file = open(filePath, "rt", "utf-8")
    else:
        file = stdin

    for line in file:
        yield line

    if filePath != "-":
        file.close()


def getTextFromES():
    es = Elasticsearch("localhost:9200")
    query = {
        "query": {
            "bool": {
                "filter": [
                    {
                        "terms": {
                            "start": [
                                "2017-02-20T16:33:25.093458-04:00",
                                "2017-02-20T16:33:30.542448-04:00"
                            ]
                        }
                    }
                ]
            }
        }
    }

    for doc in scan(es, query=query):
        try:
            text = doc["_source"]["tweet"]["text"]
            if text is not None:
                yield text
            else:
                stdout.write("x")
        except:
            stdout.write("X")


def main():
    filePath = sys.argv[1]

    maxHashtags = -1
    if len(sys.argv) > 2:
        maxHashtags = int(sys.argv[2])

    numLines = 0
    hashtagHist = {}
    # for tweet in getTextFromCsv():
    for tweet in getFromTextFile(filePath):
        hashtags = set()
        for term in tweet.split():
            if term.startswith("#") and len(term) > 1:
                hashtags.add(term.lower())
        # Increment hashtags counts.
        for hashtag in hashtags:
            count = hashtagHist.get(hashtag, 0)
            count += 1
            hashtagHist[hashtag] = count
        numLines += 1
        if numLines % 10000 == 0:
            if numLines % 100000 == 0:
                stdout.write("%d" % ((numLines / 100000) % 10))
                stdout.flush()
            else:
                stdout.write(".")
                stdout.flush()

    stdout.write(" done!\n")

    sortedItems = sorted(hashtagHist.items(), key=lambda x: x[1], reverse=True)

    for (i, item) in enumerate(sortedItems):
        sortedItems[i] = list(item) + [float(item[1]) / numLines]

    if maxHashtags > 0:
        sortedItems = sortedItems[:maxHashtags]

    print json.dumps(
        {
            "mostFreq": sortedItems,
            # "hist": hashtagHist,
            "countHashtags": len(hashtagHist),
            "countTweets": numLines
        },
        indent=2)


if __name__ == "__main__":
    main()
