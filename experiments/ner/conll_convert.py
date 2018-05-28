# coding=utf-8
import sys

from __builtin__ import open

labelsFilename = "/home/eraldo/lia/ner/labels.txt" #sys.argv[0]
corpusFilename = "/home/eraldo/lia/ner/first.txt_dev.txt" #sys.argv[1]
predictionFilename = "/home/eraldo/lia/ner/epoch69.txt" #sys.argv[2]

from data.Lexicon import Lexicon

labels = Lexicon.fromTextFile(labelsFilename, hasUnknowSymbol=False)

corpusFile = open(corpusFilename)

predictionFile = open(predictionFilename)
prediction = [int(s) for s in predictionFile.readline().strip()[:-1].split(',')]
predictionFile.close()

i = 0
for line in corpusFile:
    ftrs = line.strip().split('\t')
    if len(ftrs[1]) > 1:
        ftrs[1] = 'I-' + ftrs[1][:3]
    pred = labels.getLexicon(prediction[i])
    if len(pred) > 1:
        pred = 'I-' + pred[:3]
    print "{0} POS {1} {2}".format(ftrs[0], ftrs[1], pred)
    i += 1

corpusFile.close()
