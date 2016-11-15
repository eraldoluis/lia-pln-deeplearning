#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import sys

from numpy import random


def pickUpSentences(fileFrom, fileTo, numSentences):
    """

    :param fileFrom: file which the sentences will be picked
    :param fileTo: file which the picked sentences will be written
    :param numSentences: number of sentences to be picked
    :return:
    """
    lines = codecs.open(fileFrom, "r", encoding="utf-8").readlines()
    f = codecs.open(fileTo, "w", encoding="utf-8")

    linesNumber = [i for i in xrange(len(lines))]

    random.shuffle(linesNumber)

    for i, l in enumerate(linesNumber):
        if i == numSentences:
            break
        f.write(lines[l])

if __name__ == '__main__':
    pickUpSentences(sys.argv[1], sys.argv[2], int(sys.argv[3]))
