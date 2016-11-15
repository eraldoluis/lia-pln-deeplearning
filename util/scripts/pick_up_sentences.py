#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import sys

from numpy import random


def pickRandomlySentences(fileFrom, fileTo, numLines):
    lines = codecs.open(fileFrom, "r", encoding="utf-8").readlines()
    f = codecs.open(fileTo, "w", encoding="utf-8")

    linesNumber = [i for i in xrange(len(lines))]

    random.shuffle(linesNumber)

    for i, l in enumerate(linesNumber):
        if i == numLines:
            break
        f.write(lines[l])


if __name__ == '__main__':
    pickRandomlySentences(sys.args[1], sys.args[2], sys.args[3])
